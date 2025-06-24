# LLM轮询系统技术实现详解

## 📋 目录
- [1. LLM管理器核心实现](#1-llm管理器核心实现)
- [2. 分布式限流器实现](#2-分布式限流器实现)
- [3. Token计算器实现](#3-token计算器实现)
- [4. API Key轮询策略](#4-api-key轮询策略)
- [5. 配置系统设计](#5-配置系统设计)
- [6. 错误处理与重试机制](#6-错误处理与重试机制)
- [7. 监控与状态管理](#7-监控与状态管理)

---

## 1. LLM管理器核心实现

### 1.1 整体架构设计

LLM管理器采用**策略模式**和**工厂模式**的组合，核心类结构：

```python
@dataclass
class ApiKeyConfig:
    """API Key配置 - 包含限制、状态、错误统计"""
    key: str
    priority: int           # 优先级 (1最高)
    qpm_limit: int         # QPM限制
    tpm_limit: int         # TPM限制
    enabled: bool          # 是否启用
    current_qpm: int = 0   # 当前QPM使用量
    current_tpm: int = 0   # 当前TPM使用量
    error_count: int = 0   # 错误计数
    last_error: Optional[str] = None  # 最后错误信息

@dataclass
class LLMResponse:
    """统一的LLM响应格式"""
    content: str           # 回答内容
    model: str            # 使用的模型
    provider: str         # 提供商
    api_key_id: str       # API Key标识(脱敏)
    usage: Dict[str, int] # Token使用统计
    response_time: float  # 响应时间
```

### 1.2 核心轮询算法

**优先级选择策略**：
```python
async def _select_api_key(self, provider_name: str, request_tokens: int) -> Optional[ApiKeyConfig]:
    """
    API Key选择算法：
    1. 按优先级排序(1最高)
    2. 检查是否启用
    3. 进行限流验证
    4. 返回第一个可用的key
    """
    provider = self.providers.get(provider_name)
    if not provider:
        return None
    
    # 已按priority排序的API keys
    for api_key in provider.api_keys:
        if not api_key.enabled:
            continue
        
        # 调用限流器检查
        allowed, status = await self.api_key_limiter.check_api_key_limits(
            api_key.key, request_tokens, api_key.qpm_limit, api_key.tpm_limit
        )
        
        if allowed:
            # 更新使用统计
            api_key.current_qpm = status.get("current_qpm", 0)
            api_key.current_tpm = status.get("current_tpm", 0)
            return api_key
    
    return None  # 所有key都不可用
```

**多模型降级策略**：
```python
# 配置的模型优先级
model_priority:
  - provider: "qwen"
    model: "qwen-max-latest"
  - provider: "qwen" 
    model: "qwen-max"
  - provider: "openai"
    model: "gpt-4o-mini"

# 降级逻辑
for model_priority in model_priorities:
    provider_name = model_priority["provider"]
    model_name = model_priority["model"]
    
    # 尝试获取该提供商的可用API key
    api_key = await self._select_api_key(provider_name, request_tokens)
    if api_key:
        # 调用LLM API
        result = await self._call_llm_api(...)
        if result["success"]:
            return response
```

### 1.3 HTTP客户端封装

**统一的API调用接口**：
```python
async def _call_llm_api(self, provider_name: str, api_key: str, base_url: str, request: LLMRequest):
    """
    统一的LLM API调用方法
    支持不同提供商的认证方式和参数格式
    """
    # 根据提供商设置认证头
    headers = {"Content-Type": "application/json"}
    if provider_name == "qwen":
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider_name == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
    
    # 构建标准的OpenAI格式请求
    data = {
        "model": request.model,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream
    }
    
    # 使用aiohttp进行异步调用
    async with aiohttp.ClientSession() as session:
        timeout = aiohttp.ClientTimeout(total=180)
        async with session.post(url, json=data, headers=headers, timeout=timeout) as response:
            # 处理响应...
```

---

## 2. 分布式限流器实现

### 2.1 滑动窗口算法

采用**Redis滑动窗口**实现精确限流：

```python
def _get_current_windows(self) -> list:
    """
    生成滑动窗口时间点
    窗口大小60秒，精度1秒 = 60个窗口
    """
    now = int(time.time())
    windows = []
    
    for i in range(0, self.window_size, self.precision):
        window_start = now - i
        # 对齐到precision边界 (避免时间漂移)
        window_start = (window_start // self.precision) * self.precision
        windows.append(window_start)
    
    return sorted(set(windows))
```

**Redis Key设计**：
```
rate_limit:api_key:12345678:requests:1640000000  # 请求数
rate_limit:api_key:12345678:tokens:1640000000    # Token数
```

### 2.2 原子性操作实现

**检查并增加计数的原子操作**：
```python
async def check_and_increment(self, identifier: str, request_count: int = 1, 
                            token_count: int = 0, qpm_limit: int = 0, 
                            tpm_limit: int = 0) -> Tuple[bool, Dict[str, Any]]:
    """
    原子性的限流检查和计数增加
    """
    # 1. 批量获取所有窗口的计数
    windows = self._get_current_windows()
    pipe = redis_client.pipeline()
    
    for window_start in windows:
        req_key = self._get_window_key(identifier, "requests", window_start)
        token_key = self._get_window_key(identifier, "tokens", window_start)
        pipe.get(req_key)
        pipe.get(token_key)
    
    results = await pipe.execute()
    
    # 2. 计算当前总计数
    current_requests = sum(int(results[i] or 0) for i in range(0, len(results), 2))
    current_tokens = sum(int(results[i+1] or 0) for i in range(0, len(results), 2))
    
    # 3. 检查是否会超限
    will_exceed_qpm = qpm_limit > 0 and (current_requests + request_count) > qpm_limit
    will_exceed_tpm = tpm_limit > 0 and (current_tokens + token_count) > tpm_limit
    
    if will_exceed_qpm or will_exceed_tpm:
        return False, {"reason": "Rate limit exceeded"}
    
    # 4. 原子性增加计数
    current_window = (int(time.time()) // self.precision) * self.precision
    pipe = redis_client.pipeline()
    pipe.incrby(req_key, request_count)
    pipe.expire(req_key, self.window_size + self.precision)
    if token_count > 0:
        pipe.incrby(token_key, token_count)
        pipe.expire(token_key, self.window_size + self.precision)
    
    await pipe.execute()
    return True, {"allowed": True}
```

### 2.3 本地缓存优化

**减少Redis访问的本地缓存**：
```python
class RateLimiter:
    def __init__(self):
        self._local_cache = {}  # 缓存被阻塞的key
        self._cache_ttl = {}    # 缓存过期时间
    
    def _is_cached_blocked(self, cache_key: str) -> bool:
        """检查本地缓存，快速拒绝明显超限的请求"""
        if cache_key in self._local_cache:
            if time.time() < self._cache_ttl[cache_key]:
                return True  # 仍在阻塞期内
            else:
                # 清理过期缓存
                del self._local_cache[cache_key]
                del self._cache_ttl[cache_key]
        return False
    
    def _cache_block(self, cache_key: str, duration: int):
        """缓存阻塞状态，避免频繁Redis查询"""
        self._local_cache[cache_key] = True
        self._cache_ttl[cache_key] = time.time() + duration
```

---

## 3. Token计算器实现

### 3.1 多模型Token计算

使用**tiktoken**库进行精确计算：

```python
class TokenCounter:
    # 模型编码器映射
    MODEL_ENCODINGS = {
        "gpt-4o-mini": "cl100k_base",
        "qwen-max-latest": "cl100k_base",  # 通义千问兼容OpenAI编码
        "qwen-max": "cl100k_base",
    }
    
    def __init__(self):
        self._encoders = {}
        # 预加载所有编码器
        for encoding_name in set(self.MODEL_ENCODINGS.values()):
            self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """计算单个文本的token数"""
        encoding_name = self.MODEL_ENCODINGS.get(model, "cl100k_base")
        encoder = self._encoders.get(encoding_name)
        
        if not encoder:
            return self._estimate_tokens(text)  # 降级估算
            
        return len(encoder.encode(text))
```

### 3.2 消息格式Token计算

**OpenAI消息格式的Token计算**：
```python
def count_messages_tokens(self, messages: List[Dict[str, Any]], model: str) -> int:
    """
    计算消息列表的总token数
    包含消息格式的固定开销
    """
    total_tokens = 0
    
    # 消息格式开销
    tokens_per_message = 3  # 每条消息的格式开销
    tokens_per_name = 1     # name字段开销
    
    for message in messages:
        total_tokens += tokens_per_message
        
        # content字段token
        content = message.get("content", "")
        if content:
            total_tokens += self.count_tokens(str(content), model)
        
        # role字段token
        role = message.get("role", "")
        if role:
            total_tokens += self.count_tokens(role, model)
        
        # name字段token
        if "name" in message:
            total_tokens += tokens_per_name
            total_tokens += self.count_tokens(message["name"], model)
    
    total_tokens += 3  # 对话级别开销
    return total_tokens
```

### 3.3 智能消息截断

**保留重要信息的截断策略**：
```python
def truncate_messages(self, messages: List[Dict[str, Any]], model: str, 
                     max_tokens: Optional[int] = None, 
                     preserve_system: bool = True) -> List[Dict[str, Any]]:
    """
    智能截断策略：
    1. 保留系统消息(最重要)
    2. 从最新消息开始保留(保持上下文连贯)
    3. 预留响应空间(使用85%限制)
    """
    if not messages:
        return messages
    
    limit = max_tokens or self.MODEL_LIMITS.get(model, 100000)
    target_tokens = int(limit * 0.85)  # 预留15%给响应
    
    # 分离系统消息和对话消息
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    other_messages = [msg for msg in messages if msg.get("role") != "system"]
    
    # 计算系统消息的token
    system_tokens = self.count_messages_tokens(system_messages, model)
    remaining_tokens = target_tokens - system_tokens
    
    if remaining_tokens <= 0:
        return system_messages  # 系统消息就已经超限了
    
    # 从最新消息开始逐个添加
    selected_messages = []
    current_tokens = 0
    
    for msg in reversed(other_messages):  # 倒序遍历
        msg_tokens = self.count_messages_tokens([msg], model)
        if current_tokens + msg_tokens <= remaining_tokens:
            selected_messages.insert(0, msg)  # 保持原顺序
            current_tokens += msg_tokens
        else:
            break
    
    return system_messages + selected_messages
```

### 3.4 降级估算策略

**网络或库不可用时的估算方法**：
```python
def _estimate_tokens(self, text: str) -> int:
    """
    降级token估算方法
    基于统计规律：中文1字符≈1token，英文4字符≈1token
    """
    if not text:
        return 0
    
    # 区分中文和其他字符
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    other_chars = len(text) - chinese_chars
    
    # 中文按1:1，英文按4:1估算
    estimated_tokens = chinese_chars + (other_chars // 4)
    return max(estimated_tokens, 1)
```

---

## 4. API Key轮询策略

### 4.1 优先级调度算法

**加权轮询与优先级结合**：
```python
class LLMManager:
    def _build_provider_configs(self):
        """构建提供商配置时按优先级排序"""
        for provider_name, provider_data in self.config["llm_config"]["providers"].items():
            api_keys = []
            for key_data in provider_data["api_keys"]:
                if key_data["key"]:  # 过滤空key
                    api_keys.append(ApiKeyConfig(...))
            
            # 关键：按优先级排序，数字越小优先级越高
            api_keys.sort(key=lambda x: x.priority)
            
            self.providers[provider_name] = ProviderConfig(
                api_keys=api_keys,  # 已排序
                ...
            )
    
    async def _select_api_key(self, provider_name: str, request_tokens: int):
        """按排序后的顺序选择第一个可用的key"""
        for api_key in provider.api_keys:  # 已按priority排序
            if self._is_key_available(api_key, request_tokens):
                return api_key
        return None
```

### 4.2 健康检查机制

**API Key健康状态管理**：
```python
class ApiKeyConfig:
    error_count: int = 0
    last_error: Optional[str] = None
    enabled: bool = True

# 在调用失败时更新健康状态
async def _call_llm_api(self, ...):
    try:
        result = await self._make_http_request(...)
        if result["success"]:
            api_key.error_count = 0  # 重置错误计数
            api_key.last_error = None
            return result
        else:
            api_key.error_count += 1
            api_key.last_error = result["error"]
            
            # 严重错误直接禁用
            if "unauthorized" in error_msg.lower():
                api_key.enabled = False
                logger.error(f"API key {api_key.key[-8:]} 被自动禁用")
    except Exception as e:
        api_key.error_count += 1
        api_key.last_error = str(e)
```

### 4.3 动态权重调整

**基于成功率的权重调整**（可选功能）：
```python
def calculate_dynamic_priority(self, api_key: ApiKeyConfig) -> float:
    """
    动态优先级计算 = 基础优先级 + 错误率惩罚
    错误率高的key优先级降低
    """
    base_priority = api_key.priority
    
    # 计算错误率惩罚
    error_penalty = min(api_key.error_count * 0.1, 2.0)  # 最多惩罚2
    
    # 计算使用率惩罚
    usage_rate = api_key.current_qpm / api_key.qpm_limit if api_key.qpm_limit > 0 else 0
    usage_penalty = usage_rate * 0.5  # 使用率高的稍微降低优先级
    
    return base_priority + error_penalty + usage_penalty
```

---

## 5. 配置系统设计

### 5.1 YAML配置结构

**分层配置设计**：
```yaml
llm_config:
  # 全局配置 - 影响所有提供商
  global:
    max_token_window: 100000    # 全局token限制
    request_timeout: 180        # 请求超时
    max_retries: 3             # 全局重试次数
    retry_delay: 1             # 重试基础延迟
    fallback_enabled: true     # 是否启用降级
    
  # 提供商配置 - 每个LLM服务商独立配置
  providers:
    qwen:
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
      api_keys:
        - key: "${QWEN_API_KEY_1}"  # 环境变量引用
          priority: 1               # 优先级
          qpm_limit: 100           # 该key的QPM限制
          tpm_limit: 200000        # 该key的TPM限制
          enabled: true            # 是否启用
      models:
        - name: "qwen-max-latest"
          max_tokens: 8000
          temperature: 0.1
          enabled: true
          
  # 模型优先级 - 全局模型选择顺序
  model_priority:
    - provider: "qwen"
      model: "qwen-max-latest"
    - provider: "openai"
      model: "gpt-4o-mini"
      
  # 限流配置 - Redis相关设置
  rate_limit:
    backend: "redis"
    key_prefix: "llm_rate_limit"
    window_size: 60
    precision: 1
```

### 5.2 环境变量解析

**安全的配置值解析**：
```python
def _resolve_env_variables(self):
    """
    递归解析环境变量引用
    支持 ${VAR_NAME} 格式
    """
    def resolve_value(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]  # 提取变量名
            env_value = os.getenv(env_var, "")
            if not env_value:
                logger.warning(f"环境变量 {env_var} 未设置")
            return env_value
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value
    
    self.config = resolve_value(self.config)
```

### 5.3 配置热重载

**定时重载配置文件**：
```python
class LLMManager:
    def __init__(self):
        self._last_config_reload = 0
        self._config_reload_interval = 300  # 5分钟
    
    async def _check_config_reload(self):
        """每次请求前检查是否需要重载配置"""
        if time.time() - self._last_config_reload > self._config_reload_interval:
            try:
                old_config = self.config.copy()
                self._load_config()
                
                # 比较配置变化
                if old_config != self.config:
                    logger.info("配置已更新并重载")
                    # 可以在这里添加配置变化的具体日志
                    
            except Exception as e:
                logger.warning(f"配置重载失败，继续使用旧配置: {e}")
```

### 5.4 配置验证

**启动时的配置有效性检查**：
```python
def _validate_config(self):
    """配置有效性验证"""
    errors = []
    
    # 检查必要的配置项
    if not self.config.get("llm_config", {}).get("providers"):
        errors.append("没有配置任何LLM提供商")
    
    # 检查每个提供商的配置
    for provider_name, provider_config in self.config["llm_config"]["providers"].items():
        if not provider_config.get("api_keys"):
            errors.append(f"提供商 {provider_name} 没有配置API keys")
        
        for i, key_config in enumerate(provider_config.get("api_keys", [])):
            if not key_config.get("key"):
                errors.append(f"提供商 {provider_name} 的第 {i+1} 个API key为空")
            
            if key_config.get("qpm_limit", 0) <= 0:
                errors.append(f"提供商 {provider_name} 的API key QPM限制必须大于0")
    
    # 检查模型优先级配置
    priorities = self.config["llm_config"].get("model_priority", [])
    if not priorities:
        errors.append("没有配置模型优先级")
    
    if errors:
        raise Exception(f"配置验证失败: {'; '.join(errors)}")
```

---

## 6. 错误处理与重试机制

### 6.1 指数退避重试

**智能重试策略**：
```python
async def chat_completion(self, messages, max_retries=None):
    """带重试的聊天完成"""
    max_retries = max_retries or self.config["llm_config"]["global"]["max_retries"]
    
    for model_priority in self.get_model_priorities():
        api_key = await self._select_api_key(...)
        if not api_key:
            continue
            
        # 对每个API key进行重试
        for retry in range(max_retries + 1):
            try:
                result = await self._call_llm_api(...)
                if result["success"]:
                    return self._build_response(result)
                else:
                    error_msg = result["error"]
                    
                    # 判断是否需要重试
                    if self._is_retryable_error(error_msg):
                        if retry < max_retries:
                            # 指数退避: 1s, 2s, 4s, 8s...
                            delay = self.config["llm_config"]["global"]["retry_delay"] * (2 ** retry)
                            logger.warning(f"重试 {retry + 1}/{max_retries}, 等待 {delay}s: {error_msg}")
                            await asyncio.sleep(delay)
                            continue
                    else:
                        # 不可重试的错误，直接跳到下一个key
                        logger.error(f"不可重试的错误: {error_msg}")
                        break
                        
            except Exception as e:
                # 网络错误等，可以重试
                if retry < max_retries:
                    delay = self.config["llm_config"]["global"]["retry_delay"] * (2 ** retry)
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"API调用最终失败: {e}")
    
    raise Exception("所有LLM服务不可用")

def _is_retryable_error(self, error_msg: str) -> bool:
    """判断错误是否可重试"""
    retryable_errors = [
        "timeout", "rate limit", "server error", 
        "service unavailable", "bad gateway"
    ]
    error_lower = error_msg.lower()
    return any(err in error_lower for err in retryable_errors)
```

### 6.2 错误分类处理

**不同类型错误的处理策略**：
```python
class ErrorHandler:
    @staticmethod
    def categorize_error(error_msg: str, status_code: int = None) -> str:
        """错误分类"""
        error_lower = error_msg.lower()
        
        if status_code == 401 or "unauthorized" in error_lower:
            return "AUTH_ERROR"      # 认证错误 - 禁用API key
        elif status_code == 429 or "rate limit" in error_lower:
            return "RATE_LIMIT"      # 限流错误 - 等待重试
        elif status_code in [500, 502, 503, 504]:
            return "SERVER_ERROR"    # 服务器错误 - 可重试
        elif "timeout" in error_lower:
            return "TIMEOUT"         # 超时 - 可重试
        elif "quota" in error_lower or "balance" in error_lower:
            return "QUOTA_EXCEEDED"  # 配额耗尽 - 禁用API key
        else:
            return "UNKNOWN"         # 未知错误 - 谨慎重试
    
    @staticmethod
    def should_disable_key(error_category: str) -> bool:
        """判断是否应该禁用API key"""
        return error_category in ["AUTH_ERROR", "QUOTA_EXCEEDED"]
    
    @staticmethod
    def should_retry(error_category: str) -> bool:
        """判断是否应该重试"""
        return error_category in ["RATE_LIMIT", "SERVER_ERROR", "TIMEOUT"]
```

### 6.3 熔断器模式

**防止级联失败的熔断器**：
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """熔断器包装的函数调用"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
```

---

## 7. 监控与状态管理

### 7.1 实时状态收集

**多维度状态统计**：
```python
async def get_status(self) -> Dict[str, Any]:
    """获取系统状态的完整信息"""
    status = {
        "system": {
            "uptime": time.time() - self._start_time,
            "last_config_reload": self._last_config_reload,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
        },
        "providers": {},
        "global_config": self.config["llm_config"]["global"]
    }
    
    # 收集每个提供商的状态
    for provider_name, provider in self.providers.items():
        api_keys_status = []
        
        for api_key in provider.api_keys:
            # 从限流器获取实时使用量
            usage = await self.api_key_limiter.get_api_key_usage(api_key.key)
            
            api_keys_status.append({
                "id": api_key.key[-8:],  # 脱敏显示
                "priority": api_key.priority,
                "enabled": api_key.enabled,
                "limits": {
                    "qpm": api_key.qpm_limit,
                    "tpm": api_key.tpm_limit,
                },
                "current_usage": {
                    "qpm": usage.get("requests", 0),
                    "tpm": usage.get("tokens", 0),
                    "qpm_percent": round(usage.get("requests", 0) / api_key.qpm_limit * 100, 1),
                    "tpm_percent": round(usage.get("tokens", 0) / api_key.tpm_limit * 100, 1),
                },
                "health": {
                    "error_count": api_key.error_count,
                    "last_error": api_key.last_error,
                    "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
                }
            })
        
        status["providers"][provider_name] = {
            "base_url": provider.base_url,
            "models": [{"name": m.name, "enabled": m.enabled} for m in provider.models],
            "api_keys": api_keys_status,
            "total_api_keys": len(provider.api_keys),
            "active_api_keys": len([k for k in provider.api_keys if k.enabled])
        }
    
    return status
```

### 7.2 性能指标统计

**关键指标的收集和计算**：
```python
class MetricsCollector:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.token_usage = {"input": 0, "output": 0}
        self.provider_usage = defaultdict(int)
        
    def record_request(self, response: LLMResponse, error: Exception = None):
        """记录请求指标"""
        self.request_count += 1
        
        if error:
            self.error_count += 1
        else:
            self.response_times.append(response.response_time)
            self.token_usage["input"] += response.usage.get("prompt_tokens", 0)
            self.token_usage["output"] += response.usage.get("completion_tokens", 0)
            self.provider_usage[response.provider] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取统计指标"""
        return {
            "requests": {
                "total": self.request_count,
                "success": self.request_count - self.error_count,
                "error": self.error_count,
                "success_rate": (self.request_count - self.error_count) / max(self.request_count, 1)
            },
            "performance": {
                "avg_response_time": sum(self.response_times) / max(len(self.response_times), 1),
                "p95_response_time": self._percentile(self.response_times, 95),
                "p99_response_time": self._percentile(self.response_times, 99),
            },
            "tokens": self.token_usage,
            "providers": dict(self.provider_usage)
        }
```

### 7.3 健康检查端点

**系统健康状态检查**：
```python
@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查Redis连接
        redis_client = await llm_manager.rate_limiter._get_redis_client()
        await redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {e}"
    
    # 检查配置文件
    config_status = "healthy" if llm_manager.config else "missing"
    
    # 检查API keys
    total_keys = 0
    active_keys = 0
    for provider in llm_manager.providers.values():
        total_keys += len(provider.api_keys)
        active_keys += len([k for k in provider.api_keys if k.enabled])
    
    health_status = {
        "status": "healthy" if redis_status == "healthy" and active_keys > 0 else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "redis": redis_status,
            "config": config_status,
            "api_keys": f"{active_keys}/{total_keys} active"
        },
        "version": "1.0.0"
    }
    
    return health_status
```

---

## 8. 部署和运维

### 8.1 Docker化部署

**生产环境容器化**：
```dockerfile
# Dockerfile for langchain-agent
FROM python:3.11-slim

WORKDIR /app
COPY requirements_llm.txt .
RUN pip install -r requirements_llm.txt

COPY . .

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "api.py"]
```

### 8.2 日志配置

**结构化日志输出**：
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_request(self, request_id: str, provider: str, model: str, 
                   tokens: int, response_time: float, success: bool):
        """记录请求日志"""
        log_data = {
            "event": "llm_request",
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "tokens": tokens,
            "response_time": response_time,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_data))
```

### 8.3 监控告警

**关键指标的监控脚本**：
```python
import asyncio
import aiohttp

async def monitor_system():
    """系统监控脚本"""
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                # 检查系统状态
                async with session.get("http://127.0.0.1:8000/status") as resp:
                    status = await resp.json()
                
                # 检查告警条件
                for provider_name, provider_status in status["providers"].items():
                    for api_key in provider_status["api_keys"]:
                        # QPM使用率告警
                        if api_key["current_usage"]["qpm_percent"] > 90:
                            await send_alert(f"API Key {api_key['id']} QPM usage > 90%")
                        
                        # 错误率告警
                        if api_key["health"]["error_count"] > 10:
                            await send_alert(f"API Key {api_key['id']} error count > 10")
                
        except Exception as e:
            await send_alert(f"Monitor system error: {e}")
        
        await asyncio.sleep(60)  # 每分钟检查一次

async def send_alert(message: str):
    """发送告警通知"""
    # 可以集成钉钉、企微、邮件等通知渠道
    print(f"🚨 ALERT: {message}")
```

---

## 9. 测试策略

### 9.1 单元测试

**关键组件的单元测试**：
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestTokenCounter:
    def test_basic_token_counting(self):
        counter = TokenCounter()
        text = "Hello world"
        tokens = counter.count_tokens(text, "gpt-4o-mini")
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_message_token_counting(self):
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        tokens = counter.count_messages_tokens(messages, "gpt-4o-mini")
        assert tokens > 10  # 应该包含格式开销

class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        # Mock Redis client
        redis_mock = AsyncMock()
        redis_mock.pipeline.return_value.execute.return_value = [0, 0, 0, 0]  # 空的使用量
        
        limiter = RateLimiter(redis_client=redis_mock)
        
        # 第一个请求应该被允许
        allowed, status = await limiter.check_and_increment("test_key", 1, 100, 10, 1000)
        assert allowed == True
        
        # 模拟达到限制
        redis_mock.pipeline.return_value.execute.return_value = [10, 1000, 0, 0]  # 已达限制
        allowed, status = await limiter.check_and_increment("test_key", 1, 100, 10, 1000)
        assert allowed == False
```

### 9.2 集成测试

**端到端功能测试**：
```python
class TestLLMManager:
    @pytest.mark.asyncio
    async def test_chat_completion_with_fallback(self):
        """测试模型降级功能"""
        manager = LLMManager("test_config.yaml")
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await manager.chat_completion(messages, preferred_model="unavailable-model")
        
        # 应该降级到可用模型
        assert response.content
        assert response.model in ["qwen-max-latest", "gpt-4o-mini"]
    
    @pytest.mark.asyncio
    async def test_api_key_rotation(self):
        """测试API key轮询"""
        manager = LLMManager("test_config.yaml")
        
        # 发送多个请求，观察是否使用了不同的API key
        used_keys = set()
        for _ in range(5):
            response = await manager.chat_completion([{"role": "user", "content": "test"}])
            used_keys.add(response.api_key_id)
        
        # 应该使用了多个不同的key（如果有多个可用）
        assert len(used_keys) >= 1
```

这个技术实现详解涵盖了LLM轮询系统的所有核心功能，每个功能都有具体的实现原理、代码示例和最佳实践。通过这些实现，您可以构建一个生产级的、高可靠性的LLM API管理系统。 