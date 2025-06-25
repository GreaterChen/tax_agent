"""
分布式限流器
支持QPM(每分钟请求数)和TPM(每分钟Token数)限制
使用Redis滑动窗口算法实现精确限流
"""
import time
import redis
import json
import asyncio
import os
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """分布式限流器"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 key_prefix: str = "rate_limit",
                 window_size: int = 60,
                 precision: int = 1):
        """
        初始化限流器
        
        Args:
            redis_client: Redis客户端
            key_prefix: Redis key前缀
            window_size: 窗口大小(秒)
            precision: 统计精度(秒)
        """
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.window_size = window_size
        self.precision = precision
        self._local_cache = {}  # 本地缓存，减少Redis访问
        self._cache_ttl = {}    # 缓存TTL
    
    async def _get_redis_client(self) -> redis.Redis:
        """获取Redis客户端"""
        if not self.redis_client:
            try:
                # 使用环境变量配置Redis
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/2")
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.error(f"获取Redis客户端失败: {e}")
                raise Exception("限流器初始化失败：无法连接Redis")
        return self.redis_client
    
    def _get_window_key(self, identifier: str, limit_type: str, window_start: int) -> str:
        """生成窗口key"""
        return f"{self.key_prefix}:{identifier}:{limit_type}:{window_start}"
    
    def _get_current_windows(self) -> list:
        """获取当前需要检查的时间窗口"""
        now = int(time.time())
        windows = []
        
        # 生成滑动窗口
        for i in range(0, self.window_size, self.precision):
            window_start = now - i
            # 对齐到precision边界
            window_start = (window_start // self.precision) * self.precision
            windows.append(window_start)
        
        return sorted(set(windows))  # 去重并排序
    
    async def check_and_increment(self, identifier: str, request_count: int = 1, 
                                token_count: int = 0, qpm_limit: int = 0, 
                                tpm_limit: int = 0) -> Tuple[bool, Dict[str, Any]]:
        """
        检查并增加计数
        
        Args:
            identifier: 标识符(如API key)
            request_count: 请求数增量
            token_count: token数增量  
            qpm_limit: QPM限制
            tpm_limit: TPM限制
            
        Returns:
            (是否允许, 状态信息)
        """
        try:
            redis_client = await self._get_redis_client()
            
            # 先检查本地缓存(用于快速拒绝)
            cache_key = f"{identifier}:check"
            if self._is_cached_blocked(cache_key):
                return False, {
                    "allowed": False,
                    "reason": "cached_blocked",
                    "retry_after": self._get_cache_retry_after(cache_key)
                }
            
            current_time = int(time.time())
            windows = self._get_current_windows()
            
            # 使用管道批量操作Redis
            pipe = redis_client.pipeline()
            
            # 清理过期的窗口
            expire_time = current_time - self.window_size
            
            # 收集当前计数
            request_keys = []
            token_keys = []
            
            for window_start in windows:
                if window_start >= expire_time:
                    req_key = self._get_window_key(identifier, "requests", window_start)
                    token_key = self._get_window_key(identifier, "tokens", window_start)
                    request_keys.append(req_key)
                    token_keys.append(token_key)
                    pipe.get(req_key)
                    pipe.get(token_key)
            
            # 执行批量查询
            results = await asyncio.get_event_loop().run_in_executor(None, pipe.execute)
            
            # 计算当前总计数
            current_requests = 0
            current_tokens = 0
            
            for i in range(0, len(results), 2):
                req_count = int(results[i] or 0)
                token_count_val = int(results[i + 1] or 0)
                current_requests += req_count
                current_tokens += token_count_val
            
            # 检查限制
            will_exceed_qpm = qpm_limit > 0 and (current_requests + request_count) > qpm_limit
            will_exceed_tpm = tpm_limit > 0 and (current_tokens + token_count) > tpm_limit
            
            if will_exceed_qpm or will_exceed_tpm:
                # 缓存拒绝状态
                self._cache_block(cache_key, self.precision)
                
                reason = []
                if will_exceed_qpm:
                    reason.append(f"QPM limit exceeded: {current_requests + request_count} > {qpm_limit}")
                if will_exceed_tpm:
                    reason.append(f"TPM limit exceeded: {current_tokens + token_count} > {tpm_limit}")
                
                return False, {
                    "allowed": False,
                    "reason": "; ".join(reason),
                    "current_qpm": current_requests,
                    "current_tpm": current_tokens,
                    "retry_after": self.precision
                }
            
            # 增加计数 - 使用当前时间窗口
            current_window = (current_time // self.precision) * self.precision
            req_key = self._get_window_key(identifier, "requests", current_window)
            token_key = self._get_window_key(identifier, "tokens", current_window)
            
            # 原子性增加计数并设置过期时间
            pipe = redis_client.pipeline()
            pipe.incrby(req_key, request_count)
            pipe.expire(req_key, self.window_size + self.precision)  # 多一点时间确保不提前过期
            if token_count > 0:
                pipe.incrby(token_key, token_count)
                pipe.expire(token_key, self.window_size + self.precision)
            
            await asyncio.get_event_loop().run_in_executor(None, pipe.execute)
            
            return True, {
                "allowed": True,
                "current_qpm": current_requests + request_count,
                "current_tpm": current_tokens + token_count,
                "qpm_limit": qpm_limit,
                "tpm_limit": tpm_limit
            }
            
        except Exception as e:
            logger.error(f"限流检查失败: {e}")
            # 限流器故障时允许请求通过（可配置策略）
            return True, {
                "allowed": True,
                "reason": f"rate_limiter_error: {str(e)}",
                "fallback": True
            }
    
    async def get_current_usage(self, identifier: str) -> Dict[str, int]:
        """
        获取当前使用量
        
        Args:
            identifier: 标识符
            
        Returns:
            当前使用量统计
        """
        try:
            redis_client = await self._get_redis_client()
            
            current_time = int(time.time())
            windows = self._get_current_windows()
            expire_time = current_time - self.window_size
            
            pipe = redis_client.pipeline()
            
            for window_start in windows:
                if window_start >= expire_time:
                    req_key = self._get_window_key(identifier, "requests", window_start)
                    token_key = self._get_window_key(identifier, "tokens", window_start)
                    pipe.get(req_key)
                    pipe.get(token_key)
            
            results = await asyncio.get_event_loop().run_in_executor(None, pipe.execute)
            
            current_requests = 0
            current_tokens = 0
            
            for i in range(0, len(results), 2):
                current_requests += int(results[i] or 0)
                current_tokens += int(results[i + 1] or 0)
            
            return {
                "requests": current_requests,
                "tokens": current_tokens,
                "window_size": self.window_size
            }
            
        except Exception as e:
            logger.error(f"获取使用量失败: {e}")
            return {"requests": 0, "tokens": 0, "window_size": self.window_size}
    
    def _is_cached_blocked(self, cache_key: str) -> bool:
        """检查本地缓存是否被阻塞"""
        if cache_key in self._local_cache:
            if time.time() < self._cache_ttl[cache_key]:
                return True
            else:
                # 清理过期缓存
                del self._local_cache[cache_key]
                del self._cache_ttl[cache_key]
        return False
    
    def _cache_block(self, cache_key: str, duration: int):
        """缓存阻塞状态"""
        self._local_cache[cache_key] = True
        self._cache_ttl[cache_key] = time.time() + duration
    
    def _get_cache_retry_after(self, cache_key: str) -> int:
        """获取缓存的重试时间"""
        if cache_key in self._cache_ttl:
            return max(0, int(self._cache_ttl[cache_key] - time.time()))
        return 0
    
    async def reset_limits(self, identifier: str):
        """重置指定标识符的限制"""
        try:
            redis_client = await self._get_redis_client()
            
            # 清理所有相关的key
            pattern = f"{self.key_prefix}:{identifier}:*"
            keys = await asyncio.get_event_loop().run_in_executor(
                None, redis_client.keys, pattern
            )
            
            if keys:
                await asyncio.get_event_loop().run_in_executor(
                    None, redis_client.delete, *keys
                )
            
            # 清理本地缓存
            cache_key = f"{identifier}:check"
            if cache_key in self._local_cache:
                del self._local_cache[cache_key]
                del self._cache_ttl[cache_key]
                
            logger.info(f"重置限制成功: {identifier}")
            
        except Exception as e:
            logger.error(f"重置限制失败: {e}")

    async def reserve_tokens(self, identifier: str, estimated_request_tokens: int, 
                           tpm_limit: int, response_multiplier: float = 3.0) -> Tuple[bool, Dict[str, Any]]:
        """
        预留Token额度（双阶段TPM控制的第一阶段）
        
        Args:
            identifier: 标识符
            estimated_request_tokens: 预估的请求token数
            tpm_limit: TPM限制
            response_multiplier: 回复token预估倍数（通常回复比请求长2-4倍）
            
        Returns:
            (是否成功预留, 预留信息)
        """
        try:
            # 预估总token = 请求token + 预估回复token
            estimated_response_tokens = int(estimated_request_tokens * response_multiplier)
            estimated_total_tokens = estimated_request_tokens + estimated_response_tokens
            
            # 使用预估总token进行预检查和预留
            allowed, status = await self.check_and_increment(
                identifier,
                request_count=0,  # 不增加请求计数，只预留token
                token_count=estimated_total_tokens,
                qpm_limit=0,  # 此阶段不检查QPM
                tpm_limit=tpm_limit
            )
            
            if allowed:
                # 记录预留信息用于后续调整
                reservation_key = f"{identifier}:reservation:{int(time.time())}"
                redis_client = await self._get_redis_client()
                reservation_data = {
                    "estimated_total": estimated_total_tokens,
                    "request_tokens": estimated_request_tokens,
                    "estimated_response": estimated_response_tokens,
                    "timestamp": int(time.time())
                }
                
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    redis_client.setex, 
                    reservation_key, 
                    300,  # 5分钟过期
                    json.dumps(reservation_data)
                )
                
                status.update({
                    "reservation_key": reservation_key,
                    "estimated_total_tokens": estimated_total_tokens,
                    "estimated_response_tokens": estimated_response_tokens
                })
            
            return allowed, status
            
        except Exception as e:
            logger.error(f"Token预留失败: {e}")
            return False, {"error": str(e)}

    async def finalize_token_usage(self, identifier: str, reservation_key: str, 
                                 actual_request_tokens: int, actual_response_tokens: int) -> Dict[str, Any]:
        """
        最终确定实际Token使用量（双阶段TPM控制的第二阶段）
        
        Args:
            identifier: 标识符
            reservation_key: 预留key
            actual_request_tokens: 实际请求token数
            actual_response_tokens: 实际回复token数
            
        Returns:
            调整结果信息
        """
        try:
            redis_client = await self._get_redis_client()
            
            # 获取预留信息
            reservation_data_str = await asyncio.get_event_loop().run_in_executor(
                None, redis_client.get, reservation_key
            )
            
            if not reservation_data_str:
                logger.warning(f"未找到预留信息: {reservation_key}")
                # 如果没有预留信息，直接记录实际使用量
                actual_total = actual_request_tokens + actual_response_tokens
                await self._adjust_token_count(identifier, actual_total)
                return {"status": "fallback_recorded", "actual_tokens": actual_total}
            
            reservation_data = json.loads(reservation_data_str)
            estimated_total = reservation_data["estimated_total"]
            actual_total = actual_request_tokens + actual_response_tokens
            
            # 计算调整量
            adjustment = actual_total - estimated_total
            
            if adjustment != 0:
                await self._adjust_token_count(identifier, adjustment)
            
            # 删除预留记录
            await asyncio.get_event_loop().run_in_executor(
                None, redis_client.delete, reservation_key
            )
            
            return {
                "status": "finalized",
                "estimated_tokens": estimated_total,
                "actual_tokens": actual_total,
                "adjustment": adjustment,
                "efficiency": round(actual_total / estimated_total, 2) if estimated_total > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Token使用量最终确定失败: {e}")
            return {"status": "error", "error": str(e)}

    async def _adjust_token_count(self, identifier: str, adjustment: int):
        """调整token计数"""
        if adjustment == 0:
            return
            
        try:
            redis_client = await self._get_redis_client()
            current_time = int(time.time())
            current_window = (current_time // self.precision) * self.precision
            token_key = self._get_window_key(identifier, "tokens", current_window)
            
            # 调整当前窗口的token计数
            pipe = redis_client.pipeline()
            pipe.incrby(token_key, adjustment)
            pipe.expire(token_key, self.window_size + self.precision)
            await asyncio.get_event_loop().run_in_executor(None, pipe.execute)
            
            logger.info(f"调整{identifier}的token计数: {adjustment}")
            
        except Exception as e:
            logger.error(f"调整token计数失败: {e}")


class ApiKeyRateLimiter:
    """API Key专用限流器"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def check_api_key_limits(self, api_key: str, request_tokens: int,
                                 qpm_limit: int, tpm_limit: int) -> Tuple[bool, Dict[str, Any]]:
        """
        检查API Key限制
        
        Args:
            api_key: API密钥
            request_tokens: 本次请求的token数
            qpm_limit: QPM限制
            tpm_limit: TPM限制
            
        Returns:
            (是否允许, 状态信息)
        """
        # 使用API key作为标识符
        identifier = f"api_key:{api_key[-8:]}"  # 只使用后8位避免泄露
        
        return await self.rate_limiter.check_and_increment(
            identifier=identifier,
            request_count=1,
            token_count=request_tokens,
            qpm_limit=qpm_limit,
            tpm_limit=tpm_limit
        )
    
    async def get_api_key_usage(self, api_key: str) -> Dict[str, int]:
        """获取API Key使用量"""
        identifier = f"api_key:{api_key[-8:]}"
        return await self.rate_limiter.get_current_usage(identifier) 