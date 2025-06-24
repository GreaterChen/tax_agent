"""税务问答Agent实现 - 使用LangChain原生方式"""
import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnablePassthrough

from src.tools.latex_calc.latex_calc import latex_calc_tool
from src.tools.news_query import news_query_tool
from src.tools.web_search.web_search import advanced_web_search_tool
from src.tools.vector_search.vector_search import vector_search_tool
from src.utils.custom_chat_model import create_managed_chat_model

load_dotenv()

# 配置日志
logger = logging.getLogger(__name__)

class TaxAgent:
    def __init__(self):
        # 使用自定义的管理ChatModel - 内部整合了轮询和限流功能
        self.llm = create_managed_chat_model(
            provider="auto",  # 自动选择最佳提供商和API key
            model="qwen-max-latest",  # 默认模型
            temperature=0.1
        )
        
        # 基础工具（始终可用）
        self.base_tools = [
            latex_calc_tool,
            vector_search_tool
        ]
        
        # 可选工具
        self.web_search_tool = advanced_web_search_tool
        self.news_query_tool = news_query_tool

        # 系统prompt
        self.system_prompt = """你是一个专业的税务顾问助手。你可以:
1. 回答税务相关问题
2. 使用计算器进行税务计算
3. 使用advanced_web_search工具进行高级的互联网搜索最新的税务新闻和政策
4. 使用vector_search工具在本地知识库中搜索相关信息
5. 使用session_vector_search工具搜索用户上传的文档内容

语言要求：
- 保持回答语言与提问语言一致

工具使用优先级规则：
**重要：如果可用工具中包含session_vector_search，说明用户上传了相关文档，你必须首先使用session_vector_search工具从用户上传的文档中检索相关信息，因为这些文档与用户问题强相关。**

工具使用说明：
- 需要计算时，使用latex_calc工具,接收的参数是标准的latex表达式和参数取值
- **如果有session_vector_search工具可用，必须优先使用它搜索用户上传的文档**
- 需要搜索本地知识库中的信息时，使用vector_search工具进行向量搜索
- 需要搜索互联网上的税务信息或最新政策时，使用advanced_web_search工具进行高级搜索, 只可以调用一次！
- 在向advanced_web_search工具提问时，请保证不要私自更改问题的范围、限定，最好直接原封不动使用用户对话中的问题

回答策略：
1. 如果有session_vector_search工具，优先基于用户上传的文档内容回答问题
2. 如果用户文档中的信息不足，再结合其他工具补充信息
3. 明确标注信息来源（上传文档 vs 知识库 vs 网络搜索）

回答格式要求：
1. 保持专业和友好的语气
2. 问题要叙述清晰
3. 每个关键信息点后都应该添加对应的来源引用
   - 上传文档来源格式：[来源: 上传文档 - 文件名]
   - 网络来源格式：[来源: URL]
   - 知识库来源格式：[来源: 知识库]
4. 确保引用的信息来源可靠且最新
5. 如果实在检索不到相关的信息，也可以通过你自己已有的知识回答，但是要明确说明没有检索到相关信息"""

    def _select_tools(self, web_search: bool = True, enable_rag: bool = True, 
                     session_files: List[str] = None) -> List[StructuredTool]:
        """根据参数动态选择工具"""
        tools = []
        
        # 添加基础工具
        tools.extend(self.base_tools)
        
        # 添加会话级向量搜索工具（如果有上传文件）
        if session_files and len(session_files) > 0 and enable_rag:
            session_vector_tool = self._create_session_vector_tool(session_files)
            if session_vector_tool:
                tools.insert(0, session_vector_tool)  # 放在第一位，确保优先级
        
        # 添加网络搜索工具
        if web_search:
            tools.append(self.web_search_tool)
            
        # 添加新闻查询工具
        tools.append(self.news_query_tool)
        
        return tools

    def _create_agent_executor(self, tools: List[StructuredTool]) -> AgentExecutor:
        """创建Agent执行器"""
        # 创建prompt模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        # 创建agent执行器
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )   # TODO: 参数什么意思？
        
        return agent_executor

    def query(self, question: str, thread_id: Optional[str] = None, 
              web_search: bool = True, session_files: Optional[List[str]] = None, 
              enable_rag: bool = True) -> List[str]:
        """
        执行查询 - 使用LangChain原生Agent方式
        
        Args:
            question: 用户问题
            thread_id: 线程ID
            web_search: 是否启用网络搜索
            session_files: 会话级文档文件列表
            enable_rag: 是否启用RAG功能
            
        Returns:
            AI回答列表
        """
        try:
            # 根据参数选择工具
            tools = self._select_tools(web_search, enable_rag, session_files)
            
            # 增强问题（如果有上传文件）
            enhanced_question = self._enhance_question_with_files(question, session_files, enable_rag)
            
            # 创建Agent执行器
            agent_executor = self._create_agent_executor(tools)
            
            # 执行查询
            logger.info(f"执行查询: {enhanced_question[:100]}...")
            logger.info(f"可用工具: {[tool.name for tool in tools]}")
            
            result = agent_executor.invoke({
                "input": enhanced_question
            })
            
            return [result["output"]]
            
        except Exception as e:
            logger.error(f"查询失败: {e}", exc_info=True)
            return [f"抱歉，处理您的请求时发生错误: {str(e)}"]

    def _enhance_question_with_files(self, question: str, session_files: Optional[List[str]], 
                                   enable_rag: bool) -> str:
        """根据上传文件增强问题"""
        if not session_files or len(session_files) == 0:
            return question
            
        if enable_rag:
            # RAG模式：提示Agent使用session_vector_search工具
            file_names = [os.path.basename(f) for f in session_files]
            enhanced_question = f"""用户已上传相关文档：{', '.join(file_names)}
这些文档与问题强相关，请务必先使用session_vector_search工具从上传的文档中检索相关信息再回答问题。

用户问题：{question}"""
        else:
            # 非RAG模式：直接读取文件内容
            try:
                from src.utils.file_utils import read_session_files_content
                file_contents = read_session_files_content(session_files)
                if file_contents:
                    enhanced_question = f"""以下是用户上传的相关文档内容，请基于这些内容回答用户问题：

用户问题：{question}

文档内容：{file_contents}"""
                else:
                    enhanced_question = question
            except Exception as e:
                logger.warning(f"读取文件内容失败: {e}")
                enhanced_question = question
        
        return enhanced_question

    def _create_session_vector_tool(self, session_files: List[str]) -> Optional[StructuredTool]:
        """创建会话级向量搜索工具"""
        try:
            from src.tools.session_vector_search.session_vector_search import create_session_vector_tool
            # 使用文件路径作为thread_id的一部分，确保唯一性
            thread_id = f"session_{hash(tuple(session_files))}"
            return create_session_vector_tool(session_files, thread_id)
        except Exception as e:
            logger.error(f"创建会话向量搜索工具失败: {e}")
            return None

# 创建全局实例
tax_agent = TaxAgent()
