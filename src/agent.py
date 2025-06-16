"""税务问答Agent实现"""
import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging
from langchain_community.chat_models import ChatZhipuAI, ChatTongyi
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool

from src.tools.latex_calc.latex_calc import latex_calc_tool
from src.tools.news_query import news_query_tool
from src.tools.web_search.web_search import advanced_web_search_tool
from src.tools.vector_search.vector_search import vector_search_tool


load_dotenv()

# 配置日志
logger = logging.getLogger(__name__)

class TaxAgent:
    def __init__(self):
        # self.llm = ChatTongyi(
        #     model="qwen-long",
        #     api_key=os.getenv("DASHSCOPE_API_KEY")
        # )

        # self.llm = ChatZhipuAI(
        #     model="glm-4-flash",
        #     temperature=0.1,
        #     zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY")
        # )

        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,  
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )

        # 基础工具（始终可用）
        self.base_tools = [
            latex_calc_tool,
            vector_search_tool
        ]
        
        # 可选工具
        self.web_search_tool = advanced_web_search_tool

        system_prompt = """你是一个专业的税务顾问助手。你可以:
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
- 需要搜索互联网上的税务信息或最新政策时，使用advanced_web_search工具进行高级搜索, 只可以调用一次！，如果一次搜索不到就不要尝试再搜索了
- 在向advanced_web_search工具提问时，请保证不要私自更改问题的范围、限定，比如添加年份，添加new zealand这些根本在问题没有提到的问题，最好直接原封不动使用用户对话中的问题，

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

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.memory = MemorySaver()
        
        # 初始化时不绑定工具，在查询时动态绑定
        self.system_prompt = system_prompt

    def _create_graph(self, tools: List) -> StateGraph:
        """根据工具列表创建图"""
        # 动态绑定工具
        llm_with_tools = self.llm.bind_tools(tools)
        tool_node = ToolNode(tools)
        
        # 创建代理
        agent = create_react_agent(
            model=llm_with_tools,
            tools=tools,
            prompt=self.system_prompt
        )
        
        builder = StateGraph(MessagesState)
        builder.add_node("agent", agent)
        builder.add_node("tools", tool_node)

        builder.set_entry_point("agent")
        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                "__end__": END,
            }
        )
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=self.memory)

    def query(self, question: str, thread_id: Optional[str] = None, 
              web_search: bool = True, session_files: Optional[List[str]] = None, 
              enable_rag: bool = True) -> List[str]:
        """执行查询
        
        Args:
            question: 用户问题
            thread_id: 线程ID
            web_search: 是否启用网络搜索
            session_files: 会话级文档文件列表
            enable_rag: 是否启用RAG功能
        """
        # 根据参数构建工具列表
        tools = []
        
        # 处理会话文档
        enhanced_question = question
        if session_files and len(session_files) > 0:
            if enable_rag:
                # RAG模式：创建会话级向量搜索工具
                session_vector_tool = self._create_session_vector_tool(session_files, thread_id)
                if session_vector_tool:
                    tools.append(session_vector_tool)  # 放在第一位，确保优先级
                
                # 在问题前添加提示
                file_names = [os.path.basename(f) for f in session_files]
                enhanced_question = f"""用户已上传相关文档：{', '.join(file_names)}
这些文档与问题强相关，请务必先从上传的文档中检索相关信息再回答问题。

用户问题：{question}"""
            else:
                # 非RAG模式：直接读取文件内容并加入prompt
                from src.utils.file_utils import read_session_files_content
                file_contents = read_session_files_content(session_files)
                if file_contents:
                    enhanced_question = f"""以下是用户上传的相关文档内容，请基于这些内容回答用户问题：
                    用户问题：{question}
                    文档内容：{file_contents}

"""
        
        # 添加基础工具
        tools.extend(self.base_tools)
        
        # 添加网络搜索工具（如果启用）
        if web_search:
            tools.append(self.web_search_tool)
        
        # 创建工作流
        workflow = self._create_graph(tools)
        
        messages = [HumanMessage(content=enhanced_question)]
        config = {
            "configurable": {
                "thread_id": thread_id or "default"
            },
            "recursion_limit": 10
        }
        result = []
        for step in workflow.stream({"messages": messages}, config=config, stream_mode="updates"):
            if "messages" in step['agent']:
                last_msg = step['agent']['messages'][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    result.append(last_msg.content)
        return result
    
    def _create_session_vector_tool(self, session_files: List[str], thread_id: str):
        """创建会话级向量搜索工具"""
        try:
            from src.tools.session_vector_search.session_vector_search import create_session_vector_tool
            return create_session_vector_tool(session_files, thread_id)
        except Exception as e:
            logger.error(f"创建会话级向量搜索工具失败: {str(e)}")
            return None
