"""税务问答Agent实现 (精简版本，适合2核2G小型服务器)"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition

from src.tools.web_search import web_search_tool
from src.tools.calculator import calculate

load_dotenv()

class TaxAgent:
    def __init__(self):
        # 初始化LLM (智谱AI)
        self.llm = ChatZhipuAI(
            model="glm-4-flash",
            temperature=0.5,
            zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY")
        )

        # 精简后只保留计算器和Web搜索工具
        self.tools = [
            calculate,
            web_search_tool
        ]

        # 系统提示词
        system_prompt = """你是一个专业的税务顾问助手。你可以:
1. 回答税务相关问题
2. 使用计算器进行税务计算
3. 使用web_search工具进行互联网搜索获取最新税务信息

请使用中文回答,保持专业和友好的语气。如果需要计算,使用calculator工具。
如果需要搜索互联网上的税务信息或最新政策,使用web_search工具进行搜索，回答时需要携带信息来源。

你还具有记忆功能,可以记住与用户的对话历史,这样可以提供更连贯的对话体验。
请善用这个能力,在回答时考虑之前的对话内容。

确保回答准确、清晰,并在必要时引用相关法规或信息来源。"""

        # 绑定工具到LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)

        # 创建Agent
        self.agent = create_react_agent(
            model=self.llm_with_tools,
            tools=self.tools,
            prompt=system_prompt
        )

        # 初始化内存和工作流
        self.memory = MemorySaver()
        self.workflow = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """创建Agent工作流图"""
        builder = StateGraph(MessagesState)
        builder.add_node("agent", self.agent)
        builder.add_node("tools", self.tool_node)

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

    def query(self, question: str, thread_id: Optional[str] = None) -> List[str]:
        """处理用户查询
        
        Args:
            question: 用户问题
            thread_id: 会话ID，用于记忆功能
            
        Returns:
            回答列表
        """
        messages = [HumanMessage(content=question)]
        config = {
            "configurable": {
                "thread_id": thread_id or "default"
            },
            "recursion_limit": 10
        }
        result = []
        for step in self.workflow.stream({"messages": messages}, config=config, stream_mode="updates"):
            if "messages" in step['agent']:
                last_msg = step['agent']['messages'][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    result.append(last_msg.content)
        return result