"""税务问答Agent实现 (修复无限自问自答bug + __end__ KeyError处理 + 修复Postman无返回问题)"""
import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool

from src.tools.calculator import calculate
from src.tools.news_query import news_query_tool

def check_environment():
    required_vars = {
        "ZHIPUAI_API_KEY": "智谱AI API密钥",
        "DATABASE_URL": "数据库连接URL",
        "LANGSMITH_API_KEY": "LangSmith API密钥",
        "LANGSMITH_PROJECT": "LangSmith项目名称"
    }

    missing_vars = []
    for var, desc in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({desc})")

    if missing_vars:
        print("错误: 以下必要的环境变量未设置:")
        for var in missing_vars:
            print(f"- {var}")
        sys.exit(1)

    os.environ["LANGSMITH_TRACING"] = "true"
    if not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

load_dotenv()
check_environment()

class TaxAgent:
    def __init__(self):
        self.llm = ChatZhipuAI(
            model="glm-4-flash",
            temperature=0.5,
            zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY")
        )

        self.tools = [
            calculate,
            news_query_tool
        ]

        system_prompt = """你是一个专业的税务顾问助手。你可以:
1. 回答税务相关问题
2. 使用计算器进行税务计算
3. 查询最新的税务新闻和政策

请使用中文回答,保持专业和友好的语气。如果需要计算,使用calculator工具。
如果需要查询新闻,使用news_query工具。

你还具有记忆功能,可以记住与用户的对话历史,这样可以提供更连贯的对话体验。
请善用这个能力,在回答时考虑之前的对话内容。

确保回答准确、清晰,并在必要时引用相关法规或新闻来源。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)

        self.agent = create_react_agent(
            model=self.llm_with_tools,
            tools=self.tools,
            prompt=system_prompt
        )

        self.memory = MemorySaver()
        self.workflow = self._create_graph()

    def _create_graph(self) -> StateGraph:
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