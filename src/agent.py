"""税务问答Agent实现 (修复无限自问自答bug + __end__ KeyError处理 + 修复Postman无返回问题)"""
import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI, ChatTongyi
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool

from src.tools.calculator import calculate
from src.tools.news_query import news_query_tool
from src.tools.advanced_web_search import advanced_web_search_tool

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
        self.llm = ChatTongyi(
            model="qwen-long",
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        self.tools = [
            calculate,
            # news_query_tool,
            advanced_web_search_tool
        ]

        system_prompt = """你是一个专业的税务顾问助手。你可以:
1. 回答税务相关问题
2. 使用计算器进行税务计算
3. 查询最新的税务新闻和政策
4. 使用advanced_web_search工具进行高级的互联网搜索

语言要求：
- 如果用户使用中文提问，请使用中文回答
- 如果用户使用英文提问，请使用英文回答
- 保持回答语言与提问语言一致

工具使用说明：
- 需要计算时，使用calculator工具
- 需要搜索互联网上的税务信息或最新政策时，优先使用advanced_web_search工具进行高级搜索, 只可以调用一次！一次可以获取5个搜索结果，如果一次搜索不到就不要尝试再搜索了

回答格式要求：
1. 保持专业和友好的语气
2. 问题要叙述清晰
3. 每个关键信息点后都应该添加对应的URL引用
   - 中文回答格式：[来源: URL]
   - 英文回答格式：[Source: URL]
4. 确保引用的信息来源可靠且最新
5. 如果实在检索不到相关的信息，也可以通过你自己已有的知识回答，但是要明确说明没有检索到相关信息"""

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