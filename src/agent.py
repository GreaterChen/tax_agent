"""税务问答Agent实现"""
import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI, ChatTongyi
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool

from src.tools.latex_calc.calculator import calculate
from src.tools.news_query import news_query_tool
from src.tools.web_search import advanced_web_search_tool

load_dotenv()

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

        self.tools = [
            calculate,
            advanced_web_search_tool
        ]

        system_prompt = """你是一个专业的税务顾问助手。你可以:
1. 回答税务相关问题
2. 使用计算器进行税务计算
3. 使用advanced_web_search工具进行高级的互联网搜索最新的税务新闻和政策

语言要求：
- 保持回答语言与提问语言一致

工具使用说明：
- 需要计算时，使用calculator工具
- 需要搜索互联网上的税务信息或最新政策时，使用advanced_web_search工具进行高级搜索, 只可以调用一次！，如果一次搜索不到就不要尝试再搜索了
- 在向advanced_web_search工具提问时，请保证不要私自更改问题的范围、限定，比如添加年份，添加new zealand这些根本在问题没有提到的问题，最好直接原封不动使用用户对话中的问题，

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