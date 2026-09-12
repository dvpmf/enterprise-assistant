# -*- coding: utf-8 -*-
"""
作用：用 LangGraph 把 Function Calling 的 for 循环变成一张"图"。
效果：START → agent 节点（模型"想"）→ 条件边（要不要调工具）→ tools 节点（执行）→ 回到 agent，
      直到模型不再要求调工具，走到 END。多轮对话靠 checkpointer + thread_id。
"""
from functools import lru_cache  # 作用：缓存；效果：模型与图各构建一次。

from langchain_core.messages import ToolMessage  # 作用：用于统计工具结果条数。
from langgraph.checkpoint.memory import InMemorySaver  # 作用：内存版存档器；效果：多轮对话记住上下文（重启即丢，生产换 PostgresSaver）。
from langgraph.graph import END, START, MessagesState, StateGraph  # 作用：图的核心构件。
from langgraph.prebuilt import ToolNode, tools_condition  # 作用：现成的"工具执行节点"与"条件判断函数"。

from providers import get_chat_model  # 作用：模型层工厂。
from tools import TOOLS  # 作用：工具清单。

SYSTEM_PROMPT = (
    "你是[企业智能助手]，可以查企业知识库、计算差旅报销、查询员工信息。\n"
    "规则：1. 涉及公司制度或数字计算时，必须调用工具，不要凭记忆回答；\n"
    "2. 回答简洁专业，先给结论再给依据，并注明资料出处；\n"
    "3. 工具查不到的内容，明确说'资料中未找到相关信息'。"
)  # 作用：人设与约束；效果：强制模型"先查再答"。

@lru_cache(maxsize=None)
def get_llm_with_tools():
    """作用：构建"绑定了工具的模型"（且保留主备降级）；效果：只构建一次，复用连接池。

    返回：
        Runnable: 带工具的主备链模型
    """
    primary = get_chat_model(with_fallback=False)  # 作用：拿裸 DeepSeek；效果：裸模型才有 bind_tools 方法。
    fallback = get_chat_model("qwen")  # 作用：拿百炼 qwen 作备用。
    return primary.bind_tools(TOOLS).with_fallbacks([fallback.bind_tools(TOOLS)])  # 作用：两个模型都绑好工具再串主备链；效果：工具调用能力与容灾能力同时保住。

def agent_node(state: MessagesState) -> dict:
    """作用：agent 节点 —— 整个图的"大脑"；效果：把当前所有消息发给模型，返回它这轮的回复。

    参数：
        state: 共享状态，形如 {"messages": [...]}
    返回：
        dict: 状态增量，{"messages": [AIMessage]}（可能是最终答案，也可能是"要求调工具"）
    """
    response = get_llm_with_tools().invoke(state["messages"])  # 作用：调模型；效果：拿到 AIMessage。
    return {"messages": [response]}  # 作用：只返回增量；效果：LangGraph 按 add_messages 规则追加进 state["messages"]。

def build_graph(with_memory: bool = True):
    """作用：组装并编译图；效果：返回可直接 invoke 的图对象。

    参数：
        with_memory: 是否挂存档器；效果：挂了才能用 thread_id 做多轮对话
    返回：
        CompiledStateGraph: 编译后的图
    """
    builder = StateGraph(MessagesState)  # 作用：声明状态结构；效果：MessagesState = {"messages": list} + 累加规则。

    builder.add_node("agent", agent_node)  # 作用：注册 agent 节点。
    builder.add_node("tools", ToolNode(TOOLS))  # 作用：注册工具节点；效果：ToolNode 自动执行 AIMessage 里的 tool_calls 并产出 ToolMessage。

    builder.add_edge(START, "agent")  # 作用：入口边；效果：图一启动先进 agent。
    builder.add_conditional_edges(  # 作用：条件边；效果：按判断函数的返回值决定下一步走哪。
        "agent",
        tools_condition,  # 作用：官方现成的判断函数；效果：有 tool_calls → "tools"，没有 → "__end__"。
        {"tools": "tools", END: END},  # 作用：把返回值映射到真实节点。
    )
    builder.add_edge("tools", "agent")  # 作用：回边；效果：工具执行完回到 agent 再"想"一次 —— 这就是 ReAct 循环。

    return builder.compile(checkpointer=InMemorySaver() if with_memory else None)  # 作用：编译成可执行图。

if __name__ == "__main__":
    graph = build_graph()  # 作用：建图。
    config = {"configurable": {"thread_id": "demo-1"}, "recursion_limit": 12}  # 作用：thread_id 决定记忆归属；recursion_limit 防死循环。

    questions = [  # 作用：第二问专门验证记忆。
        "公司出差住宿标准是多少？我住 3 晚花了 1500，能报多少？",
        "还记得我上一个问题问的是什么吗？",
    ]

    for question in questions:
        print("=" * 64)
        print("用户：", question)
        result = graph.invoke({"messages": [("system", SYSTEM_PROMPT), ("user", question)]}, config)  # 作用：跑一次图；效果：返回完整状态。
        tool_count = sum(1 for m in result["messages"] if isinstance(m, ToolMessage))  # 作用：统计工具调用条数。
        print("助手：", result["messages"][-1].content)
        print(f"（累计消息 {len(result['messages'])} 条，其中工具结果 {tool_count} 条）")
        print()