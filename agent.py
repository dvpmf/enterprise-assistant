# -*- coding: utf-8 -*-
"""
作用：把 LangGraph 图封装成对外的简单函数，供 FastAPI / Streamlit / 命令行调用。
效果：调用方只需要 run_agent("问题")，不用关心图、状态、工具这些内部细节。
"""
from functools import lru_cache  # 作用：缓存；效果：图只编译一次。

from langchain_core.messages import ToolMessage  # 作用：识别工具结果消息；效果：用来统计"这次调了哪些工具"。

from graph import SYSTEM_PROMPT, build_graph  # 作用：复用第 4 块定义的人设与建图函数。

DEFAULT_THREAD_ID = "default"  # 作用：默认会话标识。
RECURSION_LIMIT = 12  # 作用：图的最大步数；效果：防止工具反复调用导致死循环。

@lru_cache(maxsize=None)
def get_agent():
    """作用：单例图；效果：整个进程只编译一次，避免每个请求重建。

    返回：
        CompiledStateGraph: 带记忆的图对象
    """
    return build_graph(with_memory=True)

def run_agent(question: str, thread_id: str = DEFAULT_THREAD_ID) -> dict:
    """作用：跑一次 Agent 问答；效果：返回答案、用到的工具清单、消息条数。

    参数：
        question: 用户问题
        thread_id: 会话标识；效果：同一个 id 共享多轮记忆，不同 id 相互隔离
    返回：
        dict: {"answer": str, "tools_used": list[str], "messages": int}
    """
    graph = get_agent()  # 作用：拿单例图。
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}  # 作用：运行时配置；效果：thread_id 决定"记忆挂在哪根线上"。

    result = graph.invoke(  # 作用：跑图；效果：内部自动完成"想 → 调工具 → 再想"的循环。
        {"messages": [("system", SYSTEM_PROMPT), ("user", question)]},  # 作用：初始消息；效果：元组形式会被自动转成 SystemMessage / HumanMessage。
        config,
    )

    messages = result["messages"]  # 作用：取完整消息历史。
    tools_used = [m.name for m in messages if isinstance(m, ToolMessage)]  # 作用：收集工具名；效果：一眼看出 Agent 走了哪条路。

    return {
        "answer": messages[-1].content,  # 作用：最后一条就是模型最终回答。
        "tools_used": tools_used,
        "messages": len(messages),  # 作用：消息条数；效果：侧面反映"几步完成"。
    }

if __name__ == "__main__":
    # 作用：自检；效果：第二问用"他"指代，验证同一 thread_id 下记忆是否生效。
    for q in ["E1002 是谁？", "他属于哪个部门？"]:
        out = run_agent(q, thread_id="cli-1")  # 作用：固定 thread_id；效果：两问共享上下文。
        print("=" * 64)
        print("问题：", q)
        print("工具：", out["tools_used"])
        print("回答：", out["answer"])
        print(f"（消息 {out['messages']} 条）\n")