# -*- coding: utf-8 -*-
"""
作用：Skill 的通用定义与建图逻辑 —— Skill = 提示词 + 工具子集 + 流程 的打包。
效果：同一套 Agent 框架加载不同 Skill 时，人设与"能用的工具"跟着切换。
"""
from dataclasses import dataclass  # 作用：定义结构化数据；效果：Skill 的四个字段一眼看清、可做类型检查。

from langchain_core.tools import BaseTool  # 作用：工具类型注解。
from langgraph.checkpoint.memory import InMemorySaver  # 作用：记忆存档器。
from langgraph.graph import END, START, MessagesState, StateGraph  # 作用：图的核心构件。
from langgraph.prebuilt import ToolNode, tools_condition  # 作用：现成的工具节点与条件判断。

from providers import get_chat_model  # 作用：模型层工厂。
from tools import TOOLS  # 作用：全部工具；效果：技能从这里"挑子集"。

@dataclass(frozen=True)  # 作用：冻结的数据类；效果：字段不可改，避免运行中被意外修改。
class Skill:
    """作用：一个可复用技能包的定义。

    属性：
        name: 技能标识（英文，用于注册与查找）
        description: 技能说明（给人看，也用于后续"路由层判断该用哪个技能"）
        system_prompt: 该技能专属的人设与规则
        allowed_tools: 允许使用的工具名列表（★核心：约束工具范围，防止乱调）
    """
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str]

_TOOL_MAP = {t.name: t for t in TOOLS}  # 作用：工具名 → 工具对象。

def resolve_tools(skill: Skill) -> list[BaseTool]:
    """作用：把技能声明的工具名解析成真实工具对象；效果：名字写错时立刻报错，而不是等模型调用时才发现。

    参数：
        skill: 技能包
    返回：
        list[BaseTool]: 解析后的工具对象列表
    异常：
        ValueError: 声明了不存在的工具名
    """
    missing = [n for n in skill.allowed_tools if n not in _TOOL_MAP]  # 作用：先校验全部名字。
    if missing:
        raise ValueError(f"技能 {skill.name} 声明了不存在的工具：{missing}")
    return [_TOOL_MAP[n] for n in skill.allowed_tools]

def build_skill_graph(skill: Skill, with_memory: bool = True):
    """作用：按技能组装一张独立的 Agent 图；效果：这张图只会用该技能允许的工具与该技能的人设。

    参数：
        skill: 技能包
        with_memory: 是否挂记忆存档器
    返回：
        CompiledStateGraph: 编译后的图
    """
    tools = resolve_tools(skill)  # 作用：取工具子集；效果：其它工具压根不发给模型，它想调也调不到。
    primary = get_chat_model(with_fallback=False).bind_tools(tools)  # 作用：裸 DeepSeek + 工具子集。
    fallback = get_chat_model("qwen").bind_tools(tools)  # 作用：qwen 备用，同样只绑子集。
    llm = primary.with_fallbacks([fallback])  # 作用：绑完工具再串主备链。

    def agent_node(state: MessagesState) -> dict:
        """作用：agent 节点；效果：把当前消息发给模型，返回它这轮的回复。"""
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)  # 作用：建图；效果：每个技能有自己独立的图。
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))  # 作用：工具节点也只挂子集。
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver() if with_memory else None)