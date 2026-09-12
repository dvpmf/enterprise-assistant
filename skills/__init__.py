# -*- coding: utf-8 -*-
"""
作用：技能注册表 —— 集中登记所有 Skill，并提供按名字取技能/取图的能力。
效果：新增技能只需在这里加一行。
"""
from functools import lru_cache  # 作用：缓存；效果：每个技能的图只编译一次。

from skills.base import Skill, build_skill_graph, resolve_tools  # 作用：技能定义与建图。
from skills.onboarding_skill import ONBOARDING_SKILL  # 作用：入职技能。
from skills.reimbursement_skill import REIMBURSEMENT_SKILL  # 作用：报销技能。

SKILLS: dict[str, Skill] = {  # 作用：技能注册表；效果：按 name 查找。
    REIMBURSEMENT_SKILL.name: REIMBURSEMENT_SKILL,
    ONBOARDING_SKILL.name: ONBOARDING_SKILL,
}

def get_skill(name: str) -> Skill:
    """作用：按名字取技能；效果：名字写错时给出可选列表。

    参数：
        name: 技能标识
    返回：
        Skill: 技能包
    异常：
        KeyError: 技能不存在
    """
    if name not in SKILLS:
        raise KeyError(f"未知技能 {name!r}，可选：{list(SKILLS)}")
    return SKILLS[name]

@lru_cache(maxsize=None)
def get_skill_graph(name: str):
    """作用：按技能名取（并缓存）对应的 Agent 图。

    参数：
        name: 技能标识
    返回：
        CompiledStateGraph: 该技能的图
    """
    return build_skill_graph(get_skill(name))

def demo() -> None:
    """作用：演示"同一套 Agent 框架 + 不同技能"时，人设与可用工具的切换。"""
    print("---- 注册表中的技能 ----")
    for skill in SKILLS.values():
        print(f"* {skill.name}: {skill.description}")
        print(f"  可用工具：{[t.name for t in resolve_tools(skill)]}")  # 作用：打印工具子集；效果：肉眼可见地"工具范围被约束了"。

    graph = get_skill_graph("reimbursement")  # 作用：加载报销技能。
    config = {"configurable": {"thread_id": "skill-demo"}, "recursion_limit": 12}
    question = "E1002 是谁？"  # 作用：故意问一个"报销技能不该管"的问题；效果：验证工具约束真的生效。
    print("\n---- 报销技能被问到越界问题 ----")
    print("提问：", question)
    result = graph.invoke(
        {"messages": [("system", get_skill("reimbursement").system_prompt), ("user", question)]},
        config,
    )
    print("回答：", result["messages"][-1].content)

if __name__ == "__main__":
    demo()  # 作用：直接跑本文件时也执行演示。