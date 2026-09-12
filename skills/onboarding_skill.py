# -*- coding: utf-8 -*-
"""作用：入职问答技能 —— 只允许查制度 + 查员工信息，不带计算工具。"""
from skills.base import Skill

ONBOARDING_SKILL = Skill(
    name="onboarding",
    description="处理入职与日常人事问题：公司制度查询、员工信息查询",
    system_prompt=(
        "你是[入职小助手]，帮助新员工了解公司制度与个人信息。\n"
        "规则：1. 制度类问题必须调用 search_knowledge_base 并注明出处；\n"
        "2. 查员工信息必须调用 get_employee_info；\n"
        "3. 只回答入职与人事相关问题。"
    ),
    allowed_tools=["search_knowledge_base", "get_employee_info"],  # 作用：不含 calc_reimbursement。
)