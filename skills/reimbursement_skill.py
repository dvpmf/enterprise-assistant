# -*- coding: utf-8 -*-
"""作用：报销咨询技能 —— 只允许查制度 + 算报销，防止模型跑去查员工信息。"""
from skills.base import Skill  # 作用：复用技能定义。

REIMBURSEMENT_SKILL = Skill(  # 作用：一个技能包实例；效果：人设与工具子集一起打包。
    name="reimbursement",
    description="处理差旅报销相关咨询：查报销制度、计算可报销金额",
    system_prompt=(
        "你是[报销助手]，专门处理员工的差旅与费用报销咨询。\n"
        "规则：1. 涉及金额计算必须调用 calc_reimbursement，不要心算；\n"
        "2. 涉及制度条款必须调用 search_knowledge_base，并注明出处文件；\n"
        "3. 只回答报销相关问题，其它问题礼貌说明不在职责范围内。"
    ),
    allowed_tools=["search_knowledge_base", "calc_reimbursement"],  # 作用：只给两个工具；效果：查员工信息这类工具对它不可见。
)