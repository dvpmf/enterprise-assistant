# -*- coding: utf-8 -*-
"""
作用：Agent 的原子工具集 —— 把项目能力封装成模型能"点名调用"的函数。
效果：每个函数经 @tool 装饰后自动生成 JSON Schema（name / description / parameters），
      随请求发给模型；模型只负责"说要调哪个、参数是什么"，真正执行的是本地 Python 代码。
"""
from typing import Annotated  # 作用：给类型附带说明文字；效果：说明会自动进 JSON Schema 的参数描述，模型靠它填对参数。

from langchain_core.tools import tool  # 作用：把普通函数变成工具；效果：自动推断 name + description + parameters。

from retriever import search  # 作用：复用检索层；效果：RAG 能力成为 Agent 的第一个工具。

# ---- 差旅住宿标准（与 documents/ 里的制度保持一致）----
STANDARDS = {
    "一线": {"hotel": 500, "allowance": 120},
    "二线": {"hotel": 350, "allowance": 80},
    "其他": {"hotel": 260, "allowance": 60},
}  # 作用：把制度里的数字变成代码常量；效果：计算工具据此算钱。

# ---- 员工信息（演示用假数据，不接真实数据库）----
EMPLOYEES = {
    "E1001": {"name": "张伟", "department": "技术部", "title": "后端工程师", "manager": "李强", "join_date": "2024-07-01", "annual_leave": 10},
    "E1002": {"name": "王芳", "department": "人力资源部", "title": "HRBP", "manager": "赵敏", "join_date": "2023-03-15", "annual_leave": 15},
    "E1003": {"name": "刘洋", "department": "财务部", "title": "会计", "manager": "陈静", "join_date": "2025-01-06", "annual_leave": 5},
}  # 作用：字典当"假数据库"；效果：演示工具能查结构化数据，且不依赖外部服务。

@tool
def search_knowledge_base(
    question: Annotated[str, "要查询的问题，用自然语言描述，例如「出差住宿标准是多少」"],
) -> str:
    """在企业内部知识库中检索资料。当问题涉及公司制度、员工手册、报销规定等内部文档时使用，返回最相关的资料片段与来源文件名。"""
    docs = search(question)  # 作用：走两级检索（粗筛 + 精排）；效果：拿到最相关的 3 条。
    if not docs:  # 作用：兜住空结果；效果：告诉模型"查不到"，而不是让它瞎编。
        return "知识库中未检索到相关资料。"
    blocks = []  # 作用：收集文本块。
    for i, doc in enumerate(docs, start=1):  # 作用：逐条格式化。
        meta = doc.metadata  # 作用：取出处信息。
        blocks.append(f"[资料{i} 来源:{meta.get('source', '未知')} 页码:{meta.get('page', '-')}]\n{doc.page_content}")  # 作用：拼来源 + 正文；效果：模型能照着标注出处。
    return "\n\n".join(blocks)

@tool
def calc_reimbursement(
    city_tier: Annotated[str, "城市档位，只能是「一线」「二线」「其他」三者之一"],
    nights: Annotated[int, "住宿晚数，正整数"],
    actual_amount: Annotated[float | None, "实际住宿花费（元）。传了就顺便算超额自付，不传就只算标准上限"] = None,
) -> str:
    """计算差旅住宿费可以报销的金额（住宿费上限 + 出差补贴）。用户问「出差几天能报多少」「住 3 晚能报多少」这类需要算数的场景时使用。"""
    if city_tier not in STANDARDS:  # 作用：参数校验；效果：把非法值变成一句提示返回，让模型自己改参数重试（而不是抛异常中断）。
        return f"城市档位「{city_tier}」无效，只支持：一线 / 二线 / 其他。"
    if nights <= 0:  # 作用：兜住非正数。
        return "住宿晚数必须是正整数。"
    std = STANDARDS[city_tier]  # 作用：取出该档位标准。
    hotel_limit = std["hotel"] * nights  # 作用：住宿费上限；效果：超出部分要自付。
    allowance = std["allowance"] * nights  # 作用：出差补贴合计。
    lines = [
        f"城市档位：{city_tier}",
        f"住宿标准：{std['hotel']} 元/晚 × {nights} 晚 = 可报上限 {hotel_limit} 元",
        f"出差补贴：{std['allowance']} 元/天 × {nights} 天 = {allowance} 元",
    ]  # 作用：列出计算过程；效果：答案可核对，面试演示更好看。
    if actual_amount is not None:  # 作用：用户给了实际花费就算超额。
        reimbursable_hotel = min(actual_amount, hotel_limit)  # 作用：取小值；效果：最多只按标准上限报。
        lines.append(f"实际住宿花费：{actual_amount} 元 → 可报 {reimbursable_hotel} 元，自付 {actual_amount - reimbursable_hotel} 元")
    total_hotel = min(actual_amount, hotel_limit) if actual_amount is not None else hotel_limit  # 作用：确定住宿可报金额。
    lines.append(f"合计可报：{total_hotel + allowance} 元（住宿 + 补贴）")
    return "\n".join(lines)

@tool
def get_employee_info(
    employee_id: Annotated[str, "员工工号，格式如 E1001"],
) -> str:
    """按工号查询员工基本信息（姓名、部门、职位、直属上级、入职日期、年假余额）。"""
    info = EMPLOYEES.get(employee_id.upper())
    # 作用：查字典；效果：工号大小写不敏感。
    if not info:  # 作用：兜住查不到的情况。
        return f"未找到工号 {employee_id} 对应的员工记录，请确认工号是否正确。"
    return (
        f"工号：{employee_id.upper()}\n"
        f"姓名：{info['name']}\n"
        f"部门：{info['department']}\n"
        f"职位：{info['title']}\n"
        f"直属上级：{info['manager']}\n"
        f"入职日期：{info['join_date']}\n"
        f"年假余额：{info['annual_leave']} 天"
    )

TOOLS = [search_knowledge_base, calc_reimbursement, get_employee_info]
# 作用：工具清单；效果：给模型选、给 LangGraph 挂、给 MCP 注册，全用这一个列表。

if __name__ == "__main__":
    # 作用：自检；效果：直接本地调用每个工具（不经过模型），验证工具本身是好的。
    import json  # 作用：格式化打印 JSON Schema。

    print("工具清单：", [t.name for t in TOOLS])

    print("\n---- ① 知识库检索 ----")
    print(search_knowledge_base.invoke({"question": "出差住宿标准是多少"}))

    print("\n---- ② 报销计算 ----")
    print(calc_reimbursement.invoke({"city_tier": "一线", "nights": 3, "actual_amount": 1500}))

    print("\n---- ③ 员工信息 ----")
    print(get_employee_info.invoke({"employee_id": "E1001"}))

    print("\n---- 模型实际看到的 JSON Schema（以计算工具为例）----")
    print(json.dumps(calc_reimbursement.args_schema.model_json_schema(), ensure_ascii=False, indent=2))  # 作用：打印"发给模型的参数说明书"；效果：看清 description 与 parameters 到底长什么样。