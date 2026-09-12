# -*- coding: utf-8 -*-
"""
作用：云 API 最小验证脚本。
效果：① 验证 DeepSeek 能回话  ② 验证百炼 embedding 返回 1024 维向量  ③ 打印耗时便于观察。
"""
import time  # 作用：计时；效果：知道每次调用大概耗时多少。

from providers import get_chat_model, get_embedding  # 作用：复用模型层工厂；效果：顺便验证它可被外部正常引入。

QUESTION = "用一句话说明什么是 RAG。"  # 作用：固定测试问题；效果：每次运行输入一致，便于对比耗时与回答。

def test_chat() -> None:
    """作用：验证对话模型链路；效果：能收到非空回答即通过。"""
    llm = get_chat_model()  # 作用：取默认 provider；效果：拿到 DeepSeek + qwen 的主备链。
    start = time.perf_counter()  # 作用：记录起始时刻；效果：后面用来算耗时。
    answer = llm.invoke(QUESTION).content  # 作用：发一次真实请求；效果：拿到 AIMessage 里的正文文本。
    cost = time.perf_counter() - start  # 作用：算耗时；效果：单位秒。
    print(f"【对话模型】耗时 {cost:.2f} 秒")  # 作用：打印耗时；效果：:.2f 保留两位小数。
    print(f"【回答】{answer}")  # 作用：打印答案；效果：肉眼确认有内容且是中文。
    if not answer:  # 作用：兜住"空回答"；效果：把思考模式导致的空 content 明确暴露出来。
        raise AssertionError("回答为空 —— 检查是否开了思考模式，或模型名是否正确")

def test_embedding() -> None:
    """作用：验证向量化链路；效果：能拿到 1024 维向量即通过。"""
    emb = get_embedding()  # 作用：取百炼向量模型；效果：DeepSeek 不提供 embedding，只能走百炼。
    start = time.perf_counter()  # 作用：记录起始时刻。
    vector = emb.embed_query(QUESTION)  # 作用：把单条文本转向量；效果：返回一个浮点数列表。
    cost = time.perf_counter() - start  # 作用：算耗时。
    print(f"【embedding】耗时 {cost:.2f} 秒")  # 作用：打印耗时。
    print(f"【向量维度】{len(vector)}")  # 作用：打印维度；效果：确认是不是 1024。
    print(f"【前 5 个值】{vector[:5]}")  # 作用：切片取前 5 个数；效果：确认元素确实是浮点数。
    if len(vector) != 1024:  # 作用：维度断言；效果：建库前必须锁定维度。
        raise AssertionError(f"维度不是 1024 而是 {len(vector)}，建库前必须查清，否则检索会错位")
    print("维度校验通过：1024")  # 作用：明确告知通过。

if __name__ == "__main__":
    test_chat()  # 作用：先测对话。
    test_embedding()  # 作用：再测向量化。