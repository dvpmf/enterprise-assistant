# -*- coding: utf-8 -*-
"""
作用：检索层 —— 把「向量库连接 + 粗筛 + 精排 + 拼提示词」集中到一处。
效果：api.py（HTTP 入口）与 tools.py（Agent 工具）共用同一套检索逻辑，
      逻辑只写一遍；同时避免 api <- tools 的循环导入。
"""
from functools import lru_cache  # 作用：函数级缓存；效果：向量库连接全进程只建一次。

from langchain_chroma import Chroma  # 作用：向量库连接器；效果：存向量 + 相似度检索。

from providers import get_embedding  # 作用：向量化模型工厂；效果：拿到百炼 embedding。
from rerank import rerank_documents  # 作用：精排层；效果：对粗筛结果二次排序。

PERSIST_DIR = "chroma_db"  # 作用：向量库落盘目录。
RECALL_K = 10  # 作用：粗筛召回条数；效果：宁多勿漏，交给精排去筛。
RERANK_TOP_N = 3  # 作用：精排后保留条数；效果：只把最相关的 3 条塞进提示词。

SYSTEM_PROMPT = (
    "你是[企业智能助手]，负责依据提供的企业内部资料回答员工问题。\n"
    "规则：1. 只依据资料回答，资料没有的内容，明确说'资料中未找到相关信息'；\n"
    "2. 回答简洁专业，先给结论再给依据；\n"
    "3. 回答末尾注明资料出处（来源文件名）。"
)  # 作用：人设与约束；效果：让模型贴着资料回答，不自由发挥。


@lru_cache(maxsize=None)
def get_vectorstore() -> Chroma:
    """作用：单例向量库；效果：整个进程只建立一次连接，避免每个请求重复连库。

    返回：
        Chroma: 已连接的向量库对象
    """
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding())  # 作用：用百炼 embedding 连库；效果：查询与入库在同一向量空间。


def retrieve(query: str, k: int = RECALL_K) -> list:
    """作用：向量粗筛；效果：返回与问题语义最近的 k 条候选碎片。

    参数：
        query: 用户问题
        k: 召回条数
    返回：
        list[Document]: 候选碎片
    """
    return get_vectorstore().similarity_search(query, k=k)  # 作用：相似度检索。


def search(query: str, top_n: int = RERANK_TOP_N) -> list:
    """作用：完整两级检索（粗筛 + 精排）；效果：返回最相关的 top_n 条碎片。

    参数：
        query: 用户问题
        top_n: 精排后保留条数
    返回：
        list[Document]: 精排后的碎片（metadata 里带 rerank_score）
    """
    candidates = retrieve(query, RECALL_K)  # 作用：第一步粗筛；效果：拿到 10 条候选。
    if not candidates:  # 作用：空库或没命中；效果：直接返回空，不浪费精排调用。
        return []
    return rerank_documents(query, candidates, top_n)  # 作用：第二步精排；效果：候选压到 top_n 条。


def build_prompt(query: str, docs: list) -> str:
    """作用：把精排后的资料拼进提示词；效果：让模型基于资料回答。

    参数：
        query: 用户问题
        docs: 精排后的碎片
    返回：
        str: 完整提示词
    """
    context = "\n\n".join(  # 作用：把多条碎片拼成一段上下文；效果：空行分隔更易读。
        f"[资料{i} 来源:{d.metadata.get('source', '未知')} "
        f"页码:{d.metadata.get('page', '-')} 精排得分:{d.metadata.get('rerank_score', '-')}]\n{d.page_content}"
        for i, d in enumerate(docs, start=1)  # 作用：编号从 1 开始。
    )
    return f"{SYSTEM_PROMPT}\n\n{context}\n\n问题：{query}"  # 作用：人设 + 资料 + 问题。
