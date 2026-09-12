# -*- coding: utf-8 -*-
"""
作用：Rerank 精排层 —— 调阿里云百炼 qwen3.7-text-rerank，对召回结果做二次打分排序。
效果：把"向量粗筛"出来的候选碎片按真实相关性重排，只留最相关的 top_n 条交给大模型。
"""
import os  # 作用：读环境变量。

import requests  # 作用：发 HTTP 请求；效果：rerank 的接口不在 OpenAI 兼容规范里，得用原生 HTTP 调用。
from dotenv import load_dotenv  # 作用：加载 .env。
from langchain_core.documents import Document  # 作用：统一文档容器；效果：入参和出参都用它，与检索链路无缝衔接。

load_dotenv()  # 作用：注入环境变量；效果：下面 os.getenv 才能读到 key。

# ---- 配置区 ----
API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 作用：百炼密钥。
if not API_KEY:  # 作用：快速失败；效果：缺 key 时 import 阶段就报错，而不是等请求发出去。
    raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY，请在项目根目录的 .env 中配置")

RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.7-text-rerank")  # 作用：精排模型名。

RERANK_URL = os.getenv(  # 作用：精排接口地址；效果：注意它是百炼【原生】路径，不是 compatible-mode。
    "DASHSCOPE_RERANK_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
)

def rerank_documents(query: str, documents: list[Document], top_n: int = 3) -> list[Document]:
    """作用：对候选碎片重新打分排序；效果：返回相关性最高的 top_n 条（已按分数从高到低排好）。

    参数：
        query: 用户问题，字符串
        documents: 向量检索召回的候选碎片列表
        top_n: 精排后保留几条，默认 3
    返回：
        list[Document]: 精排后的碎片；每条会在 metadata 里带上 rerank_score
    异常：
        requests.HTTPError: 接口返回非 2xx 时抛出
    """
    if not documents:  # 作用：兜住空输入；效果：不发无意义的请求。
        return []

    payload = {  # 作用：按百炼原生协议的报文结构组装请求体。
        "model": RERANK_MODEL,  # 作用：模型名。
        "input": {  # 作用：输入区；效果：query 与 documents 都放在这一层（和 qwen3-rerank 的扁平结构不同）。
            "query": query,
            "documents": [doc.page_content for doc in documents],  # 作用：只取正文；效果：接口只要字符串列表。
        },
        "parameters": {"top_n": top_n},  # 作用：只要前 top_n 条；效果：减少返回数据量。
    }
    headers = {  # 作用：认证与内容类型；效果：两个都是必填头。
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(RERANK_URL, headers=headers, json=payload, timeout=60)  # 作用：发请求；效果：json= 自动序列化并加 Content-Type。
    resp.raise_for_status()  # 作用：非 2xx 直接抛异常；效果：不把错误响应当正常结果往下传。

    results = resp.json()["output"]["results"]  # 作用：取排序结果；效果：形如 [{"index": 0, "relevance_score": 0.93}, ...]，已按分数降序。

    reranked: list[Document] = []  # 作用：收集最终结果。
    for item in results:
        doc = documents[item["index"]]  # 作用：用 index 回到原始候选列表取原文；效果：接口默认不返回文档原文，只能靠索引映射。
        doc.metadata["rerank_score"] = round(item["relevance_score"], 4)  # 作用：把分数记进元数据；效果：方便做前后对比与排障。
        reranked.append(doc)
    return reranked

if __name__ == "__main__":
    # 作用：自检；效果：故意混入两条无关碎片，看精排能不能把它们排到后面去。
    test_query = "如何报销差旅住宿费"
    candidates = [
        Document(page_content="公司实行五天工作制，上午 9:00 上班，下午 18:00 下班。", metadata={"source": "employee_handbook.txt"}),
        Document(page_content="一线城市住宿标准 500 元/晚，住宿费需凭正规发票报销，超额部分自理。", metadata={"source": "sample_policy.pdf"}),
        Document(page_content="量子计算是计算科学的前沿领域，具有广阔的应用前景。", metadata={"source": "无关文档.txt"}),
        Document(page_content="出差补贴：一线城市 120 元/天，二线城市 80 元/天。", metadata={"source": "reimbursement_policy.txt"}),
    ]

    print("---- 精排前（原始顺序）----")
    for i, d in enumerate(candidates, start=1):
        print(f"[{i}] {d.metadata['source']}: {d.page_content[:26]}")

    top_docs = rerank_documents(test_query, candidates, top_n=2)  # 作用：只留 2 条。

    print(f"\n---- 精排后（只留 {len(top_docs)} 条）----")
    for i, d in enumerate(top_docs, start=1):
        print(f"[{i}] 得分={d.metadata.get('rerank_score')} {d.metadata['source']}: {d.page_content[:26]}")