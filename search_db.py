# -*- coding: utf-8 -*-
"""
作用：从已建立的 ChromaDB 向量库中检索与问题最相关的知识碎片。
效果：运行后打印与提问最相关的 3 个碎片及其来源。。。。
"""
from langchain_ollama import OllamaEmbeddings  # 作用：连接 embedding 模型；效果：把问题也转成向量（必须与入库用同一模型）。
from langchain_chroma import Chroma  # 作用：连接已有向量库；效果：支持相似度检索。

PERSIST_DIR = "chroma_db"  # 作用：指向库目录；效果：加载已建好的向量库。

embedding = OllamaEmbeddings(model="nomic-embed-text")  # 作用：指定翻译官；效果：查询时用同一模型对齐向量。


def search(query: str, top_k: int = 3) -> list:
    """
    作用：用问题检索最相关碎片。
    参数 query：用户问题；top_k：返回前几条。
    效果：返回最相关的碎片文档列表。
    """
    # 作用：加载已存在的向量库；效果：无需重新向量化。
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
    )
    # 作用：执行相似度检索；效果：找与问题向量"距离最近"的碎片。
    results = vectorstore.similarity_search(query, k=top_k)
    return results


def main():
    # 作用：定义测试问题；效果：验证检索能否命中"请假"相关碎片。
    question = input("请输入你的问题：")
    print("问题：", question)

    # 作用：调用检索；效果：拿到最相关的碎片。
    docs = search(question)

    # 作用：打印检索结果；效果：直观看到命中内容与来源。
    for i, doc in enumerate(docs):
        print(f"\n--- 结果 {i+1} (来源: {doc.metadata.get('source', '未知')}) ---")
        print(doc.page_content)


if __name__ == "__main__":
    main()