# -*- coding: utf-8 -*-
"""
作用：把切分好的知识碎片向量化并存入 ChromaDB 向量库。
效果：运行后在 chroma_db/ 目录生成持久化的向量库，供检索使用。
"""
from langchain_ollama import OllamaEmbeddings  # 作用：连接 Ollama 的 embedding 模型；效果：把文字变成向量。
from langchain_chroma import Chroma  # 作用：ChromaDB 向量库连接器；效果：存向量 + 支持按距离检索。
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 作用：定义持久化目录；效果：向量库存到本地，重启不丢。
PERSIST_DIR = "chroma_db"

# 作用：指定 embedding 模型；效果：用 nomic-embed-text 把文字翻译成向量。
embedding = OllamaEmbeddings(model="nomic-embed-text")

# 作用：配置切分器（与 split_docs.py 保持一致）；效果：保证入库碎片和之前预览的一致。
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

# 作用：列出要入库的文档；效果：循环读取两份企业文档。
FILE_PATHS = [
    "documents/employee_handbook.txt",
    "documents/reimbursement_policy.txt",
]


def load_and_split() -> list:
    """作用：读取文档并切分；效果：返回知识碎片列表。"""
    docs = []
    for path in FILE_PATHS:
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
    return splitter.split_documents(docs)


def main():
    # 作用：获取碎片；效果：得到待入库的小块文档。
    chunks = load_and_split()
    print(f"准备入库 {len(chunks)} 个碎片...")

    # 作用：向量化并存入 ChromaDB；效果：生成 chroma_db/ 持久化向量库。
    vectorstore = Chroma.from_documents(
        documents=chunks,  # 作用：传入碎片；效果：作为入库内容。
        embedding=embedding,  # 作用：传入翻译官；效果：每块文字自动转向量。
        persist_directory=PERSIST_DIR,  # 作用：指定落盘目录；效果：向量库保存到本地。
    )
    print(f"向量库已建立，存储位置：{PERSIST_DIR}")


if __name__ == "__main__":
    main()