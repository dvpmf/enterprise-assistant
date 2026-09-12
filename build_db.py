# -*- coding: utf-8 -*-
"""
作用：重建向量库 —— 把 documents/ 下所有格式的文档切分、向量化后写入 chroma_db/。
效果：运行后 chroma_db/ 是"用百炼 embedding 重新生成"的全新向量库。
注意：换 embedding 模型后必须重跑本脚本（新旧向量不在同一向量空间，旧库检索会失效）。
"""
import shutil  # 作用：递归删除目录；效果：一次性清空旧库，避免新旧向量混在一起。
from pathlib import Path  # 作用：路径操作；效果：判断目录是否存在。

from langchain_chroma import Chroma  # 作用：向量库连接器；效果：存向量 + 支持相似度检索。
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 作用：文本切分器；效果：把长文档切成小碎片。

from loaders import load_directory  # 作用：多格式摄入；效果：txt/md/pdf/图片 一次性全加载。
from providers import get_embedding  # 作用：向量化模型工厂；效果：拿到百炼 qwen3.7-text-embedding。

DOC_DIR = "documents"  # 作用：素材目录。
PERSIST_DIR = "chroma_db"  # 作用：向量库落盘目录；效果：重启不丢数据。
CHUNK_SIZE = 200  # 作用：每块最大字符数；效果：与 Day3 保持一致，便于对比。
CHUNK_OVERLAP = 50  # 作用：相邻块重叠字符数；效果：防止关键句被切断。

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)  # 作用：切分器实例；效果：全脚本复用同一个。

def reset_store() -> None:
    """作用：删除旧向量库；效果：保证库里只保留本次新建的向量。"""
    path = Path(PERSIST_DIR)  # 作用：转 Path 对象。
    if path.exists():  # 作用：判断旧库是否存在。
        shutil.rmtree(path)  # 作用：递归删除整个目录；效果：连同子文件一起清掉。
        print(f"已删除旧向量库：{PERSIST_DIR}/")
    else:
        print("未发现旧向量库，直接新建")

def build() -> Chroma:
    """作用：执行完整重建流程；效果：返回建好的持久化向量库对象。

    返回：
        Chroma: 已落盘的向量库
    """
    docs = load_directory(DOC_DIR)  # 作用：加载全部文档；效果：多格式差异在这一步被消灭。
    print(f"读取到 {len(docs)} 个 Document")

    chunks = splitter.split_documents(docs)  # 作用：切分；效果：整篇文档拆成 200 字左右的碎片。
    print(f"切分成 {len(chunks)} 个碎片")

    vectorstore = Chroma.from_documents(  # 作用：向量化 + 入库一步完成；效果：生成 chroma_db/。
        documents=chunks,  # 作用：待入库碎片。
        embedding=get_embedding(),  # 作用：百炼 embedding；效果：内部自动按 20 条一批发请求（我们在 providers 里设的 chunk_size）。
        persist_directory=PERSIST_DIR,  # 作用：指定落盘目录。
    )
    print(f"向量库已重建：{PERSIST_DIR}/")
    return vectorstore

def preview(vectorstore: Chroma, query: str = "如何报销差旅住宿费", k: int = 3) -> None:
    """作用：建完库立刻自测一次检索；效果：当场确认新库能查，避免"库建了但检索失效"。

    参数：
        vectorstore: 刚建好的向量库
        query: 测试问题
        k: 返回几条碎片
    """
    results = vectorstore.similarity_search(query, k=k)  # 作用：语义检索；效果：返回最相近的碎片列表。
    print(f"\n---- 自测检索：{query} ----")
    for i, doc in enumerate(results, start=1):  # 作用：编号从 1 开始；效果：打印更符合阅读习惯。
        meta = doc.metadata  # 作用：取出元数据。
        preview_text = doc.page_content[:60].replace("\n", " ")  # 作用：截前 60 字并把换行换成空格；效果：预览不跳行。
        print(f"[{i}] 来源={meta.get('source')} 类型={meta.get('type')} 页码={meta.get('page', '-')}")
        print(f"    {preview_text}...")

if __name__ == "__main__":
    reset_store()  # 作用：先清旧库；效果：这是"换 embedding 必须重建"的落地动作。
    vs = build()  # 作用：重建。
    preview(vs)  # 作用：立刻自测。
