# -*- coding: utf-8 -*-
"""
作用：读取 documents 下的企业文档，切分成"知识碎片"，供后续向量化入库。
效果：运行后打印每个碎片的序号、来源文件和前 50 字预览。。。。
"""
from langchain_community.document_loaders import TextLoader  # 作用：读取文本文件；效果：把 txt 读成文档对象。
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 作用：智能切分文本；效果：按语义切成小块。

# 作用：定义两个文档路径；效果：loader 能定位文件。
FILE_PATHS = [
    "documents/employee_handbook.txt",
    "documents/reimbursement_policy.txt",
]

# 作用：配置切分器；效果：按 200 字一块、重叠 50 字切分。
# chunk_size=块大小；chunk_overlap=相邻块重叠字数（防止断句丢失语义）。
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)


def load_all_docs() -> list:
    """作用：读取所有文档并拼接；效果：返回全部文档对象的列表。"""
    all_docs = []  # 作用：创建空列表；效果：用于收集每个文件读出的文档。
    for path in FILE_PATHS:  # 作用：遍历每个文件路径；效果：逐个读取。
        loader = TextLoader(path, encoding="utf-8")  # 作用：用 UTF-8 读取文件；效果：避免中文乱码。
        all_docs.extend(loader.load())  # 作用：把读到的文档加入总列表；效果：合并所有文档。
    return all_docs


def main():
    # 作用：读取全部原始文档；效果：得到未切分的文档列表。
    docs = load_all_docs()
    print(f"读取到 {len(docs)} 份原始文档")

    # 作用：执行切分；效果：大文档变成多个 200 字左右的小块。
    chunks = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 个知识碎片\n")

    # 作用：预览前 5 个碎片；效果：对比相邻碎片的头尾，看到 overlap 效果。
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- 碎片 {i + 1} (来源: {chunk.metadata.get('source', '未知')}) ---")
        print(f"[开头] {chunk.page_content[:60]}")
        print(f"[结尾] {chunk.page_content[-60:]}\n")


if __name__ == "__main__":
    main()