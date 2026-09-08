# -*- coding: utf-8 -*-
"""
作用：实现完整 RAG 问答：检索知识库 → 资料拼进提示词 → 交给模型生成回答。
效果：输入问题后，模型基于检索到的企业文档回答，并附上资料出处。
"""
from langchain_ollama import ChatOllama, OllamaEmbeddings  # 作用：对话模型+embedding模型；效果：一个负责说，一个负责翻译。
from langchain_chroma import Chroma  # 作用：连接向量库；效果：支持相似度检索。

# ---- 配置区：三位"员工" ----
PERSIST_DIR = "chroma_db"  # 向量库位置
EMBED_MODEL = "nomic-embed-text"  # 翻译官（文字→向量）
CHAT_MODEL = "qwen3.5:4b"  # 发言人（负责回答）

# 作用：创建 embedding 对象；效果：检索时把问题转成向量。
embedding = OllamaEmbeddings(model=EMBED_MODEL)

# 作用：创建对话模型并关闭思考模式；效果：回答直接进 content，不拖泥带水。
llm = ChatOllama(model=CHAT_MODEL, reasoning=False)

# 作用：定义系统人设；效果：约束模型只依据资料回答，不编造。
SYSTEM_PROMPT = (
    "你是[企业智能助手]，负责依据提供的企业内部资料回答员工问题。\n"
    "规则：1. 只依据资料回答，资料没有的内容，明确说'资料中未找到相关信息'；\n"
    "2. 回答简洁专业，先给结论再给依据；\n"
    "3. 回答末尾注明资料出处（来源文件名）。"
)


def retrieve(query: str, k: int = 3) -> list:
    """
    作用：从向量库检索与问题最相关的碎片。
    参数 query：问题；k：返回条数。
    效果：返回最相关的文档碎片列表。
    """
    # 作用：加载已有向量库；效果：无需重复向量化入库。
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    return vectorstore.similarity_search(query, k=k)


def build_prompt(query: str, docs: list) -> str:
    """
    作用：把检索到的资料拼进提示词，形成给模型的完整指令。
    参数 query：问题；docs：检索到的碎片。
    效果：返回"资料+问题"的完整提示词字符串。
    """
    # 作用：把每块碎片拼成带来源的文本块；效果：模型能对应看到出处。
    context = "\n\n".join(
        f"[资料{i+1} 来源:{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    # 作用：组装完整提示词；效果：系统人设+资料+问题一次发给模型。
    return f"{SYSTEM_PROMPT}\n\n{context}\n\n问题：{query}"


def ask(query: str) -> str:
    """
    作用：执行完整 RAG 问答。
    参数 query：用户问题。
    效果：返回模型基于检索资料的回答。
    """
    # 作用：第一步检索；效果：召回最相关的碎片。
    docs = retrieve(query)
    # 作用：第二步拼提示词；效果：资料注入问题上下文。
    prompt = build_prompt(query, docs)
    # 作用：第三步生成；效果：模型基于资料回答（单轮，无记忆）。
    reply = llm.invoke(prompt).content
    return reply


def main():
    print("企业 RAG 助手已启动（输入 exit 退出）")
    while True:
        query = input("你：")
        if query.strip().lower() == "exit":
            break
        answer = ask(query)
        print("助手：", answer)


if __name__ == "__main__":
    main()