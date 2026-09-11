# -*- coding: utf-8 -*-
"""
作用：把 Day4 的 RAG 问答能力包装成 HTTP API。
效果：启动后，POST 请求 http://127.0.0.1:8000/ask 传入问题，返回模型基于文档的回答。
"""
import os  # 作用：读取环境变量；效果：让模型地址能在不同环境间切换。
from fastapi import FastAPI  # 作用：引入框架；效果：创建 Web 应用。
from pydantic import BaseModel  # 作用：定义请求体结构；效果：自动校验收到的 JSON 格式。
from langchain_ollama import ChatOllama, OllamaEmbeddings  # 作用：两位模型；效果：负责回答和转向量。
from langchain_chroma import Chroma  # 作用：连接向量库；效果：支持检索。

# ---- 配置区：和 Day4 rag_qa.py 保持一致 ----
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3.5:4b"
# 作用：读取 Ollama 服务地址；效果：本机跑用默认值，容器跑读环境变量（指向宿主机）。
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


# 作用：创建 embedding 对象；效果：检索时把问题转向量。
embedding = OllamaEmbeddings(model=EMBED_MODEL,base_url=OLLAMA_BASE_URL)

# 作用：创建对话模型并关思考；效果：回答稳定进 content。
llm = ChatOllama(model=CHAT_MODEL, reasoning=False, base_url=OLLAMA_BASE_URL)

# 作用：定义人设；效果：约束模型只依据资料回答。
SYSTEM_PROMPT = (
    "你是[企业智能助手]，负责依据提供的企业内部资料回答员工问题。\n"
    "规则：1. 只依据资料回答，资料没有的内容，明确说'资料中未找到相关信息'；\n"
    "2. 回答简洁专业，先给结论再给依据；\n"
    "3. 回答末尾注明资料出处（来源文件名）。"
)


# 作用：定义请求体结构；效果：客户端必须传 question 字段，类型为字符串。
class AskRequest(BaseModel):
    question: str  # 作用：声明字段；效果：收到 JSON 里没有 question 会自动报 422 错误。


# 作用：创建 FastAPI 应用；效果：app 即 Web 服务器对象。
app = FastAPI()


def retrieve(query: str, k: int = 3) -> list:
    """作用：检索向量库；效果：返回与问题最相关的碎片。"""
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    return vectorstore.similarity_search(query, k=k)


def build_prompt(query: str, docs: list) -> str:
    """作用：把资料拼进提示词；效果：让模型基于资料回答。"""
    context = "\n\n".join(
        f"[资料{i+1} 来源:{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    return f"{SYSTEM_PROMPT}\n\n{context}\n\n问题：{query}"


# 作用：注册 POST 路由 /ask；效果：客户端 POST 问题过来就执行此函数。
@app.post("/ask")
def ask(req: AskRequest):
    """作用：处理 /ask 请求；效果：检索→拼接→生成，返回回答和出处。"""
    # 作用：调用检索；效果：拿到相关碎片。
    docs = retrieve(req.question)
    # 作用：拼接提示词并让模型回答；效果：得到基于资料的回答。
    answer = llm.invoke(build_prompt(req.question, docs)).content
    # 作用：组装返回内容；效果：客户端拿到 answer + 出处。
    return {"answer": answer, "sources": [d.metadata.get("source", "未知") for d in docs]}


if __name__ == "__main__":
    import uvicorn  # 作用：引入启动器；效果：直接运行本文件即启动。
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)