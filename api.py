# -*- coding: utf-8 -*-
"""
作用：把多格式 RAG 问答与 Agent 能力包装成 HTTP API。
效果：
    POST /ask    提问 → 召回 top10 → 精排 top3 → DeepSeek 生成答案 + 标注出处
    POST /agent  Agent 问答 → 模型自主决定调哪些工具（检索 / 报销计算 / 员工信息），多步完成任务
    POST /ingest 上传文件（txt/md/pdf/图片）→ 解析（含 OCR）→ 切分 → 百炼向量化 → 入库
    GET  /health 健康检查（顺便报告库里有多少条向量）
"""
import tempfile  # 作用：创建临时文件；效果：上传的内容先落临时文件，再交给 loaders 处理。
from pathlib import Path  # 作用：路径处理；效果：取扩展名、删临时文件。

from fastapi import FastAPI, File, HTTPException, UploadFile  # 作用：FastAPI 组件；效果：路由、文件上传、错误响应。
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 作用：切分器。
from pydantic import BaseModel  # 作用：请求/响应体结构；效果：自动校验 + 生成 /docs 文档。

from agent import run_agent  # 作用：Agent 门面；效果：/agent 端点直接复用整张 LangGraph 图。
from loaders import load_file  # 作用：多格式解析（含 OCR）。
from providers import get_chat_model  # 作用：模型层工厂；效果：拿到 DeepSeek + qwen 主备链。
from rerank import rerank_documents  # 作用：精排层。
from retriever import RECALL_K, RERANK_TOP_N, build_prompt, get_vectorstore, retrieve  # 作用：检索层；效果：向量库/粗筛/拼提示词全部复用，不重复实现。

# ---- 配置区 ----
MAX_UPLOAD_MB = 20  # 作用：上传大小上限（MB）。
ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # 作用：白名单；效果：拒绝乱七八糟的文件类型。

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)  # 作用：切分器实例；效果：入库碎片与建库时保持同样粒度。
llm = get_chat_model()  # 作用：模块级只建一次；效果：主备链全局复用，不每个请求重建连接池。


class AskRequest(BaseModel):
    """作用：/ask 的请求体；效果：客户端必须传 question 字段。"""
    question: str  # 作用：用户问题；效果：缺字段自动返回 422。

class AskResponse(BaseModel):
    """作用：/ask 的响应体；效果：结构固定，/docs 里能直接看到。"""
    answer: str  # 作用：模型答案。
    sources: list[dict]  # 作用：出处列表；效果：每条含 source/page/type/score。

class IngestResponse(BaseModel):
    """作用：/ingest 的响应体。"""
    source: str  # 作用：入库的文件名。
    chunks: int  # 作用：切出的碎片数量。
    message: str  # 作用：结果说明。


class AgentRequest(BaseModel):
    """作用：/agent 的请求体。"""
    question: str  # 作用：用户问题。
    thread_id: str = "default"  # 作用：会话标识；效果：同一个 id 共享多轮记忆，不同 id 相互隔离。


class AgentResponse(BaseModel):
    """作用：/agent 的响应体。"""
    answer: str  # 作用：最终答案。
    tools_used: list[str]  # 作用：本次调用了哪些工具；效果：演示与排障时一眼看出 Agent 走了哪条路。
    messages: int  # 作用：本轮消息总数；效果：侧面反映"几步完成"。

app = FastAPI(title="企业智能助手 API")  # 作用：创建应用；效果：所有路由挂在它身上。


@app.get("/health")
def health():
    """作用：健康检查；效果：确认服务可用 + 向量库可访问，返回向量条数。"""
    try:
        count = get_vectorstore()._collection.count()  # 作用：统计库里向量条数；效果：_collection 是 langchain-chroma 暴露的底层 chromadb 集合对象。
    except Exception as exc:  # 作用：兜住连接失败；效果：返回 503 而不是 500 崩溃栈。
        raise HTTPException(status_code=503, detail=f"向量库不可用：{exc}") from exc
    return {"status": "ok", "vectors": count}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """作用：问答主接口；效果：召回 → 精排 → 生成，返回答案与出处。

    参数：
        req: 含 question 的请求体
    返回：
        AskResponse: 答案 + 出处列表
    """
    candidates = retrieve(req.question, RECALL_K)  # 作用：第一步粗筛；效果：拿到 10 条候选。
    if not candidates:  # 作用：空库或没检索到内容；效果：直接给明确提示，不浪费模型调用。
        return AskResponse(answer="知识库为空或未检索到相关内容。", sources=[])

    top_docs = rerank_documents(req.question, candidates, RERANK_TOP_N)  # 作用：第二步精排；效果：候选从 10 条压到 3 条。

    answer = llm.invoke(build_prompt(req.question, top_docs)).content  # 作用：生成答案；效果：AIMessage 里取正文。

    sources = [  # 作用：组装出处；效果：比 Day5 只给文件名更细，带页码和精排得分。
        {
            "source": d.metadata.get("source", "未知"),
            "page": d.metadata.get("page"),
            "type": d.metadata.get("type"),
            "score": d.metadata.get("rerank_score"),
        }
        for d in top_docs
    ]
    return AskResponse(answer=answer, sources=sources)


@app.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest):
    """作用：Agent 问答接口；效果：模型自主决定调哪些工具、调几次，多步完成任务。

    参数：
        req: 含 question 与可选 thread_id 的请求体
    返回：
        AgentResponse: 答案 + 用到的工具清单 + 消息条数
    """
    result = run_agent(req.question, req.thread_id)  # 作用：交给 LangGraph 跑；效果：检索/计算/查员工全部由模型自主编排。
    return AgentResponse(**result)  # 作用：字典拆包成响应模型。


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    """作用：上传文件入库；效果：解析（含 OCR）→ 切分 → 向量化 → 追加进向量库。

    参数：
        file: multipart/form-data 上传的文件
    返回：
        IngestResponse: 文件名与入库碎片数
    异常：
        HTTPException: 类型不支持(400) / 文件过大(413) / 没解析出文字(422)
    """
    filename = file.filename or "unknown"  # 作用：取原始文件名；效果：兜住 filename 为空的情况。
    suffix = Path(filename).suffix.lower()  # 作用：取扩展名转小写；效果：.PDF 也认。
    if suffix not in ALLOWED_SUFFIXES:  # 作用：白名单校验。
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    data = file.file.read()  # 作用：读出上传内容的字节；效果：这里用同步读，因为本函数是同步端点。
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:  # 作用：大小校验；效果：1024*1024 是 1 MB 的字节数。
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB} MB 限制")

    tmp_path = None  # 作用：记录临时文件路径；效果：finally 里能清理。
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  # 作用：建临时文件；效果：保留扩展名，loaders 才能按类型分发。
            tmp.write(data)  # 作用：写入字节。
            tmp_path = Path(tmp.name)  # 作用：记下路径。
        docs = load_file(tmp_path)  # 作用：复用多格式解析器（含 OCR）。
        for doc in docs:
            doc.metadata["source"] = filename  # 作用：把随机临时文件名换成真实文件名；效果：答案里的出处才好看、可追溯。
        if not docs:  # 作用：一个字都没解析出来。
            raise HTTPException(status_code=422, detail="文件里没有解析出任何文字")
        chunks = splitter.split_documents(docs)  # 作用：切分。
        get_vectorstore().add_documents(chunks)  # 作用：增量入库；效果：向量化后追加，不覆盖已有数据。
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)  # 作用：删除临时文件；效果：missing_ok=True 表示文件不在也不报错。

    return IngestResponse(source=filename, chunks=len(chunks), message="入库成功")

if __name__ == "__main__":
    import uvicorn  # 作用：引入启动器；效果：直接运行本文件即启动。
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)  # 作用：监听所有网卡；效果：容器/虚拟机里也能被访问。