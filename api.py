# -*- coding: utf-8 -*-
"""
作用：把多格式 RAG 问答能力包装成 HTTP API。
效果：
    POST /ask    提问 → 召回 top10 → 精排 top3 → DeepSeek 生成答案 + 标注出处
    POST /ingest 上传文件（txt/md/pdf/图片）→ 解析（含 OCR）→ 切分 → 百炼向量化 → 入库
    GET  /health 健康检查（顺便报告库里有多少条向量）
"""
import tempfile  # 作用：创建临时文件；效果：上传的内容先落临时文件，再交给 loaders 处理。
from functools import lru_cache  # 作用：缓存；效果：向量库连接全进程只建一次。
from pathlib import Path  # 作用：路径处理；效果：取扩展名、删临时文件。

from fastapi import FastAPI, File, HTTPException, UploadFile  # 作用：FastAPI 组件；效果：路由、文件上传、错误响应。
from langchain_chroma import Chroma  # 作用：向量库连接器。
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 作用：切分器。
from pydantic import BaseModel  # 作用：请求/响应体结构；效果：自动校验 + 生成 /docs 文档。

from loaders import load_file  # 作用：多格式解析（含 OCR）。
from providers import get_chat_model, get_embedding  # 作用：模型层工厂。
from rerank import rerank_documents  # 作用：精排层。

# ---- 配置区 ----
PERSIST_DIR = "chroma_db"  # 作用：向量库目录。
RECALL_K = 10  # 作用：粗筛召回条数；效果：宁多勿漏，交给精排去筛。
RERANK_TOP_N = 3  # 作用：精排后保留条数；效果：只把最相关的 3 条塞进提示词。
MAX_UPLOAD_MB = 20  # 作用：上传大小上限（MB）。
ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # 作用：白名单；效果：拒绝乱七八糟的文件类型。

SYSTEM_PROMPT = (
    "你是[企业智能助手]，负责依据提供的企业内部资料回答员工问题。\n"
    "规则：1. 只依据资料回答，资料没有的内容，明确说'资料中未找到相关信息'；\n"
    "2. 回答简洁专业，先给结论再给依据；\n"
    "3. 回答末尾注明资料出处（来源文件名）。"
)  # 作用：人设与约束；效果：让模型贴着资料回答，不自由发挥。

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)  # 作用：切分器实例；效果：入库碎片与建库时保持同样粒度。
llm = get_chat_model()  # 作用：模块级只建一次；效果：主备链全局复用，不每个请求重建连接池。

@lru_cache(maxsize=None)
def get_vectorstore() -> Chroma:
    """作用：单例向量库；效果：整个进程只建立一次连接，避免每个请求重复连库。

    返回：
        Chroma: 已连接的向量库对象
    """
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding())  # 作用：用百炼 embedding 连接本地库；效果：查询与入库同一向量空间。

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

app = FastAPI(title="企业智能助手 API")  # 作用：创建应用；效果：所有路由挂在它身上。

def retrieve(query: str, k: int = RECALL_K) -> list:
    """作用：向量粗筛；效果：返回与问题语义最近的 k 条候选碎片。

    参数：
        query: 用户问题
        k: 召回条数
    返回：
        list[Document]: 候选碎片
    """
    return get_vectorstore().similarity_search(query, k=k)  # 作用：相似度检索。

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