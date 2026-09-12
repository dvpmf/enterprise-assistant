# -*- coding: utf-8 -*-
"""
作用：用 Streamlit 搭建企业助手界面 —— 聊天问答 + 显示出处 + 切换 Agent 模式 + 上传文件入库。
效果：浏览器里既能像聊天一样提问（带出处），能切换"固定流程 / 模型自主"两种问答模式，
      还能上传文档扩充知识库。
"""
import os  # 作用：读取环境变量；效果：让后端地址可在不同环境切换。

import requests  # 作用：引入 HTTP 工具；效果：界面把请求发给 FastAPI。
import streamlit as st  # 作用：引入界面工具；效果：用 Python 写网页。

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")  # 作用：后端基地址；效果：三个接口都从它拼出来，本机跑用默认值、容器跑读注入值。
TIMEOUT = 120  # 作用：请求超时秒数；效果：Agent 要多步调工具、比 /ask 慢，给足时间。

st.set_page_config(page_title="企业智能助手")  # 作用：设置浏览器标签页标题。

# 作用：初始化会话状态；效果：跨重跑保存数据（Streamlit 每次交互都会重跑整个脚本）。
if "messages" not in st.session_state:
    st.session_state.messages = []  # 作用：聊天记录；效果：元素形如 {"role", "content", "sources", "tools_used"}。

st.title("企业智能助手")  # 作用：页面大标题。


def render_extras(sources: list | None, tools_used: list | None) -> None:
    """作用：渲染"出处"与"工具调用"信息；效果：历史消息和当轮消息共用同一段渲染逻辑。

    参数：
        sources: /ask 返回的出处列表，每条含 source / page / type / score
        tools_used: /agent 返回的工具名列表
    """
    if tools_used:  # 作用：Agent 模式才有；效果：让人一眼看到模型这次调了哪些工具。
        st.caption("调用的工具：" + " → ".join(tools_used))
    if sources:  # 作用：RAG 模式才有；效果：把答案的依据摆出来。
        with st.expander(f"出处（{len(sources)} 条）"):  # 作用：折叠面板；效果：默认收起，不抢占答案的位置。
            for i, src in enumerate(sources, start=1):
                st.markdown(f"**{i}. {src.get('source')}**")  # 作用：文件名加粗；效果：最显眼的证据。
                score = src.get("score")
                st.caption(
                    f"页码：{src.get('page') or '-'} ｜ "
                    f"来源类型：{src.get('type') or '-'} ｜ "
                    f"精排得分：{score if score is not None else '-'}"
                )  # 作用：把页码、解析路径、精排得分放在一行；效果：能看出"这份资料是怎么被找到的"。


def call_ask(question: str) -> dict:
    """作用：调用 /ask 接口（固定流程：检索 → 精排 → 生成）；效果：返回答案与出处。

    参数：
        question: 用户问题
    返回：
        dict: 形如 {"answer": str, "sources": list}
    """
    resp = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=TIMEOUT)  # 作用：POST 问题给后端。
    resp.raise_for_status()  # 作用：检查状态码；效果：失败立刻抛错，交给上层显示。
    return resp.json()


def call_agent(question: str, thread_id: str = "streamlit") -> dict:
    """作用：调用 /agent 接口（模型自主决定调哪些工具）；效果：返回答案、用到的工具、消息条数。

    参数：
        question: 用户问题
        thread_id: 会话标识；效果：同一个 id 的多轮对话共享上下文
    返回：
        dict: 形如 {"answer": str, "tools_used": list, "messages": int}
    """
    resp = requests.post(
        f"{API_BASE}/agent",
        json={"question": question, "thread_id": thread_id},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def call_ingest(uploaded_file) -> dict:
    """作用：把上传的文件送到 /ingest 入库；效果：返回文件名与入库碎片数。

    参数：
        uploaded_file: Streamlit 的 UploadedFile 对象
    返回：
        dict: 形如 {"source": str, "chunks": int, "message": str}
    异常：
        RuntimeError: 后端返回非 200 时抛出，消息里带后端的 detail
    """
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}  # 作用：组装 multipart 表单；效果：文件名一起传过去，后端用它做出处。
    resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=TIMEOUT)
    if resp.status_code != 200:  # 作用：把后端的错误说明取出来；效果：界面显示"不支持的文件类型"这种具体原因，而不是一串异常栈。
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        raise RuntimeError(detail or f"HTTP {resp.status_code}")
    return resp.json()


# ---- 侧边栏：模式切换 + 上传入库 ----
with st.sidebar:
    st.header("设置")

    mode = st.radio(  # 作用：选择问答模式；效果：让 /ask 与 /agent 的区别在界面上直接可见。
        "问答模式",
        ("知识库问答（固定流程）", "Agent 工具调用（模型自主）"),
    )
    use_agent = mode.startswith("Agent")  # 作用：转成布尔值；效果：下面只判断一次。

    if use_agent:
        st.caption("模型自己决定要不要查资料、要不要算金额，可能多步完成，比 /ask 慢。")
    else:
        st.caption("流程写死：先检索、再精排、最后生成。稳定、更快。")

    st.divider()

    st.subheader("上传文档入库")  # 作用：扩充知识库的入口；效果：让"多格式摄入"能当场演示。
    uploaded = st.file_uploader(
        "支持 txt / md / pdf / 图片（图片与扫描件走 OCR）",
        type=["txt", "md", "pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
    if uploaded is not None and st.button("上传并入库"):  # 作用：必须点按钮才上传；效果：避免界面每次重跑都重复入库。
        with st.spinner("正在解析并向量化，图片会慢一些..."):
            try:
                result = call_ingest(uploaded)
                st.success(f"入库成功：{result['source']}，切出 {result['chunks']} 个碎片")
            except Exception as exc:
                st.error(f"入库失败：{exc}")

    st.divider()

    if st.button("清空对话"):  # 作用：演示时重置；效果：不用重启服务就能从头开始。
        st.session_state.messages = []
        st.rerun()  # 作用：立刻重跑脚本；效果：清空后马上看到干净界面。


# ---- 主区域：先渲染历史对话 ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # 作用：按角色显示气泡；效果：user 与 assistant 视觉区分。
        st.write(msg["content"])
        render_extras(msg.get("sources"), msg.get("tools_used"))

# ---- 底部输入框 ----
prompt = st.chat_input("请输入你的问题，如：出差住宿能报多少？")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})  # 作用：先把用户问题记进历史。
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        sources: list = []  # 作用：先给默认值；效果：无论走哪条分支，下面渲染都不会报未定义。
        tools_used: list = []
        with st.spinner("正在调用工具..." if use_agent else "正在检索企业知识库..."):
            try:
                if use_agent:
                    data = call_agent(prompt)
                    answer = data["answer"]
                    tools_used = data.get("tools_used", [])
                else:
                    data = call_ask(prompt)
                    answer = data["answer"]
                    sources = data.get("sources", [])
            except Exception as exc:  # 作用：兜住后端不可用等情况；效果：界面上给出原因，而不是白屏报错。
                answer = f"调用后端失败：{exc}"

        st.write(answer)  # 作用：显示答案。
        render_extras(sources, tools_used)  # 作用：显示出处或工具调用。

    st.session_state.messages.append(  # 作用：整条记进历史；效果：出处和工具信息在重跑后也能保留。
        {"role": "assistant", "content": answer, "sources": sources, "tools_used": tools_used}
    )
