# -*- coding: utf-8 -*-
"""
作用：用 Streamlit 搭建企业助手聊天界面，通过 FastAPI /ask 接口获取回答。
效果：浏览器中像 ChatGPT 一样与企业助手对话。
"""
import streamlit as st  # 作用：引入界面工具；效果：用 Python 写网页。
import requests  # 作用：引入 HTTP 工具；效果：界面把问题发给 FastAPI。

# 作用：定义后端接口地址；效果：界面请求指向 api.py。
API_URL = "http://127.0.0.1:8000/ask"

# 作用：设置页面标题；效果：浏览器标签页显示"企业智能助手"。
st.set_page_config(page_title="企业智能助手", page_icon="")

# 作用：初始化会话历史；效果：若 st.session_state.messages 不存在则建空列表，避免每次重跑丢失记录。
if "messages" not in st.session_state:
    st.session_state.messages = []  # 作用：存放聊天记录；效果：[{"role":"user/assistant","content":"..."}]

# 作用：页面顶部标题；效果：界面显示大标题。
st.title("企业智能助手")

# 作用：遍历历史记录并逐个显示；效果：每次重跑把之前所有对话画出来（所以不丢）。
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # 作用：按角色显示气泡；效果：user 靠右、assistant 靠左。
        st.write(msg["content"])  # 作用：显示该条消息文字；效果：对话内容展示在气泡里。

# 作用：底部输入框；效果：用户打字回车即触发一次提问。
prompt = st.chat_input("请输入你的问题，如：请事假要提前几天？")


def ask_api(question: str) -> str:
    """作用：调用 FastAPI /ask 接口；效果：返回模型基于文档的回答。"""
    resp = requests.post(API_URL, json={"question": question})  # 作用：POST 问题给后端；效果：执行 RAG 检索+生成。
    resp.raise_for_status()  # 作用：检查状态码；效果：失败立刻报错。
    return resp.json()["answer"]  # 作用：提取回答；效果：只取 answer 字段（出处可选展示）。


# 作用：判断用户是否输入了内容；效果：有输入才执行下面的问答流程。
if prompt:
    # 作用：把用户问题记入历史；效果：重跑后还能看到这句。
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 作用：立即显示用户气泡；效果：不用等模型返回就能看到自己说了啥。
    with st.chat_message("user"):
        st.write(prompt)

    # 作用：调用后端接口获取回答；效果：走完整 RAG 链路。
    with st.spinner("正在检索企业知识库..."):  # 作用：显示等待动画；效果：提示用户系统在干活。
        answer = ask_api(prompt)

    # 作用：把回答记入历史；效果：下次重跑能显示。
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # 作用：显示 AI 回答气泡；效果：答案展示在界面上。
    with st.chat_message("assistant"):
        st.write(answer)