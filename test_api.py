# -*- coding: utf-8 -*-
"""
作用：用 requests 向本地 FastAPI 接口发送 POST 请求，测试 RAG API。
效果：运行后打印服务器返回的回答和出处。
"""
import requests  # 作用：引入 HTTP 工具；效果：发送 POST 请求。

# 作用：定义接口地址；效果：指向本地 api.py 的 /ask。
API_URL = "http://127.0.0.1:8000/ask"


def ask(question: str):
    """作用：发送提问；效果：拿到服务器回答。"""
    # 作用：构造请求体；效果：符合 api.py 里 AskRequest 的要求（必须有 question）。
    payload = {"question": question}
    # 作用：发送 POST；效果：服务器执行检索+生成，返回结果。
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()  # 作用：检查状态码；效果：非 200 立刻报错。
    return resp.json()  # 作用：解析响应；效果：得到 {answer, sources}。


if __name__ == "__main__":
    # 作用：测一个问题；效果：验证 API 全链路是否通。
    result = ask("请事假要提前几天申请？")
    print("回答：", result["answer"])
    print("出处：", result["sources"])