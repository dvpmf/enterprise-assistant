# -*- coding: utf-8 -*-
"""
作用：FastAPI 最小示例：搭建一个能响应 HTTP 请求的服务器。
效果：运行后，浏览器访问 http://127.0.0.1:8000/ 能看到问候语。
"""
from fastapi import FastAPI  # 作用：引入 FastAPI 框架；效果：能创建服务器应用。

# 作用：创建 FastAPI 应用实例；效果：app 就是你的"服务器对象"。
app = FastAPI()


# 作用：注册一个路由；效果：当有人访问根路径 "/" 时，执行下面的函数。
@app.get("/")
def read_root():
    """作用：处理 GET 请求的响应逻辑；效果：返回一段 JSON 数据给访问者。"""
    return {"message": "你好，我是企业助手 API"}

# 作用：程序入口；效果：直接运行本文件即可启动服务器，无需敲 uvicorn 命令。
if __name__ == "__main__":
    import uvicorn  # 作用：引入 uvicorn；效果：在代码里调用它启动。
    uvicorn.run("minimal_api:app", host="127.0.0.1", port=8000, reload=True)