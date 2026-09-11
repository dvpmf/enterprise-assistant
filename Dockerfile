# 作用：指定基础镜像；效果：从官方 Python 3.12 精简版开始搭建环境。
FROM python:3.12-slim

# 作用：设置容器内的工作目录；效果：后面所有命令都在 /app 下执行。
WORKDIR /app

# 作用：先把依赖清单复制进镜像；效果：只要不改依赖，就能复用缓存、不用重装。
COPY requirements.txt .

# 作用：安装依赖；效果：镜像里具备项目所需的库。
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 作用：把当前目录的项目代码全部复制进镜像；效果：/app 下有全部代码和文档。
COPY . .

# 作用：声明容器提供 8000 端口；效果：起说明作用（真正的端口映射在 compose 里配）。
EXPOSE 8000

# 作用：容器启动时默认执行这条命令；效果：用 uvicorn 启动 FastAPI。
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]