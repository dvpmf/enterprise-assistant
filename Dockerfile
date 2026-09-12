# 作用：指定基础镜像；效果：钉死 Debian 12 版本，避免基础镜像更新导致"今天能构建、明天构建失败"。
FROM python:3.12-slim-bookworm

# 作用：设置容器内工作目录；效果：后面命令都在 /app 下执行。
WORKDIR /app

# 作用：把 Debian 官方源换成国内镜像；效果：构建时能正常下载系统包（官方源 deb.debian.org 在国内经常连不上，会报 Unable to connect）。
# 同时兼容两种源文件格式：新版镜像用 debian.sources（deb822 格式），老版用 sources.list。
RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
        if [ -f "$f" ]; then \
            sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' "$f"; \
        fi; \
    done

# 作用：安装 OCR 依赖的系统库；效果：opencv 需要 libGL.so.1 和 libglib，slim 镜像默认没装，不装会 ImportError。
# rm -rf /var/lib/apt/lists/* 作用：删掉 apt 缓存；效果：镜像体积小几十 MB。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 作用：先复制依赖清单；效果：只要不改依赖就能复用缓存层，重建快很多。
COPY requirements.txt .

# 作用：安装依赖；效果：走清华源，国内构建速度快。
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 作用：复制项目代码；效果：/app 下有全部代码与 documents。
COPY . .

# 作用：声明容器提供 8000 端口；效果：起说明作用（真正的端口映射在 compose 里配）。
EXPOSE 8000

# 作用：容器启动命令；效果：用 uvicorn 启动 FastAPI。
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]