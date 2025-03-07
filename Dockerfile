FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    supervisor=4.2.5-1 \
    tzdata \
    libgl1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
ENV TZ=Asia/Shanghai

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt


# 创建目录和文件
# RUN mkdir -p /app/logs/supervisor && \
#     chmod -R 755 /app/logs/supervisor/ && \
#     chown -R root:root /app/logs/supervisor/