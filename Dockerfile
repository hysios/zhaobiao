FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖（sentence-transformers 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .

# 安装 PyTorch CPU-only 版本（减小镜像体积），再安装其他依赖
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 预下载 embedding 模型（baked into image，避免运行时下载）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 复制应用代码和数据
COPY data.py indexer.py retriever.py app.py doc_processor.py ./
COPY data/ ./data/

# Streamlit 配置
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# 启动：先索引，再运行 Streamlit
CMD ["sh", "-c", "python indexer.py && streamlit run app.py"]
