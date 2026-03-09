# 招标文件合规审查工具

基于 RAG（检索增强生成）的中国招标投标法合规审查助手，支持上传标书文件，逐页对照《招标投标法》及实施条例进行合规分析。

## 功能特性

- **法律知识库**：内置《招标投标法》（2017年修正版）及实施条例，共 152 条，向量索引
- **标书上传**：支持 PDF / Word（.doc/.docx）格式，按页滑动窗口分块
- **逐页合规审查**：对每页标书内容单独检索相关法条，输出 ❌违规 / ⚠️风险 / ✅合规 三类结论
- **多模型支持**：Claude claude-opus-4-6（Anthropic）/ Qwen-Plus / Qwen-Max / Qwen-Turbo（阿里云 DashScope）
- **流式输出**：实时显示审查过程
- **Docker 部署**：一键启动 app + Qdrant 向量数据库

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API Key
```

```env
ANTHROPIC_API_KEY=sk-ant-...        # Claude 模型必填
DASHSCOPE_API_KEY=sk-...            # Qwen 模型必填
ANTHROPIC_BASE_URL=https://...      # 可选，代理地址
```

### 2. Docker 启动（推荐）

```bash
docker compose up --build
```

浏览器访问 [http://localhost:8501](http://localhost:8501)

### 3. 本地开发

```bash
# 启动 Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 安装依赖
pip install -r requirements.txt

# 索引法律条文
python indexer.py

# 启动应用
streamlit run app.py
```

## 使用方式

1. 在左侧侧边栏选择 LLM 模型
2. 上传标书文件（PDF 或 Word）
3. 点击文件旁的「审查」按钮，启动逐页合规扫描
4. 或在聊天框输入具体问题进行针对性查询

## 技术架构

| 组件 | 技术 |
|------|------|
| Embedding | BAAI/bge-small-zh-v1.5（512维，中文优化） |
| 向量数据库 | Qdrant |
| LLM | Claude claude-opus-4-6 / Qwen（DashScope） |
| UI | Streamlit |
| 文档解析 | pdfplumber + python-docx |
| 部署 | Docker Compose |

## 项目结构

```
├── app.py              # Streamlit 主应用
├── retriever.py        # 向量检索模块
├── doc_processor.py    # 文档解析（PDF/Word）
├── indexer.py          # 法律条文索引
├── data.py             # 法律数据加载
├── data/               # 招标投标法 JSONL 数据
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 免责声明

本工具仅供参考，不构成正式法律意见。实际法律事务请咨询专业律师。
