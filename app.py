"""
中国招标法 RAG 合规审查助手
支持 Claude (Anthropic) / Qwen (DashScope) 双模型
"""

import os
import anthropic
import streamlit as st
from openai import OpenAI

from retriever import LawRetriever
from doc_processor import process_document, file_hash

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="招标法律顾问",
    page_icon="⚖️",
    layout="wide",
)

SYSTEM_PROMPT = """你是一位专业的中国招标投标合规审查顾问，深入熟悉《中华人民共和国招标投标法》（2017年修正版）及其实施条例。

**核心职责：对照法律条文审查标书的合规性，识别潜在违规风险。**

回答规则：
1. 若用户上传了标书，**主动对照【相关法律条款】逐项审查【标书原文片段】**，指出：
   - ❌ 违规或疑似违规条款（引用具体法条编号说明违规原因）
   - ⚠️ 存在法律风险或模糊地带的内容
   - ✅ 符合法律规定的关键条款（简要说明）
2. 审查结论需结构清晰，优先列出违规和风险项
3. 引用法条时注明具体编号，如"依据《招标投标法》第三条"或"依据实施条例第十七条"
4. 若检索内容不足以判断，注明"需结合完整标书进一步核查"
5. 语言专业简洁，适合提交法务或合规团队参考
6. 如问题与招标合规无关，礼貌说明本工具专注于招标法律合规审查"""

# ── 模型配置 ──────────────────────────────────────────────────────────────────
MODELS = {
    "Claude claude-opus-4-6":    "claude",
    "Qwen3.5-72B":       "qwen3.5-72b",
    "Qwen3.5-32B":       "qwen3.5-32b",
    "Qwen3-Max":         "qwen3-max-2026-01-23",
    "Qwen3-235B-A22B":   "qwen3-235b-a22b",
    "Qwen3-32B":         "qwen3-32b",
    "Qwen3-14B":         "qwen3-14b",
    "Qwen3-8B":          "qwen3-8b",
    "Qwen-Max":          "qwen-max",
    "Qwen-Plus":         "qwen-plus",
    "Qwen-Turbo":        "qwen-turbo",
}

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ── 初始化（缓存，只运行一次）──────────────────────────────────────────────
@st.cache_resource(show_spinner="加载检索模型，请稍候...")
def get_retriever() -> LawRetriever:
    return LawRetriever()


@st.cache_resource
def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    kwargs = {"api_key": api_key} if api_key else {}
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


@st.cache_resource
def get_qwen_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def stream_response(messages: list, model_key: str, max_tokens: int = 2048):
    """
    统一流式调用接口，根据 model_key 路由到 Claude 或 Qwen。
    逐字符 yield 文本片段。
    """
    if model_key == "claude":
        with get_anthropic_client().messages.stream(
            model="claude-opus-4-6",
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            thinking={"type": "adaptive"},
        ) as stream:
            for text in stream.text_stream:
                yield text
    else:
        # Qwen via DashScope (OpenAI-compatible)
        oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        stream = get_qwen_client().chat.completions.create(
            model=model_key,
            messages=oai_messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Session State 初始化 ──────────────────────────────────────────────────────
if "messages"       not in st.session_state:
    st.session_state.messages       = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []
if "indexed_hashes" not in st.session_state:
    st.session_state.indexed_hashes = set()
if "scan_target"    not in st.session_state:
    st.session_state.scan_target    = None

retriever = get_retriever()

# ── 布局 ─────────────────────────────────────────────────────────────────────
col_main, col_sidebar = st.columns([3, 1])

# ══════════════════════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════════════════════
with col_sidebar:

    # ── 模型选择 ──────────────────────────────────────────────────────────────
    st.subheader("🤖 选择模型")
    selected_label = st.selectbox(
        "LLM",
        list(MODELS.keys()),
        label_visibility="collapsed",
    )
    model_key = MODELS[selected_label]

    # 检查对应 API Key 是否配置
    if model_key == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.warning("未设置 ANTHROPIC_API_KEY")
    else:
        if not os.getenv("DASHSCOPE_API_KEY"):
            st.warning("未设置 DASHSCOPE_API_KEY")

    st.divider()

    # ── PDF 上传区 ────────────────────────────────────────────────────────────
    st.subheader("📂 上传标书文件")
    uploaded_file = st.file_uploader(
        "支持 PDF / Word 格式",
        type=["pdf", "doc", "docx"],
        help="上传后按页提取文本（相邻两页合并为一个检索块），与法律条文联合检索",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        fhash      = file_hash(file_bytes)

        if fhash in st.session_state.indexed_hashes or retriever.doc_exists(fhash):
            st.success(f"✅ 已索引：{uploaded_file.name}")
            st.session_state.indexed_hashes.add(fhash)
        else:
            with st.spinner(f"正在解析并索引「{uploaded_file.name}」..."):
                try:
                    chunks = process_document(file_bytes, uploaded_file.name)
                    if not chunks:
                        st.warning("未能从文件中提取文本，请检查文件内容")
                    else:
                        retriever.index_doc(chunks)
                        st.session_state.indexed_hashes.add(fhash)
                        st.success(f"✅ 索引完成：{uploaded_file.name}（{len(chunks)} 个片段）")
                except Exception as e:
                    st.error(f"解析失败：{e}")

    # 已上传文件列表
    doc_names = retriever.list_docs()
    if doc_names:
        st.caption(f"已索引标书（{len(doc_names)} 份）：")
        for name in doc_names:
            col_name, col_scan, col_del = st.columns([3, 1, 1])
            col_name.caption(f"📄 {name}")
            if col_scan.button("审查", key=f"scan_{name}", help=f"逐页审查 {name}"):
                st.session_state.scan_target = name
                st.rerun()
            if col_del.button("✕", key=f"del_{name}", help=f"删除 {name}"):
                retriever.remove_doc(name)
                if st.session_state.scan_target == name:
                    st.session_state.scan_target = None
                st.rerun()

    st.divider()

    # ── 检索结果展示 ──────────────────────────────────────────────────────────
    st.subheader("📋 检索到的内容")
    if st.session_state.last_retrieved:
        law_results = [r for r in st.session_state.last_retrieved if r["source"] == "law"]
        doc_results = [r for r in st.session_state.last_retrieved if r["source"] == "doc"]

        if law_results:
            st.caption("⚖️ 法律条款")
            for i, r in enumerate(law_results, 1):
                label = f"{r['id']} ({r['score']})"
                with st.expander(label, expanded=(i == 1)):
                    src = f"{r['title']} · " if r.get("title") else ""
                    st.caption(f"{src}{r['chapter']}")
                    st.text(r["text"])

        if doc_results:
            st.caption("📄 标书原文")
            for r in doc_results:
                with st.expander(f"{r['filename']} 第{r['page']}页 ({r['score']})"):
                    st.text(r["text"])
    else:
        st.info("提问后，这里将展示检索到的相关内容")

    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.last_retrieved = []
        st.rerun()

    st.caption("本工具仅供参考，不构成正式法律意见")

# ══════════════════════════════════════════════════════════════════════════════
# 主聊天区
# ══════════════════════════════════════════════════════════════════════════════
with col_main:
    st.title("⚖️ 招标文件合规审查")
    if doc_names:
        doc_hint = f"已加载 {len(doc_names)} 份标书，可直接发送审查指令"
        st.caption(f"📋 {doc_hint} · 基于《招标投标法》及实施条例 · {selected_label}")
    else:
        st.caption(f"👈 请先在左侧上传标书文件，再进行合规审查 · {selected_label}")

    # ── 逐页扫描模式 ──────────────────────────────────────────────────────────
    if st.session_state.scan_target:
        scan_name = st.session_state.scan_target
        st.info(f"📑 正在逐页审查：**{scan_name}**（使用 {selected_label}）")

        pages = retriever.get_doc_pages(scan_name)
        if not pages:
            st.warning("未找到该文件的内容，请重新上传。")
            st.session_state.scan_target = None
        else:
            if st.button("▶ 开始逐页合规审查", type="primary"):
                PAGE_REVIEW_PROMPT = (
                    "请对以下标书片段进行合规审查，"
                    "对照《招标投标法》及实施条例，"
                    "指出❌违规条款、⚠️风险点，以及✅合规内容，引用具体法条编号。"
                    "若该片段无实质内容可审查，请简要说明。"
                )

                full_report = []
                progress = st.progress(0, text="准备审查...")
                report_area = st.empty()

                for idx, chunk in enumerate(pages):
                    progress.progress(
                        (idx + 1) / len(pages),
                        text=f"审查第 {chunk['page']} 页（{idx+1}/{len(pages)}）..."
                    )

                    law_results = retriever.retrieve_law(chunk["text"], top_k=4)
                    law_context = retriever.format_context(law_results)

                    page_prompt = (
                        f"【相关法律条款】\n{law_context}\n\n"
                        f"【标书第 {chunk['page']} 页原文】\n{chunk['text']}\n\n"
                        f"{PAGE_REVIEW_PROMPT}"
                    )

                    page_result = f"\n\n---\n### 第 {chunk['page']} 页\n"
                    for text in stream_response(
                        [{"role": "user", "content": page_prompt}],
                        model_key, max_tokens=1024
                    ):
                        page_result += text

                    full_report.append(page_result)
                    report_area.markdown("".join(full_report))

                progress.empty()
                st.success(f"✅ 审查完成，共 {len(pages)} 页")

                report_text = f"**【{scan_name} 逐页合规审查报告】**\n" + "".join(full_report)
                st.session_state.messages.append({"role": "assistant", "content": report_text})
                st.session_state.scan_target = None

            if st.button("✕ 取消审查"):
                st.session_state.scan_target = None
                st.rerun()

        st.divider()

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    if user_input := st.chat_input("输入审查指令，如：请审查本标书的合规性 / 检查资质要求是否违规..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 检索
        with st.spinner("检索相关内容..."):
            retrieved = retriever.retrieve(user_input, top_k=5)
            st.session_state.last_retrieved = retrieved
            context = retriever.format_context(retrieved)

        # 构建带上下文的消息
        user_with_context = (
            f"【检索到的参考内容】\n{context}\n\n"
            f"【用户问题】\n{user_input}"
        )

        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        api_messages.append({"role": "user", "content": user_with_context})

        # 流式调用
        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""

            for text in stream_response(api_messages, model_key):
                full_response += text
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
