"""
文档处理模块：支持 PDF / DOC / DOCX
分块策略：按页提取后，以滑动窗口（当前页 + 上一页）合并，保留跨页上下文
"""

import io
import hashlib
from pathlib import Path
from typing import List, Dict

import pdfplumber
from docx import Document as DocxDocument

# Word 文档"虚拟页"大小（按字符数划分，模拟页面）
WORD_PAGE_CHARS = 1000


# ── 公共工具 ──────────────────────────────────────────────────────────────────

def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:12]


def _sliding_window(pages: List[Dict], filename: str, fhash: str) -> List[Dict]:
    """
    滑动窗口分块：
      chunk[0] = page[0]
      chunk[i] = page[i-1] + page[i]   (i >= 1)
    每个 chunk 记录的 page 号为当前页（较后的那页）
    """
    chunks = []
    for i, cur in enumerate(pages):
        if i == 0:
            text = cur["text"]
        else:
            text = pages[i - 1]["text"] + "\n" + cur["text"]
        text = text.strip()
        if text:
            chunks.append({
                "filename":  filename,
                "file_hash": fhash,
                "page":      cur["page"],
                "chunk_idx": i,
                "text":      text,
            })
    return chunks


# ── PDF ───────────────────────────────────────────────────────────────────────

def _extract_pdf_pages(file_bytes: bytes) -> List[Dict]:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def process_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    fhash = file_hash(file_bytes)
    pages = _extract_pdf_pages(file_bytes)
    return _sliding_window(pages, filename, fhash)


# ── Word (DOC / DOCX) ─────────────────────────────────────────────────────────

def _extract_docx_pages(file_bytes: bytes) -> List[Dict]:
    """
    docx 没有原生页面边界，按 WORD_PAGE_CHARS 字符数切分"虚拟页"
    遇到显式分页符时强制换页
    """
    doc = DocxDocument(io.BytesIO(file_bytes))

    pages, page_num, buf = [], 1, []
    for para in doc.paragraphs:
        # 检测显式分页符
        has_break = any(
            run.text == "" and "<w:br" in run._r.xml and 'w:type="page"' in run._r.xml
            for run in para.runs
            if run._r is not None
        )
        text = para.text.strip()
        if text:
            buf.append(text)

        if has_break or sum(len(t) for t in buf) >= WORD_PAGE_CHARS:
            combined = "\n".join(buf).strip()
            if combined:
                pages.append({"page": page_num, "text": combined})
            page_num += 1
            buf = []

    # 剩余内容
    if buf:
        combined = "\n".join(buf).strip()
        if combined:
            pages.append({"page": page_num, "text": combined})

    return pages


def process_docx(file_bytes: bytes, filename: str) -> List[Dict]:
    fhash = file_hash(file_bytes)
    pages = _extract_docx_pages(file_bytes)
    return _sliding_window(pages, filename, fhash)


# ── 统一入口 ──────────────────────────────────────────────────────────────────

def process_document(file_bytes: bytes, filename: str) -> List[Dict]:
    """根据扩展名自动选择处理方式"""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return process_pdf(file_bytes, filename)
    elif ext in (".docx", ".doc"):
        return process_docx(file_bytes, filename)
    else:
        raise ValueError(f"不支持的文件类型：{ext}，仅支持 PDF / DOC / DOCX")
