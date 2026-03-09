"""
PDF 处理模块：提取文本、分块，供向量索引使用
"""

import io
import hashlib
from typing import List, Dict

import pdfplumber

CHUNK_SIZE = 500     # 每块字符数
CHUNK_OVERLAP = 80   # 相邻块重叠字符数（保持语义连贯）


def extract_pages(file_bytes: bytes) -> List[Dict]:
    """逐页提取 PDF 文本，返回 [{page, text}, ...]"""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def _split_chunks(text: str) -> List[str]:
    """将长文本切成带重叠的固定大小块"""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def process_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    处理 PDF 文件，返回分块列表：
    [{"filename": str, "file_hash": str, "page": int, "chunk_idx": int, "text": str}]
    """
    file_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    pages = extract_pages(file_bytes)

    chunks, chunk_idx = [], 0
    for page_info in pages:
        for chunk_text in _split_chunks(page_info["text"]):
            chunks.append({
                "filename": filename,
                "file_hash": file_hash,
                "page": page_info["page"],
                "chunk_idx": chunk_idx,
                "text": chunk_text,
            })
            chunk_idx += 1

    return chunks


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:12]
