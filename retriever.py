"""
向量检索模块：使用 BAAI/bge-small-zh-v1.5 + Qdrant 进行语义检索
支持两个 collection：
  - zhaobiao_law  : 法律条文（静态，由 indexer.py 写入）
  - zhaobiao_docs : 上传的标书文件（动态，由 app.py 写入）
"""

import os
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

LAW_COLLECTION  = "zhaobiao_law"
DOC_COLLECTION  = "zhaobiao_docs"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM   = 512


class LawRetriever:
    def __init__(self, qdrant_host: str = None, qdrant_port: int = 6333):
        host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        self.client = QdrantClient(host=host, port=qdrant_port)
        self.model  = SentenceTransformer(EMBEDDING_MODEL)
        self._ensure_doc_collection()

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _ensure_doc_collection(self):
        """确保标书 collection 存在"""
        names = [c.name for c in self.client.get_collections().collections]
        if DOC_COLLECTION not in names:
            self.client.create_collection(
                collection_name=DOC_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def embed(self, text: str) -> List[float]:
        """将文本转为向量（bge 模型建议加查询前缀提升检索质量）"""
        prefixed = f"为这个句子生成表示以用于检索相关文章：{text}"
        vector = self.model.encode(prefixed, normalize_embeddings=True)
        return vector.tolist()

    def _query(self, collection: str, query_vector: List[float], top_k: int,
               filter_: Optional[Filter] = None) -> list:
        result = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=filter_,
        )
        return result.points

    # ── 法律条文检索 ──────────────────────────────────────────────────────────

    def retrieve_law(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索法律条文"""
        vec = self.embed(query)
        points = self._query(LAW_COLLECTION, vec, top_k)
        return [
            {
                "source":  "law",
                "id":      p.payload.get("id", ""),
                "title":   p.payload.get("title", ""),
                "chapter": p.payload.get("chapter", ""),
                "text":    p.payload.get("text", ""),
                "score":   round(p.score, 4),
            }
            for p in points
        ]

    # ── 标书文件索引与检索 ────────────────────────────────────────────────────

    def doc_exists(self, file_hash: str) -> bool:
        """判断该 hash 对应文件是否已索引"""
        result = self.client.scroll(
            collection_name=DOC_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
            ]),
            limit=1,
            with_payload=False,
        )
        return len(result[0]) > 0

    def index_doc(self, chunks: List[Dict[str, Any]]):
        """将标书分块写入 DOC_COLLECTION（chunks 来自 pdf_processor.process_pdf）"""
        if not chunks:
            return
        texts   = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)

        existing_count = self.client.count(collection_name=DOC_COLLECTION).count
        points = [
            PointStruct(
                id=existing_count + i,
                vector=vec.tolist(),
                payload={
                    "source":    "doc",
                    "filename":  chunk["filename"],
                    "file_hash": chunk["file_hash"],
                    "page":      chunk["page"],
                    "chunk_idx": chunk["chunk_idx"],
                    "text":      chunk["text"],
                },
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=DOC_COLLECTION, points=points)

    def retrieve_doc(self, query: str, top_k: int = 3,
                     filename: str = None) -> List[Dict[str, Any]]:
        """检索已上传标书，可按文件名过滤"""
        vec = self.embed(query)
        filter_ = None
        if filename:
            filter_ = Filter(must=[
                FieldCondition(key="filename", match=MatchValue(value=filename))
            ])
        points = self._query(DOC_COLLECTION, vec, top_k, filter_)
        return [
            {
                "source":   "doc",
                "filename": p.payload.get("filename", ""),
                "page":     p.payload.get("page", 0),
                "text":     p.payload.get("text", ""),
                "score":    round(p.score, 4),
            }
            for p in points
        ]

    def list_docs(self) -> List[str]:
        """返回已索引的文件名列表（去重）"""
        result = self.client.scroll(
            collection_name=DOC_COLLECTION,
            limit=1000,
            with_payload=["filename"],
        )
        seen, names = set(), []
        for p in result[0]:
            fn = p.payload.get("filename", "")
            if fn and fn not in seen:
                seen.add(fn)
                names.append(fn)
        return names

    def remove_doc(self, filename: str):
        """删除某文件的所有向量"""
        self.client.delete(
            collection_name=DOC_COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="filename", match=MatchValue(value=filename))
            ]),
        )

    # ── 联合检索 ──────────────────────────────────────────────────────────────

    def retrieve_all(self, query: str, law_k: int = 4,
                     doc_k: int = 3) -> List[Dict[str, Any]]:
        """同时检索法律条文和已上传标书，合并结果"""
        results = self.retrieve_law(query, top_k=law_k)
        doc_count = self.client.count(collection_name=DOC_COLLECTION).count
        if doc_count > 0:
            results += self.retrieve_doc(query, top_k=doc_k)
        return results

    # ── 格式化 ────────────────────────────────────────────────────────────────

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """将检索结果格式化为 prompt 上下文"""
        if not results:
            return "（未检索到相关内容）"
        law_parts, doc_parts = [], []
        for r in results:
            if r["source"] == "law":
                src = f"{r['title']}·" if r["title"] else ""
                law_parts.append(
                    f"【{src}{r['chapter']}·{r['id']}】（相关度 {r['score']}）\n{r['text']}"
                )
            else:
                doc_parts.append(
                    f"【标书：{r['filename']} 第{r['page']}页】（相关度 {r['score']}）\n{r['text']}"
                )
        sections = []
        if law_parts:
            sections.append("## 相关法律条款\n" + "\n\n".join(law_parts))
        if doc_parts:
            sections.append("## 标书原文片段\n" + "\n\n".join(doc_parts))
        return "\n\n".join(sections)

    def get_doc_pages(self, filename: str) -> List[Dict[str, Any]]:
        """按页码顺序返回指定文件的所有分块（用于逐页审查）"""
        result = self.client.scroll(
            collection_name=DOC_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="filename", match=MatchValue(value=filename))
            ]),
            limit=2000,
            with_payload=True,
        )
        points = result[0]
        chunks = [
            {
                "page":      p.payload.get("page", 0),
                "chunk_idx": p.payload.get("chunk_idx", 0),
                "text":      p.payload.get("text", ""),
            }
            for p in points
        ]
        # 按 chunk_idx 排序（即原始页面顺序）
        chunks.sort(key=lambda c: c["chunk_idx"])
        return chunks

    # ── 兼容旧接口 ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve_all(query, law_k=top_k, doc_k=3)
