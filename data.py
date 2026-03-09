"""
招标法知识库数据加载器
从 data/ 目录下的 JSONL 文件加载《招标投标法》和《招标投标法实施条例》
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict

# data/ 目录路径（相对于本文件）
DATA_DIR = Path(__file__).parent / "data"

# 要加载的文件（优先使用原始版，而非标记版）
JSONL_FILES = [
    "中华人民共和国招标投标法.jsonl",
    "中华人民共和国招标投标法实施条例.jsonl",
]

# 提取条文编号的正则：匹配 "第X条" 开头（支持中文数字和阿拉伯数字）
_ARTICLE_ID_RE = re.compile(r"^(第[零一二三四五六七八九十百\d]+条)")


def _extract_article_id(content: str) -> str:
    """从条文内容中提取条文编号，如 '第三条'"""
    m = _ARTICLE_ID_RE.match(content.strip())
    return m.group(1) if m else ""


def load_articles(data_dir: Path = DATA_DIR) -> List[Dict[str, str]]:
    """
    从 JSONL 文件加载所有条文，返回标准化列表：
    [{"id": "第X条", "title": "...", "chapter": "...", "text": "..."}]
    """
    articles = []
    for filename in JSONL_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"数据文件不存在: {filepath}\n"
                f"请确保 data/ 目录已挂载或复制到容器中"
            )
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 标记版文件每行末尾有 &&&&，去除后再解析
                line = line.removesuffix("&&&&").strip()
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = obj.get("content", "").strip()
                if not content:
                    continue
                articles.append({
                    "id": _extract_article_id(content),
                    "title": obj.get("title", ""),
                    "chapter": obj.get("chapter", ""),
                    "text": content,
                })
    return articles


def get_all_articles() -> List[Dict[str, str]]:
    """返回所有条文（对外统一接口）"""
    return load_articles()


def get_article_count() -> int:
    """返回条文总数"""
    return len(load_articles())


if __name__ == "__main__":
    articles = load_articles()
    print(f"共加载 {len(articles)} 条条文")
    for source in set(a["title"] for a in articles):
        count = sum(1 for a in articles if a["title"] == source)
        print(f"  {source}: {count} 条")
    print("\n示例（前2条）：")
    for a in articles[:2]:
        print(f"  [{a['title']}·{a['chapter']}·{a['id']}]")
        print(f"  {a['text'][:60]}...")
