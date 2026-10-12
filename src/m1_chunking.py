from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field


_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_UNAVAILABLE = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  ⚠️  Bỏ qua PDF: chưa cài pypdf (chạy: pip install -r requirements.txt).")
        return ""

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    raw_sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+|\n\n', text) if sentence.strip()]
    sentences: list[str] = []
    pending_headers: list[str] = []
    for sentence in raw_sentences:
        if all(line.lstrip().startswith("#") for line in sentence.splitlines() if line.strip()):
            pending_headers.append(sentence)
            continue
        prefix = "\n\n".join(pending_headers)
        sentences.append(f"{prefix}\n\n{sentence}".strip() if prefix else sentence)
        pending_headers = []
    if pending_headers:
        if sentences:
            sentences[-1] = f"{sentences[-1]}\n\n{'\n\n'.join(pending_headers)}"
        else:
            sentences.append("\n\n".join(pending_headers))
    if not sentences:
        return []

    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_UNAVAILABLE
    try:
        if os.getenv("RAG_FORCE_FALLBACK") == "1":
            _SEMANTIC_MODEL_UNAVAILABLE = True
        if _SEMANTIC_MODEL is None and not _SEMANTIC_MODEL_UNAVAILABLE:
            from sentence_transformers import SentenceTransformer
            _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

        if _SEMANTIC_MODEL is None:
            raise RuntimeError("Semantic model is unavailable")

        from numpy import dot
        from numpy.linalg import norm

        embeddings = _SEMANTIC_MODEL.encode(sentences)
        similarities = [
            float(dot(embeddings[index - 1], embeddings[index]) /
                  (norm(embeddings[index - 1]) * norm(embeddings[index]) + 1e-9))
            for index in range(1, len(sentences))
        ]
    except Exception as exc:
        # The pipeline remains usable in an offline environment where the model
        # has not yet been downloaded. A lexical fallback keeps sentence boundaries.
        _SEMANTIC_MODEL_UNAVAILABLE = True
        print(f"  ⚠️  Semantic model unavailable; using lexical fallback: {exc}")

        def word_set(sentence: str) -> set[str]:
            return set(re.findall(r"\w+", sentence.lower()))

        similarities = []
        for index in range(1, len(sentences)):
            previous, current = word_set(sentences[index - 1]), word_set(sentences[index])
            similarities.append(len(previous & current) / (len(previous | current) or 1))

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "chunk_index": index, "strategy": "semantic"},
        )
        for index, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(_pack_paragraphs(text, parent_size)):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_index": parent_index, "chunk_type": "parent", "parent_id": parent_id},
            parent_id=parent_id,
        ))

        for child_index, child_text in enumerate(_split_to_limit(parent_text, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_index": len(children),
                    "child_index": child_index,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                },
                parent_id=parent_id,
            ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    header_pattern = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_lines: list[str] = []

    def flush_section() -> None:
        if current_header or any(line.strip() for line in current_lines):
            sections.append((current_header, current_lines.copy()))

    for line in text.splitlines():
        match = header_pattern.match(line)
        if match:
            flush_section()
            current_header = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    flush_section()

    return [
        Chunk(
            text="\n".join(lines).strip(),
            metadata={
                **metadata,
                "chunk_index": index,
                "section": header or "(preamble)",
                "strategy": "structure",
            },
        )
        for index, (header, lines) in enumerate(sections)
        if "\n".join(lines).strip()
    ]


def _split_to_limit(text: str, limit: int) -> list[str]:
    """Split text without losing characters, preferring paragraph/word boundaries."""
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        boundary = max(
            remaining.rfind("\n\n", 0, limit + 1),
            remaining.rfind("\n", 0, limit + 1),
            remaining.rfind(" ", 0, limit + 1),
        )
        # A single token may be larger than limit; splitting it is safer than looping.
        if boundary <= 0:
            boundary = limit
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _pack_paragraphs(text: str, limit: int) -> list[str]:
    """Pack whole paragraphs into chunks and split only oversized paragraphs."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    packed: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for piece in _split_to_limit(paragraph, limit):
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and len(candidate) > limit:
                packed.append(current)
                current = piece
            else:
                current = candidate
    if current:
        packed.append(current)
    return packed


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
