from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: F401  (runs load_dotenv() so .env values are visible below)

# Day 24 config has no Groq block; read the Day 18 settings from the environment instead.
# Enrichment stays off unless GROQ_ENRICHMENT_ENABLED=true, so M5 uses its local fallback.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_ENRICHMENT_ENABLED = os.getenv("GROQ_ENRICHMENT_ENABLED", "false").lower() in {"1", "true", "yes"}


def create_llm_client():
    """Return a Groq client using the installed OpenAI-compatible SDK."""
    if not GROQ_API_KEY:
        return None

    import httpx
    from openai import OpenAI
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        timeout=10.0,
        max_retries=0,
        http_client=httpx.Client(timeout=10.0),
    )


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    response = _chat(
        "Tóm tắt đoạn văn sau trong tối đa 2 câu ngắn gọn bằng tiếng Việt. Chỉ trả về phần tóm tắt.",
        text,
        max_tokens=150,
    )
    return response or _fallback_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    response = _chat(
        f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên một dòng, không đánh số.",
        text,
        max_tokens=200,
    )
    if response:
        questions = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip() for line in response.splitlines()]
        questions = [question for question in questions if question]
        if questions:
            return questions[:n_questions]
    return [f"Đoạn văn quy định gì về {_fallback_topic(text)}?"][:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _chat(
        "Viết đúng một câu ngắn mô tả vị trí và chủ đề của đoạn văn trong tài liệu. Chỉ trả về câu đó.",
        f"Tài liệu: {document_title or 'không rõ'}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    ) or _fallback_context(document_title, text)
    return f"{context}\n\n{text}" if context else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    response = _chat(
        'Trích xuất metadata và chỉ trả về JSON hợp lệ: '
        '{"topic":"...","entities":["..."],"category":"policy|hr|it|finance|general","language":"vi|en"}.',
        text,
        max_tokens=150,
    )
    metadata = _parse_json_object(response) if response else None
    return metadata if metadata is not None else _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    response = _chat(
        """Phân tích đoạn văn và chỉ trả về một JSON hợp lệ, không Markdown:
{
  "summary": "tóm tắt tối đa 2 câu",
  "questions": ["3 câu hỏi mà đoạn văn trả lời"],
  "context": "một câu mô tả vị trí/chủ đề của đoạn văn",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|general", "language": "vi|en"}
}""",
        f"Tài liệu: {source or 'không rõ'}\n\nĐoạn văn:\n{text}",
        max_tokens=400,
    )
    result = _parse_json_object(response) if response else None
    if isinstance(result, dict):
        questions = result.get("questions", [])
        metadata = result.get("metadata", {})
        return {
            "summary": str(result.get("summary", "")).strip(),
            "questions": [str(question).strip() for question in questions if str(question).strip()][:3]
            if isinstance(questions, list) else [],
            "context": str(result.get("context", "")).strip(),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
    return {
        "summary": _fallback_summary(text),
        "questions": generate_hypothesis_questions(text),
        "context": _fallback_context(source, text),
        "metadata": _fallback_metadata(text),
    }


def _chat(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    """Make one Groq chat request, returning None on absent credentials or failure."""
    if not GROQ_API_KEY or not GROQ_ENRICHMENT_ENABLED:
        return None
    try:
        client = create_llm_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"reasoning_effort": "low"},
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  ⚠️  Groq enrichment failed: {exc}")
        return None


def _parse_json_object(response: str | None) -> dict | None:
    if not response:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _fallback_summary(text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if sentence.strip()]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _fallback_topic(text: str) -> str:
    words = re.findall(r"[\wÀ-ỹ]+", text)
    return " ".join(words[:8]).lower() or "nội dung đoạn văn"


def _fallback_context(source: str, text: str) -> str:
    source_label = f" trong tài liệu {source}" if source else ""
    return f"Đoạn trích{source_label} nói về {_fallback_topic(text)}."


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    if any(term in lowered for term in ("mật khẩu", "vpn", "bảo mật", "mfa")):
        category = "it"
    elif any(term in lowered for term in ("lương", "chi phí", "thưởng", "tạm ứng")):
        category = "finance"
    elif any(term in lowered for term in ("nhân viên", "nghỉ phép", "thử việc", "đào tạo")):
        category = "hr"
    else:
        category = "policy"
    return {"topic": _fallback_topic(text), "entities": [], "category": category, "language": "vi"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
