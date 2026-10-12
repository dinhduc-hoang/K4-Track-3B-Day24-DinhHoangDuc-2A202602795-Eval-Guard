from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH, ARTIFACTS_DIR
from src.experiment_support import write_json

Distribution = str  # "factual" | "multi_hop" | "adversarial"
DISTRIBUTIONS = ("factual", "multi_hop", "adversarial")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

@dataclass
class RagasEvaluation:
    """Task 2 output: measured rows plus model identifiers reported by m4."""
    results: list[RagasResult]
    evaluator_models: list[str]
    evaluator_details: dict = field(default_factory=dict)

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python scripts/setup_answers.py --live"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict]) -> dict:
    """Provided exporter: preserve the student-produced Task 3 rows verbatim."""
    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_question": [asdict(r) for r in results],
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom,
    }
    return report


def save_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict],
                        path: str = os.path.join(ARTIFACTS_DIR, "ragas_50q.json")) -> None:
    write_json(path, build_phase_a_report(results, clusters, bottom))



# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups: dict[str, list[dict]] = {d: [] for d in DISTRIBUTIONS}
    for item in test_set:
        dist = item["distribution"]
        if dist not in groups:
            raise ValueError(f"Unknown distribution {dist!r} for question {item.get('id')}")
        groups[dist].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> RagasEvaluation:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Gợi ý — import từ Day 18 của bạn:
        from src.m4_eval import evaluate_ragas

    Steps:
        1. Extract questions, answers, contexts, ground_truths từ answers list
        2. Gọi evaluate_ragas() từ m4_eval.py
        3. Kết hợp kết quả với distribution info từ answers list
        4. Return RagasEvaluation(results, raw["metadata"]["evaluator_models"])
           m4 must report its actual LLM/embedding models; never guess JUDGE_MODEL.
    """
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError as exc:
        raise ValueError("Missing src/m4_eval.py; cannot evaluate or declare evaluator models") from exc

    questions     = [a["question"]     for a in answers]
    ans_texts     = [a["answer"]       for a in answers]
    contexts      = [list(a["contexts"]) for a in answers]
    ground_truths = [a["ground_truth"] for a in answers]

    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    per_q = list(raw.get("per_question", []))
    if len(per_q) != len(answers):
        # zip() would silently drop rows and shift metrics onto the wrong question.
        raise ValueError(f"m4 returned {len(per_q)} per_question rows for {len(answers)} answers")

    evaluator_models = (raw.get("metadata") or {}).get("evaluator_models")
    if not isinstance(evaluator_models, list) or not evaluator_models:
        raise ValueError("m4 metadata.evaluator_models missing; fix the m4 adapter, do not guess")

    results = []
    for a, pq in zip(answers, per_q):
        scores = {m: _metric_value(pq, m, a["id"]) for m in METRICS}
        results.append(RagasResult(
            question_id=a["id"], distribution=a["distribution"],
            question=a["question"], answer=a["answer"],
            contexts=a["contexts"], ground_truth=a["ground_truth"],
            **scores,
        ))
    return RagasEvaluation(results, list(evaluator_models), dict(raw["metadata"]))


def _metric_value(record, metric: str, question_id) -> float:
    """Read one metric from an m4 record (object or mapping) as a finite Python float in [0, 1]."""
    value = record.get(metric) if isinstance(record, dict) else getattr(record, metric, None)
    if value is None:
        raise ValueError(f"Question {question_id}: m4 did not return {metric}")
    value = float(value)  # also unwraps NumPy/Pandas scalars
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Question {question_id}: {metric}={value} is not a finite score in [0, 1]")
    return value


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    # Stable sort: ties keep input order so the ranking is reproducible.
    bottom = sorted(results, key=lambda r: r.avg_score)[:10]
    output = []
    for rank, r in enumerate(bottom, 1):
        diagnosis, fix = diagnose(r)
        output.append({
            "rank": rank,
            "question_id": r.question_id,
            "distribution": r.distribution,
            "question": r.question,
            "avg_score": round(r.avg_score, 4),
            "worst_metric": r.worst_metric,
            "scores": {m: round(getattr(r, m), 4) for m in METRICS},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return output


# No leading \b: in source names such as "nghi_phep_nam_v2024.md" the "_" is a word char.
_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])v\s?\d{1,4}(?:\.\d+)?(?![\d.]*\d)", re.IGNORECASE)
_ABSTAIN_RE = re.compile(r"không tìm thấy", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\bkhông\b|\bchưa\b|\bcấm\b", re.IGNORECASE)


def _versions(text: str) -> list[str]:
    return sorted({v.lower().replace(" ", "") for v in _VERSION_RE.findall(text)})


def diagnose(r: RagasResult) -> tuple[str, str]:
    """Metric-tree hypothesis refined by what the question/answer/contexts/reference show.

    The metric only says where the score dropped; the content signals below say why,
    so two rows with the same worst metric can get different diagnoses and fixes.
    """
    base_diag, base_fix = DIAGNOSTIC_TREE[r.worst_metric]
    notes = [f"{base_diag} ({r.worst_metric}={getattr(r, r.worst_metric):.2f})"]
    fixes = [base_fix]
    context_text = "\n".join(r.contexts)

    if not r.contexts:
        notes.append("không có context nào được truy hồi")
        fixes.append("kiểm tra index/search trả kết quả cho câu này")
    if _ABSTAIN_RE.search(r.answer):
        notes.append("pipeline trả 'Không tìm thấy' — cần đọc contexts để phân biệt thiếu retrieval và lỗi tổng hợp")
        fixes.append("đối chiếu điều khoản trong contexts; nếu đã đủ thì sửa generation/ghép chunk, "
                     "nếu còn thiếu thì kiểm tra index và tăng top-k trước rerank")

    ref_versions = _versions(r.ground_truth)
    if len(ref_versions) >= 2:
        ctx_versions = _versions(context_text)
        seen = ", ".join(ctx_versions) if ctx_versions else "không ghi phiên bản"
        notes.append(f"bẫy phiên bản: reference phân biệt {', '.join(ref_versions)}; contexts: {seen}")
        fixes.append("gắn metadata effective/superseded khi index, ưu tiên bản hiện hành "
                     "và yêu cầu answer nêu rõ phiên bản")

    if r.ground_truth.strip().upper().startswith("KHÔNG"):
        if _NEGATION_RE.search(r.answer):
            notes.append("bẫy phủ định: answer có phủ định nhưng điểm vẫn thấp — đọc lại điều kiện đi kèm")
        else:
            notes.append("bẫy phủ định: reference là KHÔNG nhưng answer không chứa phủ định")
        fixes.append("prompt yêu cầu trả lời có/không trước rồi trích điều kiện từ context")

    if r.distribution == "multi_hop" and r.worst_metric in ("context_recall", "context_precision"):
        notes.append("câu nhiều bước: contexts có thể chỉ phủ một trong các nguồn cần ghép")
        fixes.append("tách câu hỏi thành sub-queries hoặc tăng RERANK_TOP_K cho multi-hop")

    return "; ".join(notes), "; ".join(dict.fromkeys(fixes))


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Matrix thống kê worst_metric của mọi câu, không coi mọi câu là một failure.
    Nhóm yếu nhất = distribution có avg_score trung bình thấp nhất (bỏ nhóm rỗng).
    Tổng matrix theo cột chỉ là kích thước nhóm 20/20/10, không đo chất lượng.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    matrix = {metric: {d: 0 for d in DISTRIBUTIONS} for metric in METRICS}
    for r in results:
        matrix[r.worst_metric][r.distribution] += 1

    groups = {d: [r for r in results if r.distribution == d] for d in DISTRIBUTIONS}
    # Empty groups are left out of the division instead of being scored as 0 quality.
    means = {d: sum(r.avg_score for r in rows) / len(rows) for d, rows in groups.items() if rows}
    metric_means = {
        d: {m: sum(getattr(r, m) for r in rows) / len(rows) for m in METRICS}
        for d, rows in groups.items() if rows
    }
    if not means:
        raise ValueError("cluster_analysis needs at least one result")

    # Weakest = lowest mean quality; column totals only restate group sizes (20/20/10).
    dominant_dist = min(means, key=means.get)
    dominant_metric = max(METRICS, key=lambda m: sum(matrix[m].values()))
    weakest_metric_in_dist = min(metric_means[dominant_dist], key=metric_means[dominant_dist].get)

    ranking = ", ".join(f"{d}={means[d]:.3f}" for d in sorted(means, key=means.get))
    insight = (
        f"Mean avg_score theo distribution (thấp→cao): {ranking}. "
        f"'{dominant_dist}' yếu nhất; trong nhóm này metric trung bình thấp nhất là "
        f"'{weakest_metric_in_dist}' ({metric_means[dominant_dist][weakest_metric_in_dist]:.3f}). "
        f"Trên toàn bộ {len(results)} câu, '{dominant_metric}' là worst metric của "
        f"{sum(matrix[dominant_metric].values())} câu (matrix đếm mọi câu, không chỉ câu lỗi). "
        f"Hướng kiểm tra đầu tiên: {DIAGNOSTIC_TREE[dominant_metric][1]}; "
        "kết luận nguyên nhân cần đọc question/answer/contexts của bottom 10."
    )
    return {
        "matrix": matrix,
        "distribution_avg_scores": means,
        "distribution_metric_means": metric_means,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "a", *sys.argv[1:]]))
