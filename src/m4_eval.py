from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math, hashlib
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, OPENAI_API_KEY, ROOT_DIR, TEST_SET_PATH

# The evaluator model is kept separate from GENERATION_MODEL/JUDGE_MODEL so Phase A
# metadata names what RAGAS actually called. Requests go to Groq via config.py env routing.
# Groq free tier caps each model at 200K tokens/day; generation + Phase B/C had already used
# gpt-oss-20b's quota, and RAGAS on 50 questions needs ~300-450K tokens. Qwen has its own
# daily quota and answers without reasoning tokens, so it is the evaluator here.
EVALUATOR_LLM_MODEL = os.getenv("RAGAS_EVALUATOR_MODEL", "qwen/qwen3.8-27b")
# Cells that stay NaN after the primary retries (e.g. faithfulness of a fragment answer such
# as "1 tháng lương", from which the primary model extracts no statements) are measured by a
# second, declared evaluator instead of being left NaN or filled with an invented score.
FALLBACK_EVALUATOR_LLM_MODEL = os.getenv("RAGAS_FALLBACK_EVALUATOR_MODEL", "openai/gpt-oss-120b")
# Evaluator endpoint. With OPENROUTER_API_KEY set, only the RAGAS evaluator goes to OpenRouter
# (same model IDs, own quota); generation/judge/guard keep the Groq routing from config.py.
# Groq free tier allows 200K tokens/day per model, which ran out after 18 of 50 questions.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
RAGAS_EVALUATOR_BASE_URL = os.getenv(
    "RAGAS_EVALUATOR_BASE_URL", "https://openrouter.ai/api/v1" if OPENROUTER_API_KEY else "")
RAGAS_EVALUATOR_PROVIDER = "openrouter" if "openrouter.ai" in RAGAS_EVALUATOR_BASE_URL else "groq"
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "4" if RAGAS_EVALUATOR_PROVIDER == "openrouter" else "1"))
RAGAS_BATCH_SIZE = int(os.getenv("RAGAS_BATCH_SIZE", "5"))
RAGAS_JOB_TIMEOUT = int(os.getenv("RAGAS_JOB_TIMEOUT", "300"))
RAGAS_REQUEST_TIMEOUT = float(os.getenv("RAGAS_REQUEST_TIMEOUT", "90"))
RAGAS_MAX_TOKENS = int(os.getenv("RAGAS_MAX_TOKENS", "2048"))
# Per-row checkpoint of measured scores (gitignored); keyed by inputs + evaluator models.
RAGAS_CACHE_PATH = os.getenv("RAGAS_CACHE_PATH", os.path.join(ROOT_DIR, ".cache", "ragas_rows.json"))
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
NAN_RETRY_ROUNDS = 2


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    Returns per_question in input order plus metadata.evaluator_models. Failures raise:
    in Phase A an empty or partial result would be mistaken for a measurement.
    """
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts, and ground_truths must have equal lengths")
    if not questions:
        return _empty_evaluation()

    inputs = (questions, answers, contexts, ground_truths)
    n = len(questions)
    cache = _load_cache()
    primary_keys = [_row_key(*row, EVALUATOR_LLM_MODEL) for row in zip(*inputs)]
    _fill_cells(inputs, primary_keys, set(range(n)), cache, EVALUATOR_LLM_MODEL,
                rounds=1 + NAN_RETRY_ROUNDS, label="RAGAS")

    # Cells the primary evaluator still could not score go to the declared fallback model.
    primary_missing = {i for i in range(n) if _missing(cache, primary_keys[i])}
    fallback_keys = [_row_key(*row, FALLBACK_EVALUATOR_LLM_MODEL) for row in zip(*inputs)]
    fallback_cells: list[dict] = []
    if primary_missing:
        _fill_cells(inputs, fallback_keys, primary_missing, cache, FALLBACK_EVALUATOR_LLM_MODEL,
                    rounds=1 + NAN_RETRY_ROUNDS, label=f"RAGAS fallback {FALLBACK_EVALUATOR_LLM_MODEL}",
                    only=lambda i: _missing(cache, primary_keys[i]))

    rows = []
    for i in range(n):
        primary = cache.get(primary_keys[i], {})
        fallback = cache.get(fallback_keys[i], {})
        row = {}
        for m in METRIC_NAMES:
            if _is_score(primary.get(m)):
                row[m] = primary[m]
            else:
                row[m] = fallback.get(m, math.nan)
                fallback_cells.append({"index": i, "metric": m, "model": FALLBACK_EVALUATOR_LLM_MODEL,
                                       "score": row[m]})
        rows.append(row)
    for cell in fallback_cells:
        print(f"  fallback cell: row {cell['index']} {cell['metric']}={cell['score']} "
              f"({cell['model']})", flush=True)

    per_question = [
        EvalResult(question=q, answer=a, contexts=list(c), ground_truth=g,
                   **{name: row[name] for name in METRIC_NAMES})
        for q, a, c, g, row in zip(questions, answers, contexts, ground_truths, rows)
    ]
    evaluator_models = [EVALUATOR_LLM_MODEL, EMBEDDING_MODEL]
    if fallback_cells:
        evaluator_models.insert(1, FALLBACK_EVALUATOR_LLM_MODEL)
    return {
        **{name: _finite_mean([row[name] for row in rows]) for name in METRIC_NAMES},
        "per_question": per_question,
        "metadata": {"evaluator_models": evaluator_models, "fallback_cells": fallback_cells},
    }


def _missing(cache: dict, key: str, metrics=METRIC_NAMES) -> list[str]:
    entry = cache.get(key, {})
    return [m for m in metrics if not _is_score(entry.get(m))]


def _fill_cells(inputs, keys, rows_in_scope, cache, model, *, rounds, label, only=None) -> None:
    """Measure missing cells for `model`, in batches, checkpointing the cache after each batch.

    Batching + checkpoint means an interrupted run (sleep, dropped connection, quota) resumes
    instead of losing measured rows. Timeouts / parse failures surface as NaN; later rounds
    re-measure only those cells instead of keeping NaN or substituting a score.
    `only(i)` limits which metrics count as missing (fallback: the primary's missing cells).
    """
    questions, answers, contexts, ground_truths = inputs
    provider = None

    def missing(i: int) -> list[str]:
        wanted = METRIC_NAMES if only is None else only(i)
        return _missing(cache, keys[i], wanted)

    for round_index in range(rounds):
        todo = sorted(i for i in rows_in_scope if missing(i))
        if not todo:
            return
        if provider is None:
            provider = _ragas_provider_config(model)
            print(f"  evaluator {model} via {RAGAS_EVALUATOR_PROVIDER} "
                  f"(workers={RAGAS_MAX_WORKERS})", flush=True)
        retry = "" if round_index == 0 else f" retry {round_index}"
        print(f"  {label}{retry}: {len(todo)} rows to evaluate "
              f"({len(rows_in_scope) - len(todo)} already measured/cached)", flush=True)
        for start in range(0, len(todo), RAGAS_BATCH_SIZE):
            batch = todo[start:start + RAGAS_BATCH_SIZE]
            needed = [m for m in METRIC_NAMES if any(m in missing(i) for i in batch)]
            measured = _evaluate_rows([questions[i] for i in batch], [answers[i] for i in batch],
                                      [contexts[i] for i in batch], [ground_truths[i] for i in batch],
                                      provider, needed)
            for i, row in zip(batch, measured):
                entry = cache.setdefault(keys[i], {})
                for name, value in row.items():
                    if _is_score(value) and not _is_score(entry.get(name)):
                        entry[name] = value
            _save_cache(cache)
            done = sum(1 for i in rows_in_scope if not missing(i))
            print(f"  checkpoint: {done}/{len(rows_in_scope)} rows complete", flush=True)


def _evaluate_rows(questions, answers, contexts, ground_truths, provider,
                   metric_names=METRIC_NAMES) -> list[dict]:
    from datasets import Dataset
    from ragas import RunConfig, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    # Groq accepts one completion per request (n=1); RAGAS defaults to 3 generated
    # questions for answer_relevancy, so lower strictness to 1 (as in Day 18).
    answer_relevancy.strictness = 1
    available = {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
                 "context_precision": context_precision, "context_recall": context_recall}

    dataset = Dataset.from_dict({
        "question": list(questions),
        "answer": list(answers),
        "contexts": [list(c) for c in contexts],
        "ground_truth": list(ground_truths),
    })
    evaluation = evaluate(
        dataset,
        metrics=[available[name] for name in metric_names],
        # One worker + retries: rate-limit responses are retried, not turned into NaN. The job
        # timeout leaves room for Groq retry-after waits (8K tokens/min on the free tier).
        run_config=RunConfig(timeout=RAGAS_JOB_TIMEOUT, max_retries=20, max_wait=60,
                             max_workers=RAGAS_MAX_WORKERS),
        raise_exceptions=False,
        **provider,
    )
    dataframe = evaluation.to_pandas()
    if len(dataframe) != len(questions):
        raise ValueError(f"RAGAS returned {len(dataframe)} rows for {len(questions)} questions")
    return [{name: float(row[name]) if row.get(name) is not None else math.nan
             for name in metric_names}
            for _, row in dataframe.iterrows()]


def _is_score(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0


def _row_key(question: str, answer: str, contexts: list[str], ground_truth: str,
             model: str = EVALUATOR_LLM_MODEL) -> str:
    """Cache identity: the exact evaluated inputs plus the evaluator configuration."""
    payload = json.dumps({"q": question, "a": answer, "c": list(contexts), "g": ground_truth,
                          "llm": model, "emb": EMBEDDING_MODEL,
                          "answer_relevancy_strictness": 1},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if not os.path.exists(RAGAS_CACHE_PATH):
        return {}
    with open(RAGAS_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(RAGAS_CACHE_PATH), exist_ok=True)
    tmp = RAGAS_CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    os.replace(tmp, RAGAS_CACHE_PATH)  # atomic: an interrupt never leaves a half-written cache


def _finite_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu.",
            "Thắt chặt system prompt và đặt temperature về 0.",
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng.",
            "Cải thiện chunking hoặc bổ sung truy vấn từ khóa BM25.",
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp ở vị trí đầu.",
            "Bổ sung Cross-Encoder reranking hoặc lọc metadata.",
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi.",
            "Viết lại prompt để yêu cầu trả lời trực tiếp hơn.",
        ),
    }
    analyses = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyses.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": sum(metrics.values()) / len(metrics),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(analyses, key=lambda item: item["score"])[:bottom_n]


def _empty_evaluation() -> dict:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }


_EMBEDDINGS = None


def _reasoning_options(model: str) -> dict | None:
    """Keep evaluator behaviour the same on both providers.

    gpt-oss: low reasoning effort (Groq: top-level `reasoning_effort`; OpenRouter: `reasoning`).
    Qwen: Groq serves it without reasoning tokens, so reasoning is disabled on OpenRouter too.
    """
    if RAGAS_EVALUATOR_PROVIDER == "openrouter":
        if "gpt-oss" in model:
            return {"reasoning": {"effort": "low"}}
        return {"reasoning": {"enabled": False}}
    return {"reasoning_effort": "low"} if "gpt-oss" in model else None


def _ragas_provider_config(model: str = EVALUATOR_LLM_MODEL) -> dict:
    """Explicit evaluator LLM + embeddings, so the reported model IDs are the ones used."""
    global _EMBEDDINGS
    if not (OPENROUTER_API_KEY or OPENAI_API_KEY):
        raise ValueError("OPENROUTER_API_KEY or GROQ_API_KEY/OPENAI_API_KEY is required for the RAGAS evaluator")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # Base URL comes from OPENAI_API_BASE (set by config.py when GROQ_API_KEY is present).
    # The SDK retry honours Groq's retry-after header (tight 8K tokens/min), so a 429
    # waits exactly as long as needed; RAGAS RunConfig retries remain as a second layer.
    # `timeout` must be set here: the HTTP client reads it only at construction, and RAGAS's
    # later `request_timeout` assignment does not reach it. Without it, a dead connection
    # (e.g. after host sleep) hangs every call until the RAGAS job timeout.
    # Explicit max_tokens: without it OpenRouter reserves the model's full output window
    # (32K–131K tokens) against the credit balance and rejects the request with 402.
    llm = ChatOpenAI(model=model, temperature=0, max_retries=6, timeout=RAGAS_REQUEST_TIMEOUT,
                     max_tokens=RAGAS_MAX_TOKENS,
                     openai_api_key=(OPENROUTER_API_KEY if RAGAS_EVALUATOR_PROVIDER == "openrouter"
                                     else OPENAI_API_KEY),
                     openai_api_base=RAGAS_EVALUATOR_BASE_URL or None,
                     extra_body=_reasoning_options(model))
    # Local BGE-M3 (same model as retrieval) for answer_relevancy's similarity step; loaded once
    # and shared by the primary and fallback evaluators.
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return {
        "llm": LangchainLLMWrapper(llm),
        "embeddings": LangchainEmbeddingsWrapper(_EMBEDDINGS),
    }


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    def json_value(value):
        """Keep JSON standards-compliant when a provider omits a metric."""
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_value(item) for item in value]
        return value

    report = {
        "aggregate": json_value({k: v for k, v in results.items() if k != "per_question"}),
        "num_questions": len(results.get("per_question", [])),
        "failures": json_value(failures),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
