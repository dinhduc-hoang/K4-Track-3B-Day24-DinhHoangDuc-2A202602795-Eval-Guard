from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


# Structured identifiers this layer is responsible for. The English spaCy NER behind
# PERSON/LOCATION/NRP/DATE_TIME mislabels ordinary Vietnamese words ("Hãy", "bao nhiêu",
# "Xin lỗi") and blocked 7/8 clean questions in the first live run, so those types are
# not used here; requests about another person's data are left to the NeMo input policy.
PII_ENTITIES = frozenset({"VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER"})


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = [r for r in analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
               if r.entity_type in PII_ENTITIES]
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    # Default anonymizer operator replaces each span with <ENTITY_TYPE>.
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {"type": r.entity_type, "text": text[r.start:r.end],
         "score": round(float(r.score), 3), "start": r.start, "end": r.end}
        for r in sorted(results, key=lambda r: (r.start, r.end))
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — câu từ chối Colang 1.0; built-in self-check flows được bật trong YAML
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    result = await rails.generate_async(
        messages=[{"role": "user", "content": text}],
        options={"rails": ["input"], "log": {"activated_rails": True}},
    )
    response = result.response[0]["content"]
    # The decision comes from the rail log; refusal words can appear in clean inputs too.
    stopped = _stopped_rails(result)
    return {
        "allowed":        not stopped,
        "blocked_reason": _rail_reason("nemo_input_rail", stopped),
        "response":       response,
    }


def _stopped_rails(result) -> list:
    return [rail for rail in result.log.activated_rails if getattr(rail, "stop", False)]


def _rail_reason(prefix: str, stopped: list) -> str | None:
    if not stopped:
        return None
    names = [name for name in (getattr(rail, "name", None) for rail in stopped)
             if isinstance(name, str) and name]
    return f"{prefix}: {', '.join(names)}" if names else prefix


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra PII và nội dung nhạy cảm. Self-check này không thay RAGAS faithfulness
    hoặc kiểm tra factual accuracy với retrieved contexts.

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    # The existing answer is passed as the assistant turn; nothing new is generated.
    result = await rails.generate_async(
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        options={"rails": ["output"], "log": {"activated_rails": True}},
    )
    response = result.response[0]["content"]
    stopped = _stopped_rails(result)
    if not stopped:
        return {"safe": True, "flagged_reason": None, "final_answer": answer}
    # A flagged answer must never reach the user unchanged, even if the rail echoed it.
    final_answer = response if response and response != answer else OUTPUT_REDACTED_MESSAGE
    return {
        "safe":           False,
        "flagged_reason": _rail_reason("nemo_output_rail", stopped),
        "final_answer":   final_answer,
    }


OUTPUT_REDACTED_MESSAGE = ("Xin lỗi, câu trả lời đã bị chặn vì có thể chứa dữ liệu cá nhân "
                           "hoặc thông tin bí mật.")


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    async def _run_all():
        results = []
        for item in adversarial_set:
            text = item["input"]
            blocked_by = None
            blocked_reason = None

            # Layer 1: Presidio PII (synchronous, fast). A hit stops the request here.
            pii_result = pii_scan(text, analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"
                types = sorted({e["type"] for e in pii_result["entities"]})
                blocked_reason = "pii: " + ", ".join(types) if types else "pii"

            # Layer 2: NeMo input rail, awaited inside the single event loop.
            if blocked_by is None:
                rail_result = await check_input_rail(text, rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"
                    blocked_reason = rail_result["blocked_reason"]

            # `expected` is read only after the guard decided, to score the decision.
            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":             item["id"],
                "category":       item["category"],
                "input":          text,
                "expected":       item["expected"],
                "actual":         actual,
                "blocked_by":     blocked_by,
                "blocked_reason": blocked_reason,
                "passed":         actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())  # one loop for the whole batch, not one per input
    passed = sum(1 for r in results if r["passed"])
    print(f"Guard input suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: measure INPUT layers sequentially, not end-to-end guard/RAG latency.

    Reuse engines. Inside ONE event loop, execute two real warm-up calls per layer,
    excluded from samples. Then collect exactly n_runs timed samples (cycle clean
    inputs), always measuring BOTH PII and NeMo input even if PII blocks.
    Use perf_counter; keep raw milliseconds. summarize_latency() supplies linear
    percentiles at (n - 1) * q. Submission requires n_runs >= 20.
    Short-circuit request timing, RAG and output timing are outside this scope.
    """
    from src.experiment_support import summarize_latency
    if not test_inputs or n_runs < 1:
        raise ValueError("Need inputs and n_runs >= 1")
    # Engines are built before timing so initialisation never lands in a sample.
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    async def _measure():
        # Real warm-up calls on both layers (model load, HTTP client, caches), then discarded.
        for i in range(WARMUP_RUNS):
            text = test_inputs[i % len(test_inputs)]
            pii_scan(text, analyzer, anonymizer)
            await check_input_rail(text, rails)

        samples = []
        for i in range(n_runs):
            text = test_inputs[i % len(test_inputs)]
            # Both layers always run: this measures the layers, not a short-circuited request.
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            t2 = time.perf_counter()
            pii_ms = (t1 - t0) * 1000
            nemo_ms = (t2 - t1) * 1000
            samples.append({"presidio_ms": pii_ms, "nemo_input_ms": nemo_ms,
                            "input_guard_ms": pii_ms + nemo_ms})
        return samples

    return summarize_latency(asyncio.run(_measure()), warmup_runs=WARMUP_RUNS)


WARMUP_RUNS = 2


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "c", *sys.argv[1:]]))
