"""Offline regressions for submission schemas, provenance and model attribution."""
from dataclasses import asdict
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config import ROOT_DIR, TEST_SET_PATH, HUMAN_LABELS_PATH
from src.experiment_support import metadata, read_json, write_json
from src.phase_a_ragas import RagasEvaluation, RagasResult, build_phase_a_report
from src.phase_b_judge import JudgeResult
from src.report_contracts import (PAIR_SOURCES, validate_answers, validate_metadata,
                                  validate_phase_a, validate_phase_b, validate_schema)

pytestmark = pytest.mark.support


@pytest.fixture
def answers():
    return [{**q, "answer": "pipeline answer", "contexts": ["context"],
             "generation_mode": "llm", "generation_model": "generator-test"}
            for q in read_json(TEST_SET_PATH)]


@pytest.fixture
def phase_a_data(tmp_path, answers):
    answer_path = tmp_path / "answers_50q.json"
    write_json(answer_path, answers)
    results = [RagasResult(a["id"], a["distribution"], a["question"], a["answer"],
                           a["contexts"], a["ground_truth"], .8, .8, .8, .8) for a in answers]
    bottom = [{"rank": i + 1, "question_id": r.question_id, "distribution": r.distribution,
               "question": r.question, "avg_score": r.avg_score, "worst_metric": r.worst_metric,
               "diagnosis": "student diagnosis", "suggested_fix": "student fix"}
              for i, r in enumerate(results[:10])]
    matrix = {m: {d: 0 for d in ("factual", "multi_hop", "adversarial")}
              for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    matrix["faithfulness"] = {"factual": 20, "multi_hop": 20, "adversarial": 10}
    clusters = {"matrix": matrix, "dominant_failure_distribution": "factual",
                "dominant_failure_metric": "faithfulness", "insight": "Evidence from measured rows."}
    report = build_phase_a_report(results, clusters, bottom)
    report["metadata"] = metadata([answer_path, TEST_SET_PATH], models={
        "generation": ["generator-test"], "evaluator": ["eval-alternative", "embedding-alternative"],
        "judge": [], "guard": []})
    return report, answers, results, bottom, clusters, answer_path


@pytest.fixture
def phase_b_data(tmp_path, answers):
    questions, humans = read_json(TEST_SET_PATH), read_json(HUMAN_LABELS_PATH)
    by_answer = {a["id"]: a for a in answers}
    by_question = {q["id"]: q for q in questions}
    by_human = {h["question_id"]: h for h in humans}
    pairs = []
    for qid in (1, 5, 12, 21, 41):
        result = JudgeResult(by_question[qid]["question"], by_answer[qid]["answer"],
                             by_human[qid]["model_answer"], "A", "A", "A", "first", "second",
                             True, {"A": .9, "B": .2}, {"A": .9, "B": .2})
        pairs.append({"question_id": qid, **asdict(result)})
    longer = sum(len(row["answer_a"]) > len(row["answer_b"]) for row in pairs)
    judgments = [{"question_id": h["question_id"], "question": h["question"],
                  "model_answer": h["model_answer"], "label": h["human_label"], "reasoning": "reference"}
                 for h in humans]
    answer_path = tmp_path / "answers_50q.json"
    write_json(answer_path, answers)
    report = {"metadata": metadata([answer_path, TEST_SET_PATH, HUMAN_LABELS_PATH], models={
        "generation": ["generator-test"], "evaluator": [], "judge": ["judge-test"], "guard": []}),
        "pair_sources": dict(PAIR_SOURCES), "pairwise_results": pairs, "binary_judgments": judgments,
        "calibration": {"question_ids": [h["question_id"] for h in humans],
                        "judge_labels": [h["human_label"] for h in humans],
                        "human_labels": [h["human_label"] for h in humans], "cohen_kappa": 1},
        "bias_report": {"total_judged": 5, "position_bias_count": 0, "position_bias_rate": 0,
                        "verbosity_bias": longer / 5, "interpretation": "Mocked test evidence.",
                        "verbosity_details": {"a_wins_a_longer": longer,
                                              "b_wins_b_longer": 0, "total_decisive": 5}}}
    return report, answers, questions, humans


def test_valid_answers_and_reports(answers, phase_a_data, phase_b_data):
    validate_answers(answers, read_json(TEST_SET_PATH))
    validate_phase_a(*phase_a_data[:2])
    validate_phase_b(*phase_b_data)
    validate_metadata(phase_a_data[0], [phase_a_data[-1], TEST_SET_PATH])


@pytest.mark.parametrize("field,value", [
    ("generation_mode", None), ("generation_mode", "pretend"), ("generation_mode", True),
    ("generation_model", None), ("generation_model", 123), ("generation_model", "   ")])
def test_answers_reject_missing_or_wrong_generation_fields(answers, field, value):
    if value is None:
        answers[0].pop(field)
    else:
        answers[0][field] = value
    with pytest.raises(ValueError, match=field):
        validate_answers(answers, read_json(TEST_SET_PATH))


def test_fallback_requires_null_model(answers):
    answers[0]["generation_mode"] = "extractive_fallback"
    with pytest.raises(ValueError, match="generation_model"):
        validate_answers(answers, read_json(TEST_SET_PATH))
    answers[0]["generation_model"] = None
    validate_answers(answers, read_json(TEST_SET_PATH))


@pytest.mark.parametrize("field", [None, "A", "B", "reference", "extra"])
def test_pair_sources_missing_or_wrong_is_rejected(phase_b_data, field):
    report, answers, questions, humans = phase_b_data
    if field is None:
        del report["pair_sources"]
    elif field == "extra":
        report["pair_sources"][field] = "unverified source"
    else:
        report["pair_sources"][field] = "wrong-file.json:answer"
    with pytest.raises(ValueError, match="pair_sources"):
        validate_phase_b(report, answers, questions, humans)


@pytest.mark.parametrize("mutation", ["missing_metadata", "old_version", "wrong_models", "bad_timestamp", "bool_metric"])
def test_phase_a_schema_rejects_missing_and_wrong_fields(phase_a_data, mutation):
    report, answers = phase_a_data[:2]
    if mutation == "missing_metadata":
        del report["metadata"]
    elif mutation == "old_version":
        report["metadata"]["format_version"] = 1
    elif mutation == "wrong_models":
        report["metadata"]["models"]["evaluator"] = "not a list"
    elif mutation == "bad_timestamp":
        report["metadata"]["created_at"] = "yesterday"
    else:
        report["per_question"][0]["faithfulness"] = True
    with pytest.raises(ValueError, match="phase_a schema"):
        validate_phase_a(report, answers)


def test_phase_a_rejects_generation_model_mismatch(phase_a_data):
    report, answers = phase_a_data[:2]
    report["metadata"]["models"]["generation"] = ["wrong-generator"]
    with pytest.raises(ValueError, match="generation models"):
        validate_phase_a(report, answers)


@pytest.mark.parametrize("mutation", ["empty_judge", "missing_details", "wrong_detail_count", "bool_score"])
def test_phase_b_schema_and_detail_contracts(phase_b_data, mutation):
    report, answers, questions, humans = phase_b_data
    if mutation == "empty_judge":
        report["metadata"]["models"]["judge"] = []
    elif mutation == "missing_details":
        del report["bias_report"]["verbosity_details"]
    elif mutation == "wrong_detail_count":
        report["bias_report"]["verbosity_details"]["total_decisive"] = 4
    else:
        report["pairwise_results"][0]["scores_pass1"]["A"] = True
    with pytest.raises(ValueError):
        validate_phase_b(report, answers, questions, humans)


def test_phase_a_runner_keeps_actual_evaluator_models(phase_a_data, monkeypatch, tmp_path):
    from src import experiment_runners as runners, phase_a_ragas as phase
    report, answers, results, bottom, clusters, answer_path = phase_a_data
    monkeypatch.setattr(runners, "ANSWERS_PATH", answer_path)
    monkeypatch.setattr(runners, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(phase, "run_ragas_50q", lambda rows: RagasEvaluation(results, ["eval-alternative", "embedding-alternative"]))
    monkeypatch.setattr(phase, "bottom_10", lambda rows: bottom)
    monkeypatch.setattr(phase, "cluster_analysis", lambda rows: clusters)
    actual = runners.run_phase_a()
    assert actual['evaluator_details'] == {}
    assert actual["metadata"]["models"] == report["metadata"]["models"]
    assert "model" not in actual["metadata"]
    assert (tmp_path / "artifacts/ragas_50q.json").is_file()


@pytest.mark.parametrize("evaluation", [RagasEvaluation([], []), RagasEvaluation([], [" "]), []])
def test_phase_a_runner_never_guesses_missing_evaluator(phase_a_data, monkeypatch, evaluation):
    from src import experiment_runners as runners, phase_a_ragas as phase
    monkeypatch.setattr(runners, "ANSWERS_PATH", phase_a_data[-1])
    monkeypatch.setattr(phase, "run_ragas_50q", lambda rows: evaluation)
    with pytest.raises(ValueError, match="evaluator"):
        runners.run_phase_a()


def test_generation_records_provider_model_and_null_fallback(monkeypatch):
    from scripts import setup_answers
    monkeypatch.setattr(setup_answers, "GENERATION_MODEL", "generation-requested")
    search = SimpleNamespace(search=lambda q: [SimpleNamespace(text="context", score=.8, metadata={})])
    reranker = SimpleNamespace(rerank=lambda q, docs, top_k: [SimpleNamespace(text="context")])
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(model="generation-provider-snapshot",
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])
    assert setup_answers.run_query("Q", search, reranker, 3, client)[2:] == (
        "llm", "generation-provider-snapshot")
    assert client.chat.completions.create.call_args.kwargs["model"] == "generation-requested"
    client.chat.completions.create.side_effect = RuntimeError("generation failed")
    assert setup_answers.run_query("Q", search, reranker, 3, client)[2:] == ("extractive_fallback", None)


def test_checker_really_uses_all_supplied_report_schemas(tmp_path, answers, capsys):
    from scripts import check_lab
    source = Path(ROOT_DIR)
    shutil.copytree(source / "doc/schemas", tmp_path / "doc/schemas")
    shutil.copytree(source / "fixtures", tmp_path / "fixtures")
    write_json(tmp_path / "artifacts/answers_50q.json", answers)
    for name in ("ragas_50q", "judge_results", "guard_results"):
        write_json(tmp_path / f"artifacts/{name}.json", {})
    assert check_lab.main(["--root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "PASS answers_50q.json" in output
    for phase in ("a", "b", "c"):
        assert f"phase_{phase} schema at $" in output


def test_schema_validation_works_outside_repo_cwd(answers, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    validate_schema(answers, "answers")
