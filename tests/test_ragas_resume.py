"""Offline regressions for a measured checkpoint and explicit fallback attribution."""
import pytest
from src import m4_eval as m

pytestmark = pytest.mark.support


def test_skip_primary_uses_only_declared_fallback_for_missing_cells(monkeypatch):
    inputs = ('Q', 'A', ['C'], 'G')
    primary_key = m._row_key(*inputs, m.EVALUATOR_LLM_MODEL)
    fallback_key = m._row_key(*inputs, m.FALLBACK_EVALUATOR_LLM_MODEL)
    cache = {primary_key: {'faithfulness': .8}}
    calls = []
    monkeypatch.setattr(m, 'RAGAS_SKIP_PRIMARY', True)
    monkeypatch.setattr(m, '_load_cache', lambda: cache)

    def measure(inputs, keys, rows, cache, model, **kwargs):
        calls.append(model)
        assert kwargs['only'](0) == ['answer_relevancy', 'context_precision', 'context_recall']
        cache[fallback_key] = {'answer_relevancy': .7, 'context_precision': .6, 'context_recall': .5}

    monkeypatch.setattr(m, '_fill_cells', measure)
    result = m.evaluate_ragas(['Q'], ['A'], [['C']], ['G'])
    assert calls == [m.FALLBACK_EVALUATOR_LLM_MODEL]
    assert result['per_question'][0].faithfulness == .8
    assert result['metadata']['primary_checkpoint_only'] is True
    assert [c['metric'] for c in result['metadata']['fallback_cells']] == ['answer_relevancy', 'context_precision', 'context_recall']


def test_complete_checkpoint_requires_no_fallback_calls(monkeypatch):
    key = m._row_key('Q', 'A', ['C'], 'G', m.EVALUATOR_LLM_MODEL)
    monkeypatch.setattr(m, 'RAGAS_SKIP_PRIMARY', True)
    monkeypatch.setattr(m, '_load_cache', lambda: {key: dict.fromkeys(m.METRIC_NAMES, .8)})
    monkeypatch.setattr(m, '_fill_cells', lambda *a, **kw: pytest.fail('No missing cells to measure'))
    result = m.evaluate_ragas(['Q'], ['A'], [['C']], ['G'])
    assert result['metadata']['fallback_cells'] == []
    assert m.FALLBACK_EVALUATOR_LLM_MODEL not in result['metadata']['evaluator_models']


@pytest.mark.parametrize('skip_primary', [False, True])
def test_unused_primary_model_is_not_reported_as_a_scoring_model(monkeypatch, skip_primary):
    key = m._row_key('Q', 'A', ['C'], 'G', m.FALLBACK_EVALUATOR_LLM_MODEL)
    monkeypatch.setattr(m, 'RAGAS_SKIP_PRIMARY', skip_primary)
    monkeypatch.setattr(m, '_load_cache', lambda: {key: dict.fromkeys(m.METRIC_NAMES, .8)})
    monkeypatch.setattr(m, '_fill_cells', lambda *a, **kw: pytest.fail('Already measured fallback cells must be retained'))
    result = m.evaluate_ragas(['Q'], ['A'], [['C']], ['G'])
    assert set(result['metadata']['evaluator_models']) == {m.EMBEDDING_MODEL, m.FALLBACK_EVALUATOR_LLM_MODEL}


def test_additional_fallback_preserves_measured_cells_and_only_fills_the_gap(monkeypatch):
    data = ('Q', 'A', ['C'], 'G')
    primary_key = m._row_key(*data, m.EVALUATOR_LLM_MODEL)
    fallback_key = m._row_key(*data, m.FALLBACK_EVALUATOR_LLM_MODEL)
    third_key = m._row_key(*data, 'third-evaluator')
    cache = {primary_key: {'faithfulness': .1},
             fallback_key: {'answer_relevancy': .2, 'context_precision': .3},
             third_key: {'faithfulness': .9, 'answer_relevancy': .9}}
    monkeypatch.setattr(m, 'RAGAS_SKIP_PRIMARY', True)
    monkeypatch.setattr(m, 'CHECKPOINT_ONLY_MODELS', {m.FALLBACK_EVALUATOR_LLM_MODEL})
    monkeypatch.setattr(m, 'ADDITIONAL_FALLBACK_EVALUATOR_MODELS', ['third-evaluator'])
    monkeypatch.setattr(m, '_load_cache', lambda: cache)
    calls = []

    def measure(inputs, keys, rows, cache, model, **kwargs):
        calls.append(model)
        assert kwargs['only'](0) == ['context_recall']
        cache[third_key]['context_recall'] = .4

    monkeypatch.setattr(m, '_fill_cells', measure)
    result = m.evaluate_ragas(['Q'], ['A'], [['C']], ['G'])
    row = result['per_question'][0]
    assert [getattr(row, metric) for metric in m.METRIC_NAMES] == [.1, .2, .3, .4]
    assert calls == ['third-evaluator']
    assert [c['model'] for c in result['metadata']['fallback_cells']] == [m.FALLBACK_EVALUATOR_LLM_MODEL, m.FALLBACK_EVALUATOR_LLM_MODEL, 'third-evaluator']


@pytest.mark.parametrize('answer,segments,expected', [
    ('**Bí mật**', ['**Bí mật**'], ['**Bí mật**']),
    ('1 tháng lương', ['1 tháng lương'], ['1 tháng lương']),
    ('Câu đầy đủ.\nTiêu đề', ['Câu đầy đủ.', 'Tiêu đề'], ['Câu đầy đủ.']),
    ('Không được!', ['Không được!'], ['Không được!']),
    ('12 ký tự', [], ['12 ký tự']),
    ('', [], []),
])
def test_statement_extraction_keeps_original_fragment(answer, segments, expected):
    from types import SimpleNamespace
    assert m._statement_sentences(answer, SimpleNamespace(segment=lambda text: segments)) == expected
