"""Recompute report summaries from saved experiments without calling an API."""
from __future__ import annotations

from datetime import datetime, timezone
from collections import Counter
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT_DIR
from src.experiment_support import read_json, write_json
from src.report_contracts import (
    validate_answers, validate_metadata, validate_phase_a, validate_phase_b, validate_phase_c,
)


def main():
    root = Path(ROOT_DIR)
    artifacts, fixtures = root / 'artifacts', root / 'fixtures'
    answers = read_json(artifacts / 'answers_50q.json')
    questions = read_json(fixtures / 'test_set_50q.json')
    humans = read_json(fixtures / 'human_labels_10q.json')
    validate_answers(answers, questions)
    sources = [artifacts / 'answers_50q.json', fixtures / 'test_set_50q.json']
    summary = {'checked_at': datetime.now(timezone.utc).isoformat(),
               'scope': 'Offline validation and recomputation of saved live artifacts; no new live measurements.',
               'answers': {'count': len(answers), 'llm_count': sum(r['generation_mode'] == 'llm' for r in answers)}}
    for phase, name in [('a', 'ragas_50q.json'), ('b', 'judge_results.json'), ('c', 'guard_results.json')]:
        path = artifacts / name
        if not path.exists():
            summary[phase] = {'status': 'missing', 'source': str(path.relative_to(root))}
            continue
        report = read_json(path)
        if phase == 'a':
            validate_phase_a(report, answers)
            inputs = sources[:2]
            metrics = ('faithfulness', 'answer_relevancy', 'context_precision', 'context_recall')
            values = {m: sum(r[m] for r in report['per_question']) / len(report['per_question']) for m in metrics}
            values['avg_score'] = sum(values.values()) / len(metrics)
            values['per_distribution'] = report['per_distribution']
            values['bottom_ids'] = [r['question_id'] for r in report['bottom_10']]
            cells = report.get('evaluator_details', {}).get('fallback_cells', [])
            for cell in cells:
                source = report['per_question'][cell['index']]
                if cell['score'] != source[cell['metric']]:
                    raise ValueError('Fallback cell score does not match measured row')
                if cell['model'] not in report['metadata']['models']['evaluator']:
                    raise ValueError('Fallback cell model is missing from evaluator metadata')
            if len({(c['index'], c['metric']) for c in cells}) != len(cells):
                raise ValueError('Duplicate fallback metric cell')
            values['fallback_cells_by_model'] = dict(Counter(c['model'] for c in cells))
            values['fallback_faithfulness_protocols'] = dict(Counter(c.get('protocol', 'unspecified') for c in cells if c['metric'] == 'faithfulness'))
        elif phase == 'b':
            validate_phase_b(report, answers, questions, humans)
            inputs = [*sources[:2], fixtures / 'human_labels_10q.json']
            cal = report['calibration']
            values = {'cohen_kappa': cal['cohen_kappa'],
                      'agreement_count': sum(a == b for a, b in zip(cal['judge_labels'], cal['human_labels'])),
                      'label_count': len(cal['judge_labels']), 'bias': report['bias_report']}
        else:
            inputs = [fixtures / n for n in ('adversarial_set_20.json', 'clean_set_8.json', 'output_set_4.json')]
            validate_phase_c(report, *(read_json(p) for p in inputs))
            values = {'attacks_passed': sum(r['passed'] for r in report['adversarial']['results']),
                      'clean_allowed': sum(r['actual'] == 'allowed' for r in report['clean']['results']),
                      'output_passed': sum(r['passed'] for r in report['output_results']),
                      'missed_attack_ids': [r['id'] for r in report['adversarial']['results'] if not r['passed']],
                      'input_guard_p95_ms': report['latency']['input_guard_ms']['p95'],
                      'latency_scope': report['latency']['scope'], 'samples': report['latency']['n_samples']}
        validate_metadata(report, inputs)
        sources.extend(p for p in [path, *inputs] if p not in sources)
        summary[phase] = {'status': 'validated', 'source': str(path.relative_to(root)),
                          'live_created_at': report['metadata']['created_at'], 'recomputed': values}
    previous_path = artifacts / 'previous_runs/guard_results_run1_presidio_ner.json'
    if previous_path.exists():
        previous = read_json(previous_path)
        inputs = [fixtures / n for n in ('adversarial_set_20.json', 'clean_set_8.json', 'output_set_4.json')]
        validate_phase_c(previous, *(read_json(p) for p in inputs), require_quality=False)
        validate_metadata(previous, inputs)
        summary['previous_guard'] = {
            'source': str(previous_path.relative_to(root)),
            'attacks_passed': sum(r['passed'] for r in previous['adversarial']['results']),
            'clean_blocked': sum(r['actual'] == 'blocked' for r in previous['clean']['results']),
            'false_positive_rate': previous['clean']['false_positive_rate'],
            'nemo_input_ms': previous['latency']['nemo_input_ms'],
            'input_guard_ms': previous['latency']['input_guard_ms'],
        }
        sources.append(previous_path)
    summary['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(artifacts / 'evidence_audit.json', summary)
    for phase in ('a', 'b', 'c'):
        print(f"Phase {phase.upper()}: {summary[phase]['status']}")
    print('Saved artifacts/evidence_audit.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
