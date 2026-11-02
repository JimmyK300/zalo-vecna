"""Summarize a completed run; never import this into the inference solver."""
import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    # scores.json exists only after all inference has completed.
    rows = json.loads((run / 'scores.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    repo = Path(__file__).resolve().parent.parent
    changed = [p for p, expected in config['source_sha256'].items()
               if hashlib.sha256((repo / p).read_bytes()).hexdigest() != expected]
    by_type = defaultdict(Counter)
    for row in rows:
        by_type[row['reference']['type']][{1: 'correct', -1: 'wrong', 0: 'abstain'}[row['score']]] += 1
    details = []
    for row in rows:
        details.append({
            'id': row['id'], 'business_id': row['business_id'],
            'question': row['question'], 'as_of_date': row['as_of_date'],
            'score': row['score'], 'prediction': row['prediction'], 'reference': row['reference'],
            'stop_reason': row['stop_reason'], 'seconds': row['question_seconds'],
            'model_calls': row['model_calls'],
            'actions': [{'response': t.get('raw_response'), 'error': t.get('error'),
                         'sql': t.get('tool_output', {}).get('sql')}
                        for t in row['trace']],
        })
    durations = sorted(row['question_seconds'] for row in rows)
    audit = {
        'score': summary['score'], 'by_reference_type': dict(by_type),
        'by_stop_reason': dict(Counter(row['stop_reason'] for row in rows)),
        'zero_model_call_questions': sum(row['model_calls'] == 0 for row in rows),
        'businesses': len({row['business_id'] for row in rows}),
        'median_seconds': statistics.median(durations),
        'p95_seconds': durations[min(len(durations)-1, int(.95*len(durations)))],
        'source_files_changed_since_start': changed,
    }
    (run / 'audit_summary.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (run / 'audit_questions.json').write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (run / 'audit_failures.json').write_text(json.dumps([r for r in details if r['score'] != 1], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
