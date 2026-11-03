"""Aggregate completed paired runs; output contains no dataset identifiers/text."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from datamate_bench.scoring import summarize


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def stats(rows):
    times=sorted(r['question_seconds'] for r in rows)
    actions=Counter();sql_seconds=0
    for row in rows:
        for entry in row['trace']:
            try:kind=json.loads(entry.get('raw_response','{}')).get('action','invalid')
            except (ValueError,AttributeError):kind='invalid'
            actions[kind]+=1
            sql_seconds+=entry.get('tool_output',{}).get('seconds',0)
    return {**summarize([r['score'] for r in rows]),'qa_seconds':sum(times),
            'median_seconds':statistics.median(times),'p95_seconds':times[min(len(times)-1,int(.95*len(times)))],
            'model_calls':sum(r['model_calls'] for r in rows),'actions':dict(actions),
            'recorded_direct_sql_seconds':sql_seconds,
            'note_sql_seconds':'Only direct sql actions record execution time here; metric internal queries are not included.',
            'stops':dict(Counter(r['stop_reason'] for r in rows))}


def paired(left,right):
    by_id={r['id']:r for r in left}
    if set(by_id)!={r['id'] for r in right}:raise ValueError('Paired question IDs differ')
    for r in right:
        if r['question']!=by_id[r['id']]['question'] or r['reference']!=by_id[r['id']]['reference']:
            raise ValueError('Paired question/reference content differs')
    return {'gained_correct':sum(by_id[r['id']]['score']!=1 and r['score']==1 for r in right),
            'lost_correct':sum(by_id[r['id']]['score']==1 and r['score']!=1 for r in right),
            'score_improved':sum(r['score']>by_id[r['id']]['score'] for r in right),
            'score_regressed':sum(r['score']<by_id[r['id']]['score'] for r in right)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--generated',type=Path,required=True)
    parser.add_argument('--arithmetic',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output exists; preserve old reports')
    before,after,generated=[read(p/'scores.json') for p in [args.before,args.after,args.generated]]
    configs=[read(p/'config.json') for p in [args.before,args.after,args.generated]]
    if any(c['input_fingerprints']!=configs[0]['input_fingerprints'] for c in configs[1:]):
        raise ValueError('Input fingerprints differ')
    subset_ids={r['id'] for r in generated}
    subset=[r for r in after if r['id'] in subset_ids]
    summary={'before_100':stats(before),'after_100':stats(after),'before_after_paired':paired(before,after),
             'hybrid_20':stats(subset),'generated_sql_20':stats(generated),
             'hybrid_to_generated_paired':paired(subset,generated),
             'arithmetic':read(args.arithmetic)['summary'],
             'source_sha256':{'before':configs[0]['source_sha256'],'after':configs[1]['source_sha256'],
                              'generated':configs[2]['source_sha256']},
             'model_identity':read(ROOT/'models/model_manifest.json'),
             'runtime_identity':read(ROOT/'tools/llama.cpp/runtime_manifest.json'),
             'warm_cache_wall_seconds':{name:read(p/'summary.json')['wall_seconds']
                                        for name,p in [('before',args.before),('after',args.after),('generated',args.generated)]},
             'limitations':['Question-held-out only; business formats may overlap development.',
                            'Single sequential run per configuration; no multi-seed variance estimate.',
                            'Warm data caches and model already loaded; not a cold-start six-hour compliance test.',
                            'Generated SQL and hybrid are both SQL-based; this is not SQL versus no SQL.',
                            'Local scorer, not official competition scoring.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'source_sha256','arithmetic'}},indent=2))


if __name__=='__main__':main()
