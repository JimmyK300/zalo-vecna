"""Publish aggregate verification results; detailed questions/labels stay local."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from datamate_bench.scoring import summarize
from datamate_bench.suite import digest
from datamate_bench.__main__ import source_fingerprints


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/verification40_review_v2')
    parser.add_argument('--base',type=Path,default=ROOT/'artifacts/verification40_base')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/answer_verification_summary.json')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing reports')
    rows=read(args.run/'results.json');summary=read(args.run/'summary.json');config=read(args.run/'config.json')
    if config['baseline_config']['source_sha256']!=source_fingerprints():raise ValueError('Solver changed')
    if any(digest(ROOT/p)!=h for p,h in config['verification_source_sha256'].items()):raise ValueError('Verification code changed')
    names={1:'correct',-1:'wrong',0:'abstain'}
    transitions={arm:dict(Counter(names[r['baseline']['score']]+'->'+names[r[arm]['score']] for r in rows)) for arm in ['code','review']}
    raw_decisions=Counter()
    error_categories=Counter()
    for r in rows:
        try:decision=json.loads(r['review'].get('raw_response','{}')).get('decision','error')
        except (ValueError,AttributeError):decision='error'
        raw_decisions[decision]+=1
        error=r['review'].get('review_error','')
        if error:
            category=('generation_budget' if 'max_tokens' in error else
                      'unknown_sql_schema' if 'no such' in error else
                      'unsupported_parameters' if 'Unsupported parameters' in error else
                      'result_shape_or_empty' if 'SQL answer needs' in error or 'Empty/truncated' in error else 'other')
            error_categories[category]+=1
    flagged=[r for r in rows if r['code']['issues']]
    # Diagnostic policy declared in the audit implementation before scoring was
    # inspected: use existing review outputs only where code found an issue.
    # No new generations; timing is an estimate from the recorded per-case costs.
    chosen={r['id']:r['review'] if r['code']['issues'] else r['baseline'] for r in rows}
    selective_scores=[chosen[r['id']]['score'] for r in rows]
    selective={**summarize(selective_scores),'review_calls_if_deployed':len(flagged),
               'gained_correct_vs_baseline':sum(r['baseline']['score']!=1 and chosen[r['id']]['score']==1 for r in rows),
               'lost_correct_vs_baseline':sum(r['baseline']['score']==1 and chosen[r['id']]['score']!=1 for r in rows),
               'estimated_extra_seconds':sum(r['review']['extra_seconds'] for r in flagged)+sum(r['code']['extra_seconds'] for r in rows if not r['code']['issues']),
               'note':'Diagnostic reuse of the same reviewer outputs, not a separate blind run. Timing estimates ignore changed server cache/order.'}
    report={**summary,'paired_transitions':transitions,'raw_model_decisions':dict(raw_decisions),'review_error_categories':dict(error_categories),
            'code_flagged_questions':len(flagged),'hard_flagged_questions':sum(any(i['severity']=='hard' for i in r['code']['issues']) for r in rows),
            'successful_repairs_leaving_same_answer':sum(r['review']['repair_executed'] and r['review']['prediction']==r['baseline']['prediction'] for r in rows),
            'selective_review_diagnostic':selective,
            'baseline_model_calls':sum(r['baseline']['model_calls'] for r in rows),
            'completed_review_texts_saved':sum('raw_response' in r['review'] for r in rows),
            'review_call_attempts':sum(r['review']['context_tokens_estimate']<=15000 for r in rows),
            'context_tokens_estimate_min':min(r['review']['context_tokens_estimate'] for r in rows),
            'context_tokens_estimate_max':max(r['review']['context_tokens_estimate'] for r in rows),
            'compressed_contexts':sum(bool(r['review']['compression']) for r in rows),
            'source_sha256':config['verification_source_sha256'],'baseline_source_sha256':config['baseline_config']['source_sha256'],
            'input_sha256':{k:v for k,v in config['baseline_config']['input_fingerprints'].items() if k in ['questions_sha256','answers_sha256','rulebook_sha256']},
            'model_identity':read(ROOT/'models/model_manifest.json'),
            'scope':'Frozen Qwen 9B hybrid agent; not the trained 1.5B pilot. Same original predictions reused in every arm.',
            'limitations':['40 uniform random unseen questions; businesses may overlap development.',
                          'One seed, same model as solver and reviewer, one repair at most; no independent-model verification.',
                          'Code gating can abstain on true unanswerable cases it cannot certify.',
                          'Code checks cover selected contracts; no general SQL semantics or ingestion proof.',
                          'Review repairs may reuse the same mistaken data mapping as the baseline.',
                          'Model/data caches warm; reported QA excludes model loading and fresh ingestion.',
                          'Initial interrupted packaging attempt retained separately; one preliminary review is outside the completed run timings.',
                          'No SFT/GRPO training in this experiment, no Colab CU; no official private/Docker six-hour test.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','baseline_source_sha256','input_sha256','model_identity','limitations']},indent=2))


if __name__=='__main__':main()
