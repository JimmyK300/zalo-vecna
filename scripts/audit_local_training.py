"""Audit completed local training; publish aggregates, keep identifiers local."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from datamate_bench.suite import digest
from datamate_bench.scoring import score,summarize
from datamate_bench.metrics import SUPPORTED
from datamate_bench.__main__ import source_fingerprints
from local_training.planner import run_action


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/local_sft_grpo/run1')
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/local_sft_grpo/dataset.json')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/local_sft_grpo_summary.json')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve old report; choose another output')
    summary=read(args.run/'summary.json');bundle=read(args.data);config=summary['config']
    if digest(args.data)!=config['data_sha256']:raise ValueError('Dataset changed')
    changed=[p for p,h in config['training_source_sha256'].items() if digest(ROOT/p)!=h]
    if source_fingerprints()!=config['solver_source_sha256']:changed.append('solver')
    if changed:raise ValueError('Inference/training source changed: '+str(changed))
    questions={r['question']['id']:r['question'] for r in bundle['evaluation']}
    # Secondary diagnostic only: same outputs, normalize direct operation-as-action aliases.
    # This was motivated by observing baseline formatting errors; not a new blind test.
    aliases={};primary={};details={}
    for arm in ['base','sft','sft_grpo']:
        rows=read(args.run/f'{arm}_scores.json');results=[];normalized_count=0
        for row in rows:
            text=row['raw_response']
            try:
                action=json.loads(text)
                if isinstance(action,dict) and action.get('action') in SUPPORTED:
                    action={**action,'operation':action['action'],'action':'metric'}
                    text=json.dumps(action,ensure_ascii=False);normalized_count+=1
            except ValueError:pass
            q=questions[row['id']]
            output=run_action(text,q,bundle['businesses'][q['business_id']])
            value=score(output['prediction'],row['reference'])
            results.append({'id':row['id'],'score':value,'result':output,'reference':row['reference']})
        aliases[arm]={**summarize([r['score'] for r in results]),'aliases_normalized':normalized_count,
                      'invalid_actions':sum(bool(r['result'].get('invalid')) for r in results)}
        primary[arm]={r['id']:r['score'] for r in rows};details[arm]=results
    def paired(left,right):
        return {'gained_correct':sum(left[k]!=1 and right[k]==1 for k in left),
                'lost_correct':sum(left[k]==1 and right[k]!=1 for k in left),
                'score_improved':sum(right[k]>left[k] for k in left),
                'score_regressed':sum(right[k]<left[k] for k in left)}
    sft=read(args.run/'sft_training_summary.json');rl=read(args.run/'grpo_training_summary.json')
    from safetensors.torch import load_file
    a=load_file(args.run/'sft_adapter/adapter_model.safetensors',device='cpu')
    b=load_file(args.run/'grpo_adapter/adapter_model.safetensors',device='cpu')
    deltas={k:(a[k].float()-b[k].float()).square().sum().item() for k in a}
    train_ids={r['question']['id'] for r in bundle['training']};eval_ids=set(questions)
    dev_ids={r['question']['id'] for r in bundle['development']}
    rollout=[json.loads(s) for s in (args.run/'grpo_rewards.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    report={'model':summary['model'],'revision':summary['revision'],'scope':summary['scope'],
            'evaluation_questions':len(questions),'train_examples':len(train_ids),'reserved_development_examples':len(dev_ids),
            'splits_disjoint':not bool(train_ids&eval_ids or dev_ids&eval_ids or train_ids&dev_ids),
            'colab_compute_units':0,'primary_strict_protocol':summary['arms'],
            'secondary_alias_normalized':aliases,
            'primary_sft_to_grpo_paired':paired(primary['sft'],primary['sft_grpo']),
            'secondary_sft_to_grpo_paired':paired({r['id']:r['score'] for r in details['sft']},{r['id']:r['score'] for r in details['sft_grpo']}),
            'sft':{'optimizer_steps':config['sft_steps'],'wall_seconds':sft['wall_seconds'],
                   'training_seconds':sft['metrics']['train_runtime'],'train_loss':sft['metrics']['train_loss']},
            'grpo':{'optimizer_steps':config['grpo_steps'],'wall_seconds':rl['wall_seconds'],
                    'training_seconds':rl['metrics']['train_runtime'],'rollouts':len(rollout),
                    'reward_counts':dict(Counter(r['reward'] for r in rollout)),
                    'steps_with_nonzero_gradient':sum(h.get('grad_norm',0)>0 for h in rl['history']),
                    'steps_with_zero_group_reward_variance':sum(h.get('frac_reward_zero_std')==1 for h in rl['history']),
                    'adapter_changed':any(v>0 for v in deltas.values()),'changed_tensors':sum(v>0 for v in deltas.values()),
                    'adapter_l2_delta':sum(deltas.values())**.5,
                    'loss_type':config['loss_type'],'scale_rewards':config['scale_rewards'],'beta':config['beta'],
                    'num_generations':config['num_generations']},
            'max_process_allocated_gib':max(sft['peak_allocated_gib'],rl['peak_allocated_gib']),
            'config':config,'dataset_summary':bundle['summary'],
            'limitations':['Single-step metric planner, not a full multi-turn SQL-generating agent.',
                          'Small question-held-out pilot; business formats were already encountered.',
                          'Fixed one-epoch SFT and 24-step RL, one seed; no checkpoint selection on evaluation.',
                          'Positive-only verified SFT demonstrations; reserved development examples not used for tuning.',
                          'Strict-protocol baseline fails tool envelopes; do not attribute all gains to semantic reasoning.',
                          'Alias-normalized rescoring is a post-hoc diagnostic on the same completions, not a new blind test.',
                          'LoRA from the 1.5B model cannot be applied to the 9B model; scores are not directly comparable.',
                          'Local scorer; no official private-test or six-hour Docker verification.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.run/'alias_normalized_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['config','dataset_summary','limitations']},indent=2))


if __name__=='__main__':main()
