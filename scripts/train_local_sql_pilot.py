"""Local LoRA SFT -> Dr. GRPO, with SQL-execution rewards and sealed evaluation."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['WANDB_DISABLED']='true'
os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from datamate_bench.scoring import score,summarize
from datamate_bench.suite import read_jsonl,digest
from datamate_bench.__main__ import source_fingerprints
from local_training.planner import run_action,reward


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/local_sft_grpo/dataset.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/local_sft_grpo/run1')
    parser.add_argument('--sft-steps',type=int,default=64)
    parser.add_argument('--grpo-steps',type=int,default=24)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('Completed experiment exists; preserve results')
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed,TrainerCallback
    from peft import LoraConfig,PeftModel,get_peft_model
    from trl import SFTTrainer,SFTConfig,GRPOTrainer,GRPOConfig
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU training fallback')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('high');set_seed(20261103)
    hub=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct'
    revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
    model_dir=hub/'snapshots'/revision
    if not (model_dir/'model.safetensors').exists():raise ValueError('Pinned base weights not cached locally')
    print('Loading cached Qwen2.5-1.5B-Instruct on CUDA',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='left'
    def load_model():
        return AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,
                      dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    model=load_model()
    config={'model':'Qwen/Qwen2.5-1.5B-Instruct','revision':revision,'device':torch.cuda.get_device_name(0),
            'base_model_sha256':digest(model_dir/'model.safetensors'),'dtype':'bfloat16',
            'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','trl','peft','datasets','accelerate']},
            'seed':20261103,'sft_steps':args.sft_steps,'grpo_steps':args.grpo_steps,
            'sft_lr':1e-4,'grpo_lr':5e-6,'lora_rank':8,'lora_targets':['q_proj','v_proj'],
            'max_total_tokens':2048,'max_completion_tokens':128,'num_generations':2,
            'loss_type':'dr_grpo','scale_rewards':'none','beta':0,
            'data_sha256':digest(args.data),'solver_source_sha256':source_fingerprints(),
            'training_source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),*sorted((ROOT/'local_training').glob('*.py'))]},
            'scope':'Single-step metric planner with SQL executor. Not the multi-turn Qwen 9B benchmark.',
            'external_training':False,'report_to':'none'}
    existing=args.output/'config.json'
    if existing.exists():
        old=json.loads(existing.read_text(encoding='utf-8'))
        if any(old[k]!=config[k] for k in ['model','revision','data_sha256','sft_steps','grpo_steps']):raise ValueError('Resume configuration changed')
        write(args.output/f'config_previous_{int(time.time())}.json',old)
    write(existing,config)
    bundle=json.loads(args.data.read_text(encoding='utf-8'))
    rows=bundle['evaluation'];training=bundle['training'];businesses=bundle['businesses']
    lengths=[len(tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)) for r in training+rows]
    print('Prompt tokens',min(lengths),max(lengths),'GPU allocated GiB',torch.cuda.memory_allocated()/2**30,flush=True)
    if max(lengths)+128>2048:raise ValueError('Pilot prompts exceed memory budget; do not silently truncate questions')
    if args.probe_only:
        inputs=tokenizer.apply_chat_template(rows[0]['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
        with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=16,do_sample=False,pad_token_id=tokenizer.pad_token_id)
        print('CUDA probe generated',output.shape[-1]-inputs['input_ids'].shape[-1],'tokens; peak GiB',torch.cuda.max_memory_allocated()/2**30,flush=True)
        return
    def evaluate(name):
        path=args.output/f'{name}_predictions.json'
        if path.exists():return json.loads(path.read_text(encoding='utf-8'))
        model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
        predictions=[];started=time.perf_counter()
        for i,row in enumerate(rows,1):
            start=time.perf_counter()
            inputs=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id,use_cache=True)
            text=tokenizer.decode(output[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
            q=row['question'];result=run_action(text,q,businesses[q['business_id']])
            predictions.append({'id':q['id'],'question':q['question'],'raw_response':text,
                                'result':result,'seconds':time.perf_counter()-start})
            write(args.output/f'{name}_partial.json',predictions)
            print(f'{name} inference {i}/{len(rows)}: {result["prediction"]["type"]}',flush=True)
        write(path,predictions)
        write(args.output/f'{name}_timing.json',{'wall_seconds':time.perf_counter()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        return predictions
    class Progress(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            print('TRAIN',state.global_step,json.dumps(logs or {},ensure_ascii=False,default=str),flush=True)
    all_predictions={'base':evaluate('base')}
    sft_path=args.output/'sft_adapter'
    if (sft_path/'adapter_model.safetensors').exists():
        model=PeftModel.from_pretrained(model,sft_path,is_trainable=True)
    else:
        peft=LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
        model=get_peft_model(model,peft);model.print_trainable_parameters();model.config.use_cache=False
        dataset=Dataset.from_list([{'prompt':r['messages'],'completion':[{'role':'assistant','content':r['completion']}]} for r in training])
        sft_args=SFTConfig(output_dir=str(args.output/'sft_training'),max_steps=args.sft_steps,
                   per_device_train_batch_size=1,gradient_accumulation_steps=2,learning_rate=1e-4,
                   warmup_steps=4,bf16=True,optim='adamw_torch',max_length=2048,packing=False,
                   completion_only_loss=True,gradient_checkpointing=True,
                   gradient_checkpointing_kwargs={'use_reentrant':False},logging_steps=1,disable_tqdm=True,
                   report_to='none',save_strategy='no',seed=20261103,dataloader_num_workers=0)
        start=time.perf_counter();trainer=SFTTrainer(model=model,args=sft_args,train_dataset=dataset,
                                                processing_class=tokenizer,callbacks=[Progress()])
        trained=trainer.train();trainer.save_model(sft_path);tokenizer.save_pretrained(sft_path)
        write(args.output/'sft_training_summary.json',{'metrics':trained.metrics,'history':trainer.state.log_history,
                'wall_seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
        del trainer;gc.collect();torch.cuda.empty_cache()
    all_predictions['sft']=evaluate('sft')
    grpo_path=args.output/'grpo_adapter'
    if (grpo_path/'adapter_model.safetensors').exists():
        del model;gc.collect();torch.cuda.empty_cache();model=PeftModel.from_pretrained(load_model(),grpo_path)
    else:
        # Only training questions/references are closed over by the reward function.
        lookup={r['question']['id']:r for r in training}
        def sql_reward(completions,case_id,**kwargs):
            values=[]
            for completion,qid in zip(completions,case_id):
                text=completion[0]['content'] if isinstance(completion,list) else completion
                row=lookup[qid];q=row['question']
                value,result=reward(text,q,businesses[q['business_id']],row['reference'])
                values.append(float(value))
                with (args.output/'grpo_rewards.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'id':qid,'completion':text,'reward':value,'result':result},ensure_ascii=False)+'\n')
            return values
        dataset=Dataset.from_list([{'prompt':r['messages'],'case_id':r['question']['id']} for r in training])
        model.config.use_cache=False
        grpo_args=GRPOConfig(output_dir=str(args.output/'grpo_training'),max_steps=args.grpo_steps,
                   per_device_train_batch_size=1,gradient_accumulation_steps=2,generation_batch_size=2,
                   num_generations=2,max_completion_length=128,temperature=.9,top_p=.95,
                   learning_rate=5e-6,bf16=True,optim='adamw_torch',loss_type='dr_grpo',scale_rewards='none',beta=0,
                   gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                   logging_steps=1,disable_tqdm=True,report_to='none',save_strategy='no',seed=20261103,
                   dataloader_num_workers=0,use_vllm=False,log_completions=False,mask_truncated_completions=True)
        start=time.perf_counter();trainer=GRPOTrainer(model=model,args=grpo_args,reward_funcs=sql_reward,
                    train_dataset=dataset,processing_class=tokenizer,callbacks=[Progress()])
        trained=trainer.train();trainer.save_model(grpo_path);tokenizer.save_pretrained(grpo_path)
        write(args.output/'grpo_training_summary.json',{'metrics':trained.metrics,'history':trainer.state.log_history,
                'wall_seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'adapter_changed_from_sft':digest(grpo_path/'adapter_model.safetensors')!=digest(sft_path/'adapter_model.safetensors')})
        del trainer;gc.collect();torch.cuda.empty_cache()
    all_predictions['sft_grpo']=evaluate('sft_grpo')
    # Read evaluation references only once all three inference arms have finished.
    ids={r['question']['id'] for r in rows}
    references={r['id']:r for r in read_jsonl(ROOT.parent/'train/answers.jsonl') if r['id'] in ids}
    summary={'model':config['model'],'revision':revision,'scope':config['scope'],'evaluation_questions':len(rows),
             'train_examples':len(training),'development_examples':len(bundle['development']),
             'colab_compute_units':0,'arms':{},'config':config}
    for name,predictions in all_predictions.items():
        for row in predictions:
            row['reference']=references[row['id']];row['score']=score(row['result']['prediction'],row['reference'])
        summary['arms'][name]={**summarize([r['score'] for r in predictions]),
                              'invalid_actions':sum(bool(r['result'].get('invalid')) for r in predictions),
                              'inference_seconds':sum(r['seconds'] for r in predictions)}
        write(args.output/f'{name}_scores.json',predictions)
    write(args.output/'summary.json',summary)
    print('FINAL',json.dumps({k:v for k,v in summary.items() if k!='config'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
