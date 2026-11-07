"""Run several models through GATE28 on one big GPU, two at a time (one vLLM server per model, separate ports).

Each spec: TAG=SOURCE:THINK where SOURCE is an HF repo id or adapter:<path.safetensors> (merged into Qwen/Qwen3.5-9B
with merge_lora_full, scale alpha/r), THINK is on|off.
  python -m harness3.run_gate_batch --pack /content/pack --out /content/out --copy-to /content/drive/MyDrive/zalo_eval/GATE28 \
      --adapter-config /content/drive/MyDrive/zalo_eval/ft1/adapter_config.json \
      stock_off=Qwen/Qwen3.5-9B:off ft1_s128=adapter:/content/drive/MyDrive/zalo_eval/ft1/s128.safetensors:off ...
"""
import argparse
import queue as _queue
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = 'Qwen/Qwen3.5-9B'


def merged_dir(tag, adapter, cfg):
    from huggingface_hub import snapshot_download
    base = snapshot_download(BASE)
    out = Path('/content/merged') / tag
    if not (out / 'merge_info.json').exists():
        subprocess.run([sys.executable, '-m', 'harness3.merge_lora_full', '--base', base, '--adapter', adapter,
                        '--config', cfg, '--out', str(out)], check=True)
    return str(out)


def run_one(spec, port, a):
    tag, rest = spec.split('=', 1)
    source, think = rest.rsplit(':', 1)
    model = merged_dir(tag, source[len('adapter:'):], a.adapter_config) if source.startswith('adapter:') else source
    log = open(f'/content/vllm_{tag}.log', 'w')
    p = subprocess.Popen(['vllm', 'serve', model, '--port', str(port), '--dtype', 'bfloat16', '--max-model-len', '32768',
                          '--gpu-memory-utilization', str(a.mem), '--max-num-seqs', '28', '--seed', '47',
                          '--enable-prefix-caching', '--served-model-name', model,
                          '--limit-mm-per-prompt', '{"image":0,"video":0}'], stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        for _ in range(360):
            time.sleep(5)
            try:
                if requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=5).ok:
                    break
            except Exception:
                pass
            if p.poll() is not None:
                raise RuntimeError(f'{tag}: vLLM exited; see /content/vllm_{tag}.log')
        else:
            raise RuntimeError(f'{tag}: vLLM did not start')
        cmd = [sys.executable, '-m', 'harness3.raw_eval', '--pack', a.pack, '--model', model, '--tag', tag, '--out', a.out,
               '--base-url', f'http://127.0.0.1:{port}/v1', '--max-tokens', '8192', '--ctx', '32768', '--workers', '28']
        if think == 'on':
            cmd.append('--thinking')
        subprocess.run(cmd, check=True)
    finally:
        p.terminate()
        try:
            p.wait(90)
        except Exception:
            p.kill()
    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
        for f in (f'{tag}.jsonl', f'{tag}.meta.json'):
            if os.path.exists(os.path.join(a.out, f)):
                shutil.copy(os.path.join(a.out, f), a.copy_to)
    if source.startswith('adapter:'):
        shutil.rmtree(Path('/content/merged') / tag, ignore_errors=True)
    print(f'{tag} done in {(time.time() - t0) / 60:.1f} min', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--copy-to', default='')
    ap.add_argument('--adapter-config', default='')
    ap.add_argument('--parallel', type=int, default=2)
    ap.add_argument('--mem', type=float, default=0.44, help='GPU memory fraction per server')
    ap.add_argument('specs', nargs='+')
    a = ap.parse_args()
    # Merges happen serially first (CPU/GPU memory), then servers run `parallel` at a time.
    for spec in a.specs:
        tag, rest = spec.split('=', 1)
        source = rest.rsplit(':', 1)[0]
        if source.startswith('adapter:'):
            merged_dir(tag, source[len('adapter:'):], a.adapter_config)
    ports = _queue.Queue()
    for i in range(a.parallel):
        ports.put(8100 + i)

    def job(spec):
        port = ports.get()
        try:
            run_one(spec, port, a)
        finally:
            ports.put(port)
    with ThreadPoolExecutor(a.parallel) as pool:
        futs = [pool.submit(job, spec) for spec in a.specs]
        for f in futs:
            try:
                f.result()
            except Exception as e:
                print('FAILED:', e, flush=True)


if __name__ == '__main__':
    main()
