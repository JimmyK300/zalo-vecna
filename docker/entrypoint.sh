#!/bin/bash
# Start a local vLLM server on both GPUs, wait for it, then run the agent.
set -euo pipefail
python -m vllm.entrypoints.openai.api_server \
  --model "$DATAMATE_MODEL" --served-model-name "$DATAMATE_MODEL" \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --seed 0 --port 8000 > /result/vllm.log 2>&1 &

until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 5; done
python -m datamate.run --data /data --out /result/submission.jsonl
