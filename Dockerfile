# SKELETON for the final offline submission. Replace the base image with the one
# required by the organisers' Docker guideline (Blackwell: CUDA >= 12.8).
FROM vllm/vllm-openai:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY datamate/ datamate/
COPY docker/entrypoint.sh /app/entrypoint.sh
# Model weights must be baked into the image (no network at run time), e.g.:
# COPY models/qwen-14b-instruct /models/llm

ENV DATAMATE_BACKEND=openai \
    DATAMATE_MODEL=/models/llm \
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    OPENAI_API_KEY=local \
    DATAMATE_TEMPERATURE=0 \
    DATAMATE_SEED=0 \
    HF_HUB_OFFLINE=1

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
