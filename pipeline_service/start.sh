#!/bin/bash
set -e

# --- Configuration (overridable via env vars) ---
VLLM_PORT=${VLLM_PORT:-8095}
VLLM_MODEL=${VLLM_MODEL:-"THUDM/GLM-4.1V-9B-Thinking"}
VLLM_REVISION=${VLLM_REVISION:-"17193d2147da3acd0da358eb251ef862b47e7545"}
GPU_UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.20}
API_KEY=${VLLM_API_KEY:-"local"}

echo "-----------------------------------------------------"
echo "STARTING VLLM SERVER (Isolated Env)"
echo "   Model: $VLLM_MODEL"
echo "   Port: $VLLM_PORT"
echo "   GPU Util: $GPU_UTIL"
echo "-----------------------------------------------------"

# 1. Start vLLM in background
/opt/vllm-env/bin/vllm serve "$VLLM_MODEL" \
    --revision "$VLLM_REVISION" \
    --port "$VLLM_PORT" \
    --api-key "$API_KEY" \
    --max-model-len 8096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_UTIL \
    --max_num_seqs 2 &

VLLM_PID=$!

# 2. Wait for vLLM to be ready (health check)
echo "Waiting for vLLM to become ready..."
MAX_RETRIES=150
COUNTER=0

while [ $COUNTER -lt $MAX_RETRIES ]; do
    if curl -s -f "http://localhost:$VLLM_PORT/health" > /dev/null; then
        echo "vLLM is READY!"
        break
    fi

    echo "   ... loading model ($COUNTER/$MAX_RETRIES)"
    sleep 5
    let COUNTER=COUNTER+1
done

if [ $COUNTER -eq $MAX_RETRIES ]; then
    echo "vLLM failed to start within timeout."
    kill $VLLM_PID
    exit 1
fi

echo "-----------------------------------------------------"
echo "STARTING MAIN FASTAPI SERVICE (Base Env)"
echo "-----------------------------------------------------"
# 3. Start the main pipeline service in background
python serve.py &
SERVE_PID=$!

# 4. No auto requests; rely on pipeline warm-up.
echo "Warm-up mode: skipping automatic FastAPI /health polling and generate requests"

# 5. Keep running — wait on the serve process
wait $SERVE_PID
