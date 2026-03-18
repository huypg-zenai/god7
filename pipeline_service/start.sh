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

# 4. Wait for FastAPI to be ready
echo "Waiting for FastAPI to become ready..."
MAX_RETRIES_API=25000
COUNTER_API=0

while [ $COUNTER_API -lt $MAX_RETRIES_API ]; do
    if curl -s -f "http://localhost:10006/health" > /dev/null; then
        echo "FastAPI is READY!"
        break
    fi
    echo "   ... loading pipeline ($COUNTER_API/$MAX_RETRIES_API)"
    sleep 5
    let COUNTER_API=COUNTER_API+1
done

if [ $COUNTER_API -eq $MAX_RETRIES_API ]; then
    echo "FastAPI failed to start within timeout."
    kill $SERVE_PID $VLLM_PID 2>/dev/null
    exit 1
fi

# 5. Auto-request with l1.png
AUTO_IMAGE="/workspace/l1.png"
AUTO_SEED=${AUTO_SEED:-42}
if [ -f "$AUTO_IMAGE" ]; then
    echo "-----------------------------------------------------"
    echo "SENDING AUTO-REQUEST: $AUTO_IMAGE (seed=$AUTO_SEED)"
    echo "-----------------------------------------------------"
    curl -s -X POST "http://localhost:10006/generate" \
        -F "prompt_image_file=@${AUTO_IMAGE}" \
        -F "seed=${AUTO_SEED}" \
        -o /workspace/auto_output.glb
    echo "Auto-request done. Output: /workspace/auto_output.glb ($(stat -c%s /workspace/auto_output.glb 2>/dev/null || echo 0) bytes)"
else
    echo "No auto-request image found at $AUTO_IMAGE, skipping."
fi

# 6. Keep running — wait on the serve process
wait $SERVE_PID
