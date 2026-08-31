#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next-uncensored-nvfp4}"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required because the model repository is gated." >&2
  exit 64
fi

if (( MAX_NUM_SEQS < 4 || MAX_NUM_SEQS % 4 != 0 )); then
  echo "MAX_NUM_SEQS must be a multiple of 4 and at least 4." >&2
  exit 64
fi

if (( SPECULATIVE_TOKENS < 1 )); then
  echo "SPECULATIVE_TOKENS must be at least 1." >&2
  exit 64
fi

export MODEL_ID SERVED_MODEL_NAME HF_HOME PORT MAX_MODEL_LEN
export CUDA_VISIBLE_DEVICES
export VLLM_PLE_CPU_OFFLOAD="${VLLM_PLE_CPU_OFFLOAD:-1}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

mkdir -p "$HF_HOME"

python3 - <<'PY'
import sys
import torch

if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected exactly two visible GPUs, found {torch.cuda.device_count()}")
for index in range(2):
    major, minor = torch.cuda.get_device_capability(index)
    if major < 12:
        raise SystemExit(
            f"GPU {index} has compute capability {major}.{minor}; NVFP4 requires Blackwell SM120+"
        )
print("validated two Blackwell GPUs")
PY

CAPTURE_SIZES="$(python3 -c "import json; print(json.dumps(list(range(4, ${MAX_NUM_SEQS} + 1, 4))))")"
COMPILATION_CONFIG="$(python3 -c 'import json,sys; sizes=json.loads(sys.argv[1]); print(json.dumps({"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":sizes,"inductor_compile_config":{"triton.autotune_at_compile_time":False}}, separators=(",",":")))' "$CAPTURE_SIZES")"
SPECULATIVE_CONFIG="$(python3 -c 'import json,sys; print(json.dumps({"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":int(sys.argv[1])}, separators=(",",":")))' "$SPECULATIVE_TOKENS")"

echo "Starting ${SERVED_MODEL_NAME} on 127.0.0.1:${PORT} (TP=2, context=${MAX_MODEL_LEN})."
echo "The API is loopback-only; use an SSH tunnel from the client."

exec vllm serve "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --compilation-config "$COMPILATION_CONFIG" \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --host 127.0.0.1 \
  --port "$PORT"
