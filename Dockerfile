ARG VLLM_BASE=vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
FROM ${VLLM_BASE}

LABEL org.opencontainers.image.source="https://github.com/cluster2600/qwen38-flash-next-vast"
LABEL org.opencontainers.image.description="Qwen3.8 Flash Next Uncensored NVFP4 on two Blackwell GPUs with vLLM"
LABEL org.opencontainers.image.licenses="MIT"

COPY scripts/patch_ple.py scripts/patch_qwen_config.py /opt/qwen/
RUN python3 /opt/qwen/patch_ple.py \
    && python3 /opt/qwen/patch_qwen_config.py \
    && rm /opt/qwen/patch_ple.py /opt/qwen/patch_qwen_config.py

COPY scripts/start.sh /opt/qwen/start.sh
COPY scripts/start-background.sh /opt/qwen/start-background.sh
COPY scripts/healthcheck.py /opt/qwen/healthcheck.py
COPY scripts/benchmark.py /opt/qwen/benchmark.py

RUN chmod 0755 /opt/qwen/start.sh /opt/qwen/start-background.sh \
    /opt/qwen/healthcheck.py /opt/qwen/benchmark.py

ENV MODEL_ID=orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 \
    SERVED_MODEL_NAME=qwen3.8-flash-next-uncensored-nvfp4 \
    HF_HOME=/workspace/huggingface \
    VLLM_PLE_CPU_OFFLOAD=1 \
    VLLM_PLE_FORCE_FP8=1 \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

WORKDIR /workspace
EXPOSE 8000

ENTRYPOINT ["/opt/qwen/start.sh"]
