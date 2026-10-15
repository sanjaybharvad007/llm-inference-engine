# CUDA runtime + PyTorch; builds a self-contained server image.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY pyproject.toml README.md ./
COPY minivllm ./minivllm
RUN pip install --no-cache-dir -e ".[gpu,hf]"

ENV MODEL_PATH=/models/model \
    PORT=8000 \
    MAX_NUM_SEQS=256 \
    GPU_MEMORY_UTILIZATION=0.9

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:${PORT}/health || exit 1
CMD ["sh", "-c", "python -m minivllm serve --model ${MODEL_PATH} --port ${PORT} --max-num-seqs ${MAX_NUM_SEQS} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"]
