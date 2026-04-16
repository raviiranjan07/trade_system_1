FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (includes engine/config/, so no separate config copy needed)
COPY src/ src/

# Copy pre-built frontend
COPY src/web/frontend/dist/ src/web/frontend/dist/

# Copy runtime configs (params.yaml, pipelines.yaml — read at runtime)
COPY configs/ configs/

# Copy ML models — ONNX + scaler only (torch not installed in production image)
# ML_V1 (MLP). Newer torch.onnx exports split weights into a sidecar .onnx.data
# file that onnxruntime expects next to the .onnx — both must be copied.
COPY models/ML_V1/direction_model.onnx models/ML_V1/
COPY models/ML_V1/direction_model.onnx.data models/ML_V1/
COPY models/ML_V1/scaler.npz models/ML_V1/

# ML_V2_ATTENTION (LSTM+Attention). Same pattern — .onnx.data sidecar required.
COPY models/ML_V2_ATTENTION/attention_model.onnx models/ML_V2_ATTENTION/
COPY models/ML_V2_ATTENTION/attention_model.onnx.data models/ML_V2_ATTENTION/
COPY models/ML_V2_ATTENTION/scaler.npz models/ML_V2_ATTENTION/

# Create data directories (persistent volume mounted here)
RUN mkdir -p data/trades/risk_logs/ml data/trades/risk_logs/ml_attn

# Expose dashboard port
EXPOSE 8080

ENV PYTHONPATH=src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "engine.bot"]
