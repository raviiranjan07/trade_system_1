FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Copy pre-built frontend
COPY src/web/frontend/dist/ src/web/frontend/dist/

# Copy config
COPY src/v12/config/ src/v12/config/

# Copy ML models (ONNX + scaler only, no torch)
COPY models/direction_v15/direction_model.onnx models/direction_v15/
COPY models/direction_v15/scaler.npz models/direction_v15/
COPY models/direction_attention/attention_model.onnx models/direction_attention/
COPY models/direction_attention/attention_model.onnx.data models/direction_attention/
COPY models/direction_attention/scaler.npz models/direction_attention/

# Create data directories (persistent volume mounted here)
RUN mkdir -p data/trades/risk_logs/ml data/trades/risk_logs/ml_attn

# Expose dashboard port
EXPOSE 8080

ENV PYTHONPATH=src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "v12.bot"]
