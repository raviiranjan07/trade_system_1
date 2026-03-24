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

# Copy ML model (ONNX + scaler only, no torch)
COPY src/v12/ml_model/direction_model.onnx src/v12/ml_model/
COPY src/v12/ml_model/scaler.npz src/v12/ml_model/

# Create data directories
RUN mkdir -p data/v12_trades data/risk_logs/ml

# Expose dashboard port
EXPOSE 8080

ENV PYTHONPATH=src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "v12.bot"]
