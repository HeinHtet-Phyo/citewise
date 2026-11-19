FROM python:3.11-slim

WORKDIR /app

# Install OS-level build dependencies for lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec and install
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>&1 | tail -3 || pip install --no-cache-dir -e .

# Copy source
COPY src/ src/
COPY app.py ./

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# HF Spaces uses port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "citewise.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
