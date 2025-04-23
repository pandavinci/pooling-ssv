# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

#RUN apt install python-is-python3

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app

# Default command (can be overridden)
#CMD ["python", "train_and_eval.py", "--local"] 