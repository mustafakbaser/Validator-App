# Multi-stage build for Validator App
# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OCR and image processing dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    # Font dependencies for PDF generation
    fonts-dejavu-core \
    fonts-dejavu-extra \
    # Additional system libraries
    libgcc-s1 \
    libstdc++6 \
    libzbar0 \
    libtesseract-dev \
    tesseract-ocr \
    tesseract-ocr-tur \
    tesseract-ocr-eng \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as python-deps

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application build
FROM python-deps as app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/application_forms /app/logs /app/temp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "verify_documents.py", "--help"]

# Stage 4: Production image (optional - for smaller image)
FROM python-deps as production

# Copy only necessary files
COPY verify_documents.py .
COPY generate_application_form.py .
COPY test_custom_tckn.py .

# Create necessary directories
RUN mkdir -p /app/application_forms /app/logs /app/temp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "verify_documents.py", "--help"]
