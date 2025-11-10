# Use a small, secure Python base
FROM python:3.12-slim

# Prevent Python from writing .pyc and forcing buffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (optional but useful for tzdata/locale)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create and set work dir
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create a mount point for persistent data (Render Disk)
# We'll mount /app/data from render.yaml
RUN mkdir -p /app/data

# Expose runtime port (Render sets $PORT; EXPOSE is just doc)
EXPOSE 8000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD python -c "import socket,os; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', int(os.environ.get('PORT','8000')))); s.close()"

# Start the server
CMD ["python", "server.py"]
