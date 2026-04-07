FROM python:3.11-slim

WORKDIR /app

# System deps (kaleido needs chromium libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
    libcairo2 libx11-6 libx11-xcb1 libxcb1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user (security best practice)
RUN useradd -m -u 1000 labuser
USER labuser

EXPOSE 7860

# 4 workers handles 5+ concurrent users comfortably
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", \
     "--timeout", "120", "--worker-class", "sync", \
     "--worker-tmp-dir", "/tmp", "app:server"]
