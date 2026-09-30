FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]