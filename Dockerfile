FROM python:3.11-slim

WORKDIR /app

# INSTALA DEPENDÊNCIAS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    gcc \
    g++ \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY . .

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]