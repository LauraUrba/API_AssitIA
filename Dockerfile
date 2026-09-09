FROM python:3.12-slim

WORKDIR /app

# ============================================================
# INSTALAR DEPENDÊNCIAS DO SISTEMA
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    gnupg \
    build-essential \
    cmake \
    g++ \
    libgomp1 \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# BAIXAR E INSTALAR OLLAMA
# ============================================================
RUN curl -fsSL https://ollama.com/install.sh | sh

# ============================================================
# INSTALAR DEPENDÊNCIAS PYTHON (EM ETAPAS)
# ============================================================
COPY requirements.txt .

# Etapa 1: Pacotes principais
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic python-dotenv

# Etapa 2: LangChain
RUN pip install --no-cache-dir langchain langchain-core langchain-community

# Etapa 3: Banco Vetorial e Data Science
RUN pip install --no-cache-dir chromadb scikit-learn numpy pandas

# Etapa 4: LangChain extras
RUN pip install --no-cache-dir langchain-chroma langchain-ollama langchain-text-splitters

# ============================================================
# CRIAR DIRETÓRIOS
# ============================================================
RUN mkdir -p api dados models cache

# ============================================================
# COPIAR CÓDIGO
# ============================================================
COPY api/ ./api/
COPY dados/ ./dados/
COPY api/db/ ./db/

# ============================================================
# SCRIPT DE INICIALIZAÇÃO
# ============================================================
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8003

ENTRYPOINT ["/docker-entrypoint.sh"]
