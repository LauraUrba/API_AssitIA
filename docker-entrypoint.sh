#!/bin/bash
# docker-entrypoint.sh - Inicia Ollama + API

echo "============================================================"
echo "INICIANDO ASSISTIA - API para TEA"
echo "============================================================"

# Configurar PATH para incluir Ollama
export PATH=$PATH:/usr/local/bin

# Adicionar /app ao PYTHONPATH (MUDANÇA: só /app é suficiente)
export PYTHONPATH="/app:${PYTHONPATH}"

# Iniciar Ollama em background
echo "[1/4] Iniciando Ollama..."
ollama serve &
OLLAMA_PID=$!

# Aguardar Ollama iniciar (AUMENTEI PARA 20s)
echo "Aguardando Ollama iniciar..."
sleep 20

# Verificar se Ollama está rodando (COM RETRY)
for i in 1 2 3; do
    if curl -s http://localhost:11434/api/health > /dev/null 2>&1; then
        echo "✅ Ollama está rodando!"
        break
    fi
    if [ $i -eq 3 ]; then
        echo "❌ Ollama não iniciou após 3 tentativas!"
        exit 1
    fi
    echo "⚠️ Ollama não respondeu, tentando novamente (tentativa $i)..."
    sleep 10
done

# Baixar modelos
echo "[2/4] Baixando modelo de embeddings (nomic-embed-text)..."
ollama pull nomic-embed-text

echo "[3/4] Baixando modelo de linguagem (llama3.2)..."
ollama pull llama3.2:latest

# Listar modelos
echo "[4/4] Modelos instalados:"
ollama list

echo ""
echo "============================================================"
echo "✅ API ASSISTIA INICIADA COM SUCESSO!"
echo "============================================================"
echo "Documentação: http://localhost:8003/docs"
echo "Health Check: http://localhost:8003/health"
echo "============================================================"

# Executar a API (MUDANÇA: cd /app antes de executar)
cd /app
exec python -m api.main --api