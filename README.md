# 🎓 AssistIA - API de Tecnologias Assistivas para TEA (com RAG)

Sistema especializado em recomendar tecnologias assistivas práticas e de baixa tecnologia para alunos com Transtorno do Espectro Autista (TEA). O projeto utiliza **Arquitetura RAG (Retrieval-Augmented Generation)** com IA local, desenvolvido como Trabalho de Conclusão de Curso (TCC).

---

## 📋 Sobre o Projeto

Esta API atua como uma especialista em Educação Inclusiva. Através de um pipeline RAG, a IA analisa o caso do aluno enviado pelo professor e consulta diretamente a base de conhecimento interna (`dados/`) contendo PDFs explicativos e planilhas estruturadas (`planilha_tea.csv`).

### 🎯 Funcionalidades

- **RAG (Busca Semântica + IA Especialista)**: Respostas baseadas rigorosamente nos PDFs e planilhas pedagógicas do projeto
- **Análise de Casos TEA**: Classifica o perfil do aluno e gera recomendações personalizadas
- **Prompt Dinâmico**: Montagem inteligente do prompt, injetando apenas regras relevantes para cada aluno
- **Geração por Área**: Uma chamada ao modelo por área de atenção, garantindo precisão e evitando erros de contagem
- **Catálogo de Tecnologias Assistivas**: Soluções práticas sem custo elevado
- **Taxonomia TAS**: Identificação baseada nas categorias (Comunicação, Motor, Atenção, Comportamentos, Regulação Sensorial, Interação Social, Estruturação)
- **Embeddings em Cache**: Geração acelerada de vetores com cache persistente (`embeddings_cache.pkl`)

### 🚫 O que NÃO faz

- **NÃO** recomenda aplicativos, tablets, celulares ou tecnologias digitais
- **NÃO** substitui parecer de equipe multidisciplinar/médica
- **NÃO** faz diagnósticos de saúde

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Descrição |
|------------|-----------|
| **FastAPI** | Framework web moderno de alta performance |
| **LangChain** | Orquestração do pipeline RAG e prompts dinâmicos |
| **ChromaDB** | Banco vetorial para busca semântica |
| **Ollama** | Execução local de modelos de IA (LLM e Embeddings) |
| **llama3.2:1b** | Modelo de linguagem para geração de atividades (versão 1B) |
| **nomic-embed-text** | Modelo de embeddings para busca semântica |
| **scikit-learn** | Similaridade semântica para o RAG |
| **Docker & Docker Compose** | Containerização completa da aplicação |
| **Fly.io** | Plataforma Cloud PaaS para deploy em produção |

### Modelos Utilizados

```bash
# Modelo para embeddings (busca semântica)
ollama pull nomic-embed-text

# Modelo para geração de atividades (LLM)
ollama pull llama3.2:1b
```

---

## 📁 Estrutura do Projeto

```
assistIA_api_tcc/
├── api/                              # Código-fonte da aplicação
│   ├── __init__.py
│   ├── criar_bd.py                   # Criação e gerenciamento do banco vetorial
│   ├── main.py                       # Rotas e endpoints FastAPI
│   ├── prompt_assistia.py            # Montagem dinâmica do prompt
│   ├── db/                           # Banco vetorial persistente (ChromaDB)
│   └── testar_busca.py               # Testes de busca semântica
├── dados/                            # Base de conhecimento do RAG
│   ├── planilha_tea.csv              # Tabela com mapeamentos da Taxonomia TAS
│   └── *.pdf                         # Documentos pedagógicos e artigos de apoio
├── models/                           # Modelos de IA locais
│   └── ollama/                       # Modelos baixados pelo Ollama
├── .dockerignore                     # Arquivos ignorados no Docker
├── .gitignore                        # Arquivos ignorados no Git
├── docker-compose.yml                # Orquestração do container local
├── Dockerfile                        # Configuração da imagem (Python 3.10-slim)
├── docker-entrypoint.sh              # Script de inicialização do container
├── fly.toml                          # Configuração de deploy no Fly.io
├── README.md                         # Documentação do projeto
├── requirements.txt                  # Dependências Python
└── runtime.txt                       # Definição do ambiente Python
```

---

## 🧠 Arquitetura do Sistema

### Fluxo de Funcionamento

```
┌─────────────────────────────────────────────────────────────────────┐
│                           PROFESSOR                                 │
│                    (envia dados do aluno)                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API (main.py)                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. Recebe dados do aluno (descrição, nível, áreas, etc)    │    │
│  │  2. Busca base de conhecimento (RAG) no banco vetorial      │    │
│  │  3. Para cada área selecionada:                             │    │
│  │     └─► Monta prompt dinâmico (prompt_assistia.py)          │    │
│  │     └─► Chama o modelo (Ollama/llama3.2:1b)                 │    │
│  │     └─► Gera UMA atividade específica                       │    │
│  │  4. Concatena todas as atividades geradas                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │   Área 1      │ │   Área 2      │ │   Área N      │
        │   Atividade   │ │   Atividade   │ │   Atividade   │
        └───────────────┘ └───────────────┘ └───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    RESPOSTA FINAL             │
                    │    (Atividades adaptadas)     │
                    └───────────────────────────────┘
```

### Pipeline RAG (Recuperação de Conhecimento)

```
┌─────────────────────────────────────────────────────────────────────┐
│                       BASE DE CONHECIMENTO                          │
│                    (dados/planilha_tea.csv + PDFs)                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CRIAÇÃO DO BANCO VETORIAL                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. Carrega PDFs e CSVs (criar_bd.py)                       │    │
│  │  2. Divide em chunks (RecursiveCharacterTextSplitter)       │    │
│  │  3. Gera embeddings (nomic-embed-text)                      │    │
│  │  4. Armazena no ChromaDB (api/db/)                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          BUSCA SEMÂNTICA                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. Recebe pergunta do professor                            │    │
│  │  2. Gera embedding da pergunta                              │    │
│  │  3. Busca chunks mais relevantes (similaridade)             │    │
│  │  4. Retorna documentos filtrados por relevância             │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GERAÇÃO DE ATIVIDADES                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. Monta prompt dinâmico com regras relevantes             │    │
│  │  2. Inclui base de conhecimento recuperada                  │    │
│  │  3. Chama LLM (llama3.2:1b) para gerar atividade            │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Como Executar Localmente

### Pré-requisitos

- **Docker** e **Docker Compose** instalados
- Ou **Python 3.10+** com ambiente virtual
- **Ollama** instalado localmente (para execução sem Docker)

### 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/assistIA_api_tcc.git
cd assistIA_api_tcc
```

### 2. Usando Docker Compose (Recomendado)

```bash
# Construir e iniciar o container
docker compose up --build

# A API estará disponível em: http://localhost:8002
```

### 3. Execução com Ollama Local (Sem Docker)

```bash
# Ativar o ambiente virtual
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Baixar os modelos necessários
ollama pull nomic-embed-text
ollama pull llama3.2:1b

# Instalar dependências
pip install -r requirements.txt

# Criar o banco vetorial (primeira execução)
python -m api.criar_bd

# Executar a API
cd api
python main.py --api
```

### 4. Testar a API

```bash
curl -X POST http://localhost:8002/gerar-atividades \
  -H "Content-Type: application/json" \
  -d '{
    "descricao_aluno": "idade: 5 anos, comunicacao: nao verbal",
    "nivel_dsm5": "2",
    "pergunta": "Quais atividades posso ajudar?",
    "areas": ["comunicacao", "social"],
    "area_principal": "interacao_social",
    "prioridade": "media",
    "interesses": ["massinha"],
    "sensibilidades": ["sons_altos"],
    "recursos": ["reciclaveis"]
  }'
```

---

## 📚 Endpoints da API

### Documentação Interativa

- **Swagger UI**: `https://assistia-api-tcc.fly.dev/docs`
- **ReDoc**: `https://assistia-api-tcc.fly.dev/redoc`

### Principal Endpoint: Analisar Aluno com TEA

**POST** `/gerar-atividades`

#### Corpo da Requisição

```json
{
  "descricao_aluno": "Idade: 5 anos, Comunicação: não verbal, Dificuldades: isolamento",
  "nivel_dsm5": "2",
  "pergunta": "Quais atividades posso ajudar esse aluno?",
  "areas": ["comunicacao", "social", "estrutura"],
  "area_principal": "interacao_social",
  "prioridade": "alta",
  "interesses": ["massinha", "desenhos"],
  "sensibilidades": ["sons_altos", "toque"],
  "recursos": ["reciclaveis", "impressora"]
}
```

#### Campos

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `descricao_aluno` | string | Descrição detalhada do aluno |
| `nivel_dsm5` | string | Nível de suporte: "1", "2" ou "3" |
| `pergunta` | string | Pergunta do professor |
| `areas` | array | Áreas de atenção: comunicacao, sensorial, motor, cognitivo, social, estrutura |
| `area_principal` | string | Área prioritária (opcional) |
| `prioridade` | string | "alta", "media" ou "baixa" |
| `interesses` | array | Interesses do aluno |
| `sensibilidades` | array | Sensibilidades sensoriais |
| `recursos` | array | Recursos disponíveis na escola |

#### Resposta

```json
{
  "atividades": "NOME DA ATIVIDADE: ...",
  "tamanho_prompt": 1101,
  "sucesso": true,
  "mensagem": "1 atividade(s) geradas com sucesso"
}
```

---

## ⚠️ Desafios Técnicos e Soluções

### 📌 1. Escolha do Modelo de IA

**Problema**: Inicialmente, foram testados vários modelos para encontrar o equilíbrio ideal entre qualidade e desempenho.

| Modelo | Tamanho | RAM Necessária | Resultado |
|--------|---------|---------------|-----------|
| `smollm2:135m` | 270 MB | ~300 MB | ❌ Gerava loops infinitos e respostas sem sentido |
| `smollm2:360m` | 725 MB | ~500 MB | ❌ Repetia texto em loop, não seguia instruções |
| `phi3:mini` | 2.2 GB | ~3.7 GB | ✅ Boa qualidade, mas estourava memória no Fly.io (OOM) |
| `llama3.2:latest` | 2.0 GB | ~3.5 GB | ⚠️ Qualidade boa, mas próxima do limite de memória |
| **`llama3.2:1b`** | **~1.3 GB** | **~1.2 GB** | ✅ **Equilíbrio perfeito: qualidade + memória** |

**Solução**: Após testes exaustivos, optou-se pelo `llama3.2:1b`, que oferece qualidade satisfatória e cabe nos 2GB de RAM do Fly.io.

### 📌 2. Otimização do Prompt

**Problema**: O prompt inicial tinha ~9500 caracteres e o modelo não conseguia seguir todas as regras simultaneamente, gerando:
- Contagem errada de atividades
- Áreas trocadas
- Interesse do aluno ignorado
- COMO APLICAR repetido

**Solução**:
- Redução do prompt para ~700 caracteres
- Remoção de exemplos longos
- Uso de comandos diretos ("Crie 1 atividade...")
- Estrutura de dados em tópicos

**Antes** (~9500 caracteres):
```
Você é um especialista em Tecnologia Assistiva...
[longas instruções e exemplos]
```

**Depois** (~700 caracteres):
```
Crie 1 atividade para aluno com TEA.
ALUNO: ...
ÁREA: ...
INTERESSE: ...
Use este formato:
NOME DA ATIVIDADE: ...
```

### 📌 3. Estratégia de Geração por Área

**Problema**: Pedir múltiplas atividades em uma única chamada sobrecarregava o modelo e causava erros.

**Solução**: Implementação de **uma chamada por área**:
- O Python controla o loop
- Cada chamada é independente
- Resultados são concatenados

### 📌 4. Configuração do `docker-entrypoint.sh`

**Problema**: Inicializar o Ollama e baixar modelos toda vez que o container iniciava.

**Solução**: Script `docker-entrypoint.sh` que:
1. Inicia o Ollama em background
2. Baixa os modelos necessários
3. Inicia a API

```bash
#!/bin/bash
ollama serve &
sleep 10
ollama pull nomic-embed-text
ollama pull llama3.2:1b
python -m api.main --api
```

### 📌 5. Configuração do Fly.io e Memória

**Problema**: O `phi3:mini` consumia ~3.7GB de RAM, causando `Out of Memory (OOM)` no Fly.io.

**Solução**:
- Troca para `llama3.2:1b` (~1.2GB)
- Configuração do `fly.toml` com 2GB de RAM

```toml
[[vm]]
  memory = "2gb"
  cpu_kind = "shared"
  cpus = 2
```

---

## 📝 Histórico de Atualizações

| Componente | Antes | Agora |
|------------|-------|-------|
| **Modelo de IA** | phi3:mini / llama3.2 | **llama3.2:1b** |
| **Tamanho do Prompt** | ~9500 caracteres | **~700 caracteres** |
| **Geração de Atividades** | Uma chamada para várias áreas | **Uma chamada por área** |
| **Estratégia de Deploy** | Render Free | **Fly.io** |
| **Inicialização** | Manual | **docker-entrypoint.sh automatizado** |
| **Memória Fly.io** | 2GB (insuficiente para phi3) | **2GB (suficiente para llama3.2:1b)** |
| **Base de Dados** | Nenhuma | **ChromaDB com PDFs + CSVs** |
| **Tamanho da Imagem** | ~855 MB | **~244 MB (compactado)** |

---

## 🔧 Comandos Úteis

### Gerenciamento do Banco Vetorial

```bash
# Criar banco do zero (todos os documentos)
python -m api.criar_bd

# Adicionar apenas novos CSVs ao banco existente
python -m api.criar_bd --adicionar

# API no servidor (modo API)
cd ~/assistIA_api_tcc/api
python main.py --api

# Modo CLI (teste interativo)
cd ~/assistIA_api_tcc/api
python main.py
```

### Deploy no Fly.io

```bash
# Build local para verificação
docker compose build --no-cache

# Deploy para produção
fly deploy

# Verificar logs
fly logs -t

# Acessar o container
fly ssh console
```

### Testes

```bash
# Testar busca semântica
python -m api.testar_busca

# Executar API localmente
cd ~/assistIA_api_tcc/api
python main.py --api

# Health check da API
curl https://assistia-api-tcc.fly.dev/health
```

---

## 👨‍🎓 Autores

- **Laura** - *Desenvolvimento e Pesquisa* - [GitHub](https://github.com/LauraUrba)

Este projeto foi desenvolvido para fins acadêmicos como Trabalho de Conclusão de Curso (TCC).

---

**Desenvolvido para o Trabalho de Conclusão de Curso (TCC)** 🎓
