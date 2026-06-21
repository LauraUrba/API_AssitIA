```markdown
# 🎓 API de Tecnologias Assistivas para TEA

Sistema especializado em recomendar tecnologias assistivas práticas e acessíveis para alunos com Transtorno do Espectro Autista (TEA), desenvolvido como Trabalho de Conclusão de Curso (TCC).

## 📋 Sobre o Projeto

Esta API utiliza modelos de linguagem (LLMs) para analisar casos de alunos com TEA e recomendar tecnologias assistivas de **baixa tecnologia** - soluções práticas que utilizam materiais recicláveis e acessíveis, focando no que o professor pode implementar imediatamente em sala de aula.

### 🎯 Funcionalidades

- **Análise de Casos TEA**: Recebe a descrição do professor sobre o aluno e gera recomendações personalizadas
- **Catálogo de Tecnologias Assistivas**: Oferece alternativas práticas com materiais recicláveis
- **Classificação de Necessidades**: Identifica áreas de necessidade baseado na **Taxonomia TAS** (Comunicação, Motor, Atenção, Comportamentos, Regulação Sensorial, Interação Social, Estruturação)
- **Busca Online**: Complementa as recomendações com resultados da internet via DuckDuckGo

### 🚫 O que NÃO faz

- **NÃO** recomenda aplicativos, tablets, computadores ou tecnologia digital
- **NÃO** substitui a avaliação de um especialista
- **NÃO** faz diagnósticos médicos

## 🛠️ Tecnologias Utilizadas

- **FastAPI** - Framework web para a API
- **llama-cpp-python** - Para carregar e rodar o modelo TinyLlama-1.1B em GGUF
- **TinyLlama-1.1B (GGUF)** - Modelo de linguagem leve e eficiente (apenas ~77 MB em execução)
- **Docker** - Containerização para otimização de memória no deploy
- **DuckDuckGo (ddgs)** - Busca online complementar

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/assistIA_api_tcc.git
cd assistIA_api_tcc
```

### 2. Crie e ative o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate      # Windows
```

### 3. Instale as dependências

```bash
pip install "fastapi[standard]" transformers torch accelerate sentence-transformers
```

#### 📌 Nota sobre a instalação do FastAPI

Quando você instala com `pip install "fastapi[standard]"`, ele vem com algumas dependências padrão opcionais, incluindo `fastapi-cloud-cli`, que permite que você implante em Nuvem FastAPI.

- Se você não quiser ter essas dependências opcionais: `pip install fastapi`
- Se quiser instalar as dependências padrão, mas sem o `fastapi-cloud-cli`: `pip install "fastapi[standard-no-fastapi-cloud-cli]"`

### 4. Verifique a instalação

```bash
pip list | grep -E "fastapi|transformers|torch|accelerate"
```

## 🐳 Docker (Novidade - Otimização para Render Free)

O projeto agora utiliza Docker para otimizar o consumo de memória e garantir compatibilidade com o Render Free. A mudança para o modelo TinyLlama-1.1B em formato GGUF reduziu drasticamente o consumo de memória de **736 MB para apenas 77 MB**.

### Construir a imagem

```bash
docker build -t assistia-api .
```

### Executar localmente

```bash
docker run -p 8000:8000 assistia-api
```

### Monitorar memória

```bash
docker stats assistia-api
```

## 🚀 Como Executar

### Desenvolvimento (sem Docker)

```bash
# Ative o ambiente virtual (se ainda não estiver ativo)
source .venv/bin/activate

# Execute a API
fastapi dev main.py
```

A API estará disponível em: `http://localhost:8000`

### Produção (com Docker)

```bash
docker run -d --name assistia-api -p 8000:8000 assistia-api
```

### Documentação Interativa

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📚 Endpoints da API

### 1. Analisar Aluno com TEA

**POST** `/analisar-aluno-tea/`

**Corpo da requisição:**

```json
{
  "descricao_professor": "Aluno com problema de barulho tem crises, e gosta de atividades manuais",
  "idade_aluno": 4,
  "nivel_suporte": "1",
  "interesses_especificos": "massinha",
  "sensibilidades_sensoriais": "auditiva",
  "incluir_estruturas": true,
  "recursos_disponiveis": "caixas, papel, tesoura"
}
```

**Campos da Taxonomia TAS (novos):**

| Campo | Descrição | Valores Sugeridos |
| :--- | :--- | :--- |
| `comunicacao` | Nível de comunicação | "Verbal", "Não verbal", "Limitada", "Mista" |
| `motor` | Aspecto motor | "Motor Fino", "Motor Grosso", "Ambos" |
| `atencao` | Nível de atenção | "Alta", "Média", "Baixa" |
| `comportamentos` | Características comportamentais | "Repetitivos", "Flexibilidade", "Agitado" |

**Resposta:**

```json
{
  "analise": "Análise detalhada do caso com recomendações...",
  "categorias_necessidade": {
    "comunicacao": "Alta",
    "motor": "Baixa",
    "atencao": "Média",
    "comportamentos": "Média",
    "regulacao_sensorial": "Média",
    "interacao_social": "Alta",
    "estruturacao": "Média"
  },
  "estruturas_recomendadas": [
    "prancha_comunicacao",
    "kit_sensorial",
    "rotina_visual",
    "historias_sociais"
  ]
}
```

### 2. Catálogo de Tecnologias Alternativas

**POST** `/catalogo-alternativas-tea/`

**Corpo da requisição:**

```json
{
  "necessidade": "atividades para autista auditiva e gosta de materiais manuais",
  "tecnologia_recusada": "fone de ouvido",
  "faixa_etaria": "criança de 4 anos",
  "categoria_preferida": "regulacao_sensorial"
}
```

### 3. Verificar Saúde da API

**GET** `/saude/`

**Resposta:**

```json
{
  "status": "saudavel",
  "modelo": "TinyLlama-1.1B (GGUF)",
  "dispositivo": "cpu",
  "parametros": "1.1B"
}
```

## 🧪 Exemplo de Uso

### Teste com cURL

```bash
curl -X 'POST' \
  'http://localhost:8000/analisar-aluno-tea/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "descricao_professor": "O aluno não gosta de barulho, pois tem crises, gosta de atividades manuais",
  "idade_aluno": 5,
  "nivel_suporte": "1",
  "interesses_especificos": "massinha",
  "sensibilidades_sensoriais": "auditiva",
  "incluir_estruturas": true
}'
```

## 📁 Estrutura do Projeto

```
assistIA_api_tcc/
├── .venv/                 # Ambiente virtual
├── .dockerignore          # Arquivos ignorados pelo Docker
├── .gitignore             # Arquivos ignorados pelo Git
├── Dockerfile             # Configuração do Docker (com download do modelo)
├── main.py               # Código principal da API
├── README.md             # Documentação do projeto
├── requirements.txt      # Dependências do projeto
├── render.yaml           # Configuração para deploy no Render
├── runtime.txt           # Versão do Python para deploy
└── models/                # Pasta onde o modelo GGUF é baixado
    └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  # Modelo (638 MB)
```

## 🐳 Otimização de Memória para Render Free

O projeto foi otimizado para rodar no Render Free (512 MB de RAM). A mudança mais significativa foi a troca do modelo de IA.

### 🔄 Mudança do Modelo de IA

| Aspecto | Antes | Depois |
| :--- | :--- | :--- |
| **Modelo** | SmolLM2-135M (PyTorch) | **TinyLlama-1.1B (GGUF)** |
| **Formato** | `.safetensors` | **`.gguf` (quantizado Q4_K_M)** |
| **Biblioteca** | `transformers` + `torch` | **`llama-cpp-python`** |
| **Memória em execução** | ~736 MB | **~77 MB** |
| **Cabe no Render Free?** | ❌ Não | ✅ **Sim** |

### 📊 Otimizações Aplicadas

| Etapa | Otimização | Resultado |
| :--- | :--- | :--- |
| **Modelo** | PyTorch (SmolLM2) → TinyLlama GGUF | 736 MB → **77 MB** |
| **Contexto** | `max_length=512` → `max_length=256` | Redução de ~30% |
| **Cache** | `use_cache=False` | Economia de memória |
| **Threads** | `OMP_NUM_THREADS=1` | Redução de overhead |
| **Quantização** | GGUF Q4_K_M | Modelo otimizado para CPU |

### 📊 Consumo de Memória Atual:

| Componente | Memória |
| :--- | :--- |
| **TinyLlama-1.1B (GGUF)** | ~77 MB |
| **FastAPI + Uvicorn** | ~50 MB |
| **Total** | **~127 MB** |

✅ **Cabe perfeitamente no Render Free (512 MB)!**

## 🏗️ Taxonomia TAS (Tecnologia Assistiva)

A API foi atualizada para seguir a Taxonomia TAS desenvolvida para categorizar:

### 📊 Categorias do Perfil do Estudante

| Categoria | Descrição | Níveis |
| :--- | :--- | :--- |
| **Comunicação** | Verbal / Não verbal / Limitada / Mista | Alta, Média, Baixa |
| **Motor** | Motor Fino / Motor Grosso / Ambos | Alta, Média, Baixa |
| **Atenção** | Capacidade de concentração | Alta, Média, Baixa |
| **Comportamentos** | Repetitivos / Flexibilidade / Agitado | Alta, Média, Baixa |
| **Regulação Sensorial** | Auditiva, Tátil, Visual, Olfativa | Alta, Média, Baixa |
| **Interação Social** | Relação com colegas | Alta, Média, Baixa |
| **Estruturação** | Necessidade de rotina | Alta, Média, Baixa |

### 📌 Tecnologias Assistivas Recomendadas

| Categoria | Recursos |
| :--- | :--- |
| **Comunicação** | Prancha PECS, Cartões com Tampinhas |
| **Sensorial** | Garrafa da Calma, Kit Sensorial, Fone de Ouvido Caseiro |
| **Estruturação** | Rotina Visual com Caixas |
| **Social** | História Social Ilustrada |
| **Motor** | Teclado Adaptado com Papelão |
| **Cognitivo** | Agenda Visual de Tarefas |

## ⚠️ Limitações Conhecidas

1. **Modelo pequeno**: O TinyLlama-1.1B é um modelo pequeno e pode não gerar respostas tão detalhadas quanto modelos maiores
2. **Contexto limitado**: O modelo tem limite de 256 tokens (configurado para economizar memória)
3. **Sem tecnologia digital**: A API NÃO recomenda apps, tablets ou tecnologia digital
4. **CPU apenas**: O modelo roda em CPU, o que pode ser mais lento que GPU
5. **Busca online**: Limitada a 3 resultados por requisição

## 🌐 Deploy

### Deploy no Render (Gratuito)

1. **Faça o push do código para o GitHub**
2. **Conecte o repositório ao Render**
3. **Selecione a opção "Docker"** como ambiente
4. **Instance Type: Free**
5. **Clique em "Deploy"**

**URL do projeto:** `https://api-assitia.onrender.com`

### Deploy Local

```bash
# Com Docker
docker build -t assistia-api .
docker run -p 8000:8000 assistia-api

# Sem Docker
fastapi dev main.py
```

## 📚 Referências

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [TinyLlama-1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Render Web Services](https://render.com/docs/web-services)

## 📝 Licença

Este projeto é para fins acadêmicos (TCC). Todos os direitos reservados.

## 👨‍🎓 Autores

- Laura - [GitHub](https://github.com/LauraUrba)

---

**Desenvolvido para o Trabalho de Conclusão de Curso (TCC)** 🎓
```

---

## ✅ **Resumo das adições ao README:**

| Seção | O que foi adicionado |
| :--- | :--- |
| **Tecnologias** | `llama-cpp-python`, TinyLlama GGUF, Docker |
| **Docker** | Seção completa sobre otimização e uso |
| **Mudança do Modelo** | Tabela comparativa Antes x Depois |
| **Otimização de Memória** | Tabela com as otimizações aplicadas |
| **Taxonomia TAS** | Documentação completa dos campos |
| **Estrutura** | Adicionado `models/` e `Dockerfile` |
| **Deploy** | Instruções atualizadas para Docker no Render |
