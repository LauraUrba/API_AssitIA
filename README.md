# Projeto descrito

```markdown
# 🎓 API de Tecnologias Assistivas para TEA

Sistema especializado em recomendar tecnologias assistivas práticas e acessíveis para alunos com Transtorno do Espectro Autista (TEA), desenvolvido como Trabalho de Conclusão de Curso (TCC).

## 📋 Sobre o Projeto

Esta API utiliza modelos de linguagem (LLMs) para analisar casos de alunos com TEA e recomendar tecnologias assistivas de **baixa tecnologia** - soluções práticas que utilizam materiais recicláveis e acessíveis, focando no que o professor pode implementar imediatamente em sala de aula.

### 🎯 Funcionalidades

- **Análise de Casos TEA**: Recebe a descrição do professor sobre o aluno e gera recomendações personalizadas
- **Catálogo de Tecnologias Assistivas**: Oferece alternativas práticas com materiais recicláveis
- **Classificação de Necessidades**: Identifica áreas de necessidade (comunicação, estruturação, regulação sensorial, interação social)

### 🚫 O que NÃO faz

- **NÃO** recomenda aplicativos, tablets, computadores ou tecnologia digital
- **NÃO** substitui a avaliação de um especialista
- **NÃO** faz diagnósticos médicos

## 🛠️ Tecnologias Utilizadas

- **FastAPI** - Framework web para a API
- **Transformers (Hugging Face)** - Para carregar e rodar o modelo de linguagem
- **PyTorch** - Framework de deep learning
- **Qwen2.5-0.5B-Instruct** - Modelo de linguagem leve e eficiente

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

## 🚀 Como Executar

### Desenvolvimento

```bash
# Ative o ambiente virtual (se ainda não estiver ativo)
source .venv/bin/activate

# Execute a API
fastapi dev main.py
```

A API estará disponível em: `http://localhost:8000`

### Produção

```bash
fastapi run main.py
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
  "modelo": "Qwen/Qwen2.5-0.5B-Instruct",
  "dispositivo": "cpu"
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
├── .gitignore            # Arquivos ignorados pelo Git
├── main.py               # Código principal da API
├── README.md             # Documentação do projeto
├── requirements.txt      # Dependências do projeto
├── render.yaml           # Configuração para deploy no Render
└── runtime.txt           # Versão do Python para deploy
```

## ⚠️ Limitações Conhecidas

1. **Modelo pequeno**: O Qwen2.5-0.5B é um modelo pequeno e pode não gerar respostas tão detalhadas quanto modelos maiores
2. **Contexto limitado**: O modelo tem limite de 2048 tokens
3. **Sem tecnologia digital**: A API NÃO recomenda apps, tablets ou tecnologia digital
4. **CPU apenas**: O modelo roda em CPU, o que pode ser mais lento que GPU

## 🔧 Solução de Problemas

### Erro: O PyCharm fecha durante o carregamento do modelo

**Causa**: Falta de memória RAM.

**Solução**:
1. Use o modelo Qwen2.5-0.5B (já configurado)
2. Feche outros programas para liberar memória
3. Se ainda assim falhar, tente o modelo SmolLM2-1.7B-Instruct (mais leve)

### Erro: "Token indices sequence length is longer than specified maximum"

**Causa**: O prompt é maior que o limite de contexto do modelo.

**Solução**: O código já implementa truncamento automático com `truncation=True` e `max_length=2048`.

### Erro: Não consigo baixar o modelo

**Solução**:
- Verifique sua conexão com a internet
- Tente usar um modelo alternativo:
  ```python
  ID_MODELO = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  ```
## 📚 Referências

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
  
## 📝 Licença

Este projeto é para fins acadêmicos (TCC). Todos os direitos reservados.

## 👨‍🎓 Autores

- Laura




