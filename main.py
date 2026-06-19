from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import gc
import re
import os

# ============================================
# CONFIGURAÇÃO INICIAL DA API
# ============================================

app = FastAPI(
    title="API de Tecnologias Assistivas para TEA - TCC",
    description="Sistema especializado em recomendar tecnologias assistivas para alunos com TEA",
    version="2.0.0"
)

print("🚀 Carregando modelo SmolLM2-135M (modo ultra-leve)...")

ID_MODELO = "HuggingFaceTB/SmolLM2-135M-Instruct"

# 🔥 OTIMIZAÇÕES PARA REDUZIR MEMÓRIA
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

tokenizador = AutoTokenizer.from_pretrained(ID_MODELO)

if tokenizador.pad_token is None:
    tokenizador.pad_token = tokenizador.eos_token

# 🔥 SEM QUANTIZAÇÃO - APENAS COM low_cpu_mem_usage
modelo = AutoModelForCausalLM.from_pretrained(
    ID_MODELO,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_cache=False,  # Desativa cache para economizar memória
)

print("✅ Modelo SmolLM2-135M carregado com sucesso!")
print(f"💻 Dispositivo em uso: {modelo.device}")

# ============================================
# CLASSES DE REQUISIÇÃO
# ============================================

class SolicitacaoAnaliseTEA(BaseModel):
    descricao_professor: str
    idade_aluno: Optional[int] = None
    nivel_suporte: Optional[str] = None
    interesses_especificos: Optional[str] = None
    sensibilidades_sensoriais: Optional[str] = None
    incluir_estruturas: bool = True
    recursos_disponiveis: Optional[str] = None
    buscar_online: bool = True


class SolicitacaoCatalogoTEA(BaseModel):
    necessidade: str
    tecnologia_recusada: str
    faixa_etaria: Optional[str] = None
    categoria_preferida: Optional[str] = None
    buscar_online: bool = True


# ============================================
# CATÁLOGO FIXO DE RECURSOS
# ============================================

CATALOGO_RECURSOS = {
    "comunicacao": [
        {
            "nome": "Prancha de Comunicação com Figuras Recortadas",
            "materiais": "Revistas velhas, tesoura, cola, papelão, velcro",
            "como_fazer": "1. Recorte figuras de revistas. 2. Cole em um papelão. 3. Fixe com velcro.",
            "como_usar": "O aluno aponta para a figura do que deseja.",
            "para_que_serve": "Facilita a comunicação não-verbal."
        },
        {
            "nome": "Cartões de Comunicação com Tampinhas",
            "materiais": "Tampinhas, papel, canetinhas, cola",
            "como_fazer": "1. Corte círculos de papel. 2. Desenhe símbolos. 3. Cole nas tampinhas.",
            "como_usar": "O aluno entrega a tampinha com o símbolo.",
            "para_que_serve": "Permite comunicação simples."
        }
    ],
    "regulacao_sensorial": [
        {
            "nome": "Garrafa da Calma",
            "materiais": "Garrafa PET, água, glitter, corante, cola quente",
            "como_fazer": "1. Encha com água. 2. Adicione glitter e corante. 3. Feche com cola quente.",
            "como_usar": "Agite e peça para observar o glitter caindo.",
            "para_que_serve": "Ajuda na regulação emocional."
        },
        {
            "nome": "Kit Sensorial com Caixas",
            "materiais": "Caixa de sapato, tecidos variados, botões",
            "como_fazer": "1. Forre a caixa com tecidos. 2. Cole botões e fitas.",
            "como_usar": "Deixe o aluno explorar as texturas.",
            "para_que_serve": "Estimula o tato."
        },
        {
            "nome": "Fone de Ouvido Caseiro",
            "materiais": "Fone velho, espuma, tecido",
            "como_fazer": "1. Retire as almofadas. 2. Encha com espuma. 3. Recubra com tecido.",
            "como_usar": "Use em momentos de muito barulho.",
            "para_que_serve": "Reduz a sobrecarga auditiva."
        }
    ],
    "estruturacao": [
        {
            "nome": "Rotina Visual com Caixas",
            "materiais": "Caixas de fósforo, papel, canetinhas",
            "como_fazer": "1. Desenhe as atividades. 2. Coloque em caixas. 3. Organize em sequência.",
            "como_usar": "Mostre a sequência do dia.",
            "para_que_serve": "Dá previsibilidade."
        }
    ],
    "interacao_social": [
        {
            "nome": "História Social Ilustrada",
            "materiais": "Papel, canetinhas, grampeador",
            "como_fazer": "1. Crie uma história. 2. Ilustre. 3. Grampeie.",
            "como_usar": "Leia antes da situação acontecer.",
            "para_que_serve": "Ensina habilidades sociais."
        }
    ]
}


# ============================================
# BUSCA ONLINE
# ============================================

def buscar_recursos_online(termo_busca: str, max_resultados: int = 3) -> List[Dict]:
    try:
        from ddgs import DDGS

        query = f"tecnologia assistiva TEA autismo {termo_busca}"
        resultados = []

        with DDGS(timeout=10) as ddgs:
            for r in ddgs.text(query, region="pt-br", max_results=max_resultados):
                resultados.append({
                    "titulo": r.get("title", ""),
                    "resumo": r.get("body", "")[:300],
                    "link": r.get("href", "")
                })

        print(f"✅ Busca online: {len(resultados)} resultados")
        return resultados

    except Exception as e:
        print(f"⚠️ Busca online indisponível: {e}")
        return []


def formatar_resultados_online(resultados: List[Dict]) -> str:
    if not resultados:
        return ""
    texto = "\n### 🌐 RECURSOS ENCONTRADOS NA INTERNET\n\n"
    for i, r in enumerate(resultados, 1):
        texto += f"**{i}. {r['titulo']}**\n\n{r['resumo']}...\n\n🔗 Fonte: {r['link']}\n\n---\n\n"
    return texto


# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def identificar_categoria(descricao: str) -> str:
    descricao = descricao.lower()
    if any(p in descricao for p in ["comunica", "fala", "conversa"]):
        return "comunicacao"
    elif any(p in descricao for p in ["barulho", "auditiva", "sensorial", "crise"]):
        return "regulacao_sensorial"
    elif any(p in descricao for p in ["rotina", "organiza", "estrutura"]):
        return "estruturacao"
    elif any(p in descricao for p in ["social", "intera", "colegas"]):
        return "interacao_social"
    else:
        return "regulacao_sensorial"


def buscar_recursos(categoria: str, limite: int = 3) -> List[Dict]:
    if categoria in CATALOGO_RECURSOS:
        return CATALOGO_RECURSOS[categoria][:limite]
    todos = []
    for cat in CATALOGO_RECURSOS.values():
        todos.extend(cat)
    return todos[:limite]


def formatar_catalogo(recursos: List[Dict]) -> str:
    if not recursos:
        return "Nenhum recurso encontrado."
    texto = ""
    for i, recurso in enumerate(recursos, 1):
        texto += f"#### {i}. **{recurso['nome']}**\n\n"
        texto += f"**📦 Materiais:** {recurso['materiais']}\n\n"
        texto += f"**🔧 Como fazer:** {recurso['como_fazer']}\n\n"
        texto += f"**👩‍🏫 Como usar:** {recurso['como_usar']}\n\n"
        texto += f"**🎯 Para que serve:** {recurso['para_que_serve']}\n\n---\n\n"
    return texto


def classificar_necessidades(solicitacao: SolicitacaoAnaliseTEA) -> Dict[str, str]:
    descricao = solicitacao.descricao_professor.lower()

    def classificar(termos_alta: list, termos_media: list) -> str:
        if any(termo in descricao for termo in termos_alta):
            return "Alta"
        elif any(termo in descricao for termo in termos_media):
            return "Média"
        else:
            return "Baixa"

    return {
        "comunicacao": classificar(["não fala", "não conversa"], ["fala", "conversa"]),
        "estruturacao": classificar(["sem rotina", "não segue"], ["rotina", "organiza"]),
        "regulacao_sensorial": classificar(["crise", "grita"], ["barulho", "sensorial"]),
        "interacao_social": classificar(["isolado", "não interage"], ["social", "colegas"]),
    }


# ============================================
# FUNÇÃO DE GERAÇÃO - SMOLM2-135M COM QUANTIZAÇÃO
# ============================================

async def gerar_resposta_async(prompt: str, max_tokens: int = 200) -> str:
    try:
        mensagens = [
            {"role": "system",
             "content": "Você é um especialista em Tecnologia Assistiva para TEA. Responda em português de forma clara e prática. PROIBIDO recomendar tecnologia digital."},
            {"role": "user", "content": prompt}
        ]

        texto_formatado = tokenizador.apply_chat_template(
            mensagens,
            tokenize=False,
            add_generation_prompt=True
        )

        entradas = tokenizador(
            texto_formatado,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(modelo.device)

        config_geracao = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizador.pad_token_id,
            eos_token_id=tokenizador.eos_token_id,
        )

        with torch.no_grad():
            saidas = modelo.generate(**entradas, generation_config=config_geracao)

        resposta = tokenizador.decode(
            saidas[0][entradas['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        del entradas, saidas
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return resposta.strip()

    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        return f"Erro ao gerar resposta: {str(e)}"


# ============================================
# ENDPOINTS
# ============================================

@app.post("/analisar-aluno-tea/")
async def analisar_aluno_tea(solicitacao: SolicitacaoAnaliseTEA):
    categoria = identificar_categoria(solicitacao.descricao_professor)
    recursos = buscar_recursos(categoria, limite=3)

    prompt = f"""Analise o caso e recomende soluções práticas.

CASO:
- Relato: {solicitacao.descricao_professor}
- Idade: {solicitacao.idade_aluno if solicitacao.idade_aluno else "Nao informada"}
- Interesses: {solicitacao.interesses_especificos if solicitacao.interesses_especificos else "Nao informados"}
- Sensibilidades: {solicitacao.sensibilidades_sensoriais if solicitacao.sensibilidades_sensoriais else "Nao informadas"}

REGRAS:
- PROIBIDO recomendar apps, tablets ou tecnologia digital.
- Use APENAS materiais recicláveis.

RESPONDA:
1. BARREIRAS: Quais as principais dificuldades?
2. SOLUÇÃO PRINCIPAL: Qual adaptação recomenda?
3. SOLUÇÕES ALTERNATIVAS: Liste 2 opções
4. ADAPTAÇÕES NA ROTINA: Como adaptar?
5. DICAS PRÁTICAS: O que fazer hoje?"""

    try:
        resposta_ia = await gerar_resposta_async(prompt)
        if len(resposta_ia.split()) > 20 and "Erro" not in resposta_ia:
            analise = f"""### ANÁLISE DO CASO

**Baseado no relato:** "{solicitacao.descricao_professor}"

**Idade:** {solicitacao.idade_aluno if solicitacao.idade_aluno else "Nao informada"}

---

{resposta_ia}

---

### 🎯 RECURSOS RECOMENDADOS

{formatar_catalogo(recursos)}

### 💡 DICAS PARA O PROFESSOR

1. Use os interesses do aluno como ponto de partida
2. Observe o que funciona e ajuste
3. Comece com um recurso de cada vez
4. Reforce positivamente qualquer participação
5. Comunique-se com a família
"""
        else:
            analise = formatar_resposta_fallback(solicitacao, recursos)
    except Exception as e:
        print(f"❌ Erro na IA: {e}")
        analise = formatar_resposta_fallback(solicitacao, recursos)

    if solicitacao.buscar_online:
        termo = solicitacao.interesses_especificos or solicitacao.descricao_professor[:60]
        resultados_online = buscar_recursos_online(termo)
        if resultados_online:
            analise += formatar_resultados_online(resultados_online)

    categorias = classificar_necessidades(solicitacao)

    resultado = {
        "analise": analise,
        "categorias_necessidade": categorias,
    }

    if solicitacao.incluir_estruturas:
        estruturas = []
        if categorias.get("estruturacao") in ["Alta", "Média"]:
            estruturas.append("rotina_visual")
        if categorias.get("regulacao_sensorial") in ["Alta", "Média"]:
            estruturas.append("kit_sensorial")
        if categorias.get("comunicacao") in ["Alta", "Média"]:
            estruturas.append("prancha_comunicacao")
        if categorias.get("interacao_social") in ["Alta", "Média"]:
            estruturas.append("historias_sociais")
        resultado["estruturas_recomendadas"] = estruturas

    return resultado


@app.post("/catalogo-alternativas-tea/")
async def catalogo_tecnologias_tea(solicitacao: SolicitacaoCatalogoTEA):
    if "comunica" in solicitacao.necessidade.lower():
        categoria = "comunicacao"
    elif any(p in solicitacao.necessidade.lower() for p in ["barulho", "sensorial"]):
        categoria = "regulacao_sensorial"
    elif "rotina" in solicitacao.necessidade.lower():
        categoria = "estruturacao"
    elif "social" in solicitacao.necessidade.lower():
        categoria = "interacao_social"
    else:
        categoria = solicitacao.categoria_preferida if solicitacao.categoria_preferida else "regulacao_sensorial"

    recursos = buscar_recursos(categoria, limite=3)

    catalogo = f"""### 📚 CATÁLOGO DE TECNOLOGIAS ASSISTIVAS

**Necessidade:** {solicitacao.necessidade}

**Alternativas recomendadas:**

{formatar_catalogo(recursos)}
"""

    if solicitacao.buscar_online:
        resultados_online = buscar_recursos_online(solicitacao.necessidade)
        if resultados_online:
            catalogo += formatar_resultados_online(resultados_online)

    return {"catalogo": catalogo}


@app.get("/saude/")
async def verificar_saude():
    return {
        "status": "saudavel",
        "modelo": "SmolLM2-135M-Instruct (8-bit)",
        "dispositivo": "cpu",
        "parametros": "135M"
    }


def formatar_resposta_fallback(solicitacao: SolicitacaoAnaliseTEA, recursos: List[Dict]) -> str:
    return f"""### ANÁLISE DO CASO

**Baseado no relato:** "{solicitacao.descricao_professor}"

**Idade:** {solicitacao.idade_aluno if solicitacao.idade_aluno else "Nao informada"}

---

### 🎯 RECURSOS RECOMENDADOS

{formatar_catalogo(recursos)}

### 💡 DICAS PARA O PROFESSOR

1. Use os interesses do aluno como ponto de partida
2. Observe o que funciona e ajuste
3. Comece com um recurso de cada vez
4. Reforce positivamente qualquer participação
5. Comunique-se com a família
"""


# ============================================
# INICIALIZAÇÃO
# ============================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🎓 API de Tecnologias Assistivas para TEA - SmolLM2-135M (8-bit)")
    print("=" * 50)
    print(f"📊 Documentação: http://localhost:8000/docs")
    print(f"📝 Endpoint: POST /analisar-aluno-tea/")
    print(f"💻 Modelo: SmolLM2-135M (quantizado 8-bit)")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)