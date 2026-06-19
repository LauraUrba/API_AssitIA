from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import gc
import re
import os
from llama_cpp import Llama

# ============================================
# CONFIGURAÇÃO INICIAL DA API
# ============================================

app = FastAPI(
    title="API de Tecnologias Assistivas para TEA - TCC",
    description="Sistema especializado em recomendar tecnologias assistivas para alunos com TEA",
    version="2.0.0"
)

print("🚀 Carregando modelo LFM2.5-350M (GGUF)...")

# 🔥 CAMINHO DO MODELO GGUF
MODEL_PATH = "./models/LFM2.5-350M-Q4_K_M.gguf"

# Verifica se o arquivo existe
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Arquivo não encontrado: {MODEL_PATH}")
    print("📥 Baixe o modelo GGUF com:")
    print("   hf download LiquidAI/LFM2.5-350M-GGUF LFM2.5-350M-Q4_K_M.gguf --local-dir ./models")
    exit(1)

# Carrega o modelo com llama.cpp
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=32768,  # Contexto de 32K tokens
    n_threads=2,  # Ajuste para a CPU
    n_gpu_layers=0,  # 0 = usar apenas CPU
    verbose=False,
)

print("✅ Modelo LFM2.5-350M (GGUF) carregado com sucesso!")
print(f"💻 Modo: CPU")


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
            "nome": "Prancha de Comunicação com Figuras Recortadas (PECS adaptado)",
            "materiais": "Revistas velhas, tesoura, cola, papelão, velcro ou fita adesiva",
            "como_fazer": "1. Recorte figuras de revistas que representem necessidades básicas. 2. Cole as figuras em quadrados de papelão. 3. Cole velcro atrás de cada figura.",
            "como_usar": "O aluno aponta ou entrega a figura do que deseja. O professor verbaliza a palavra.",
            "para_que_serve": "Facilita a comunicação não-verbal e reduz a frustração."
        },
        {
            "nome": "Cartões de Comunicação com Tampinhas",
            "materiais": "Tampinhas de garrafa, papel, canetinhas, cola",
            "como_fazer": "1. Corte círculos de papel do tamanho das tampinhas. 2. Desenhe símbolos simples. 3. Cole os papéis nas tampinhas.",
            "como_usar": "O aluno entrega a tampinha com o símbolo que representa o que quer.",
            "para_que_serve": "Permite comunicação simples e direta."
        }
    ],
    "regulacao_sensorial": [
        {
            "nome": "Garrafa da Calma (Sensory Bottle)",
            "materiais": "Garrafa PET, água, glitter, corante, cola quente",
            "como_fazer": "1. Encha a garrafa com água até 3/4. 2. Adicione glitter e corante. 3. Feche com cola quente.",
            "como_usar": "Quando o aluno estiver agitado, agite a garrafa e peça para observar o glitter caindo.",
            "para_que_serve": "Ajuda na regulação emocional e reduz crises."
        },
        {
            "nome": "Kit Sensorial com Caixas e Tecidos",
            "materiais": "Caixa de sapato, tecidos variados, botões, fitas",
            "como_fazer": "1. Forre a caixa com diferentes tecidos. 2. Cole botões e fitas para criar texturas.",
            "como_usar": "Deixe o aluno explorar as texturas livremente.",
            "para_que_serve": "Estimula o tato e ajuda na regulação sensorial."
        },
        {
            "nome": "Fone de Ouvido Caseiro (Redutor de Ruído)",
            "materiais": "Fone de ouvido velho, espuma, tecido",
            "como_fazer": "1. Retire as almofadas do fone. 2. Encha com espuma. 3. Recubra com tecido macio.",
            "como_usar": "O aluno usa em momentos de muito barulho.",
            "para_que_serve": "Reduz a sobrecarga auditiva."
        }
    ],
    "estruturacao": [
        {
            "nome": "Rotina Visual com Caixas de Fósforo",
            "materiais": "Caixas de fósforo vazias, papel, canetinhas, cola",
            "como_fazer": "1. Desenhe as atividades em tiras de papel. 2. Coloque cada uma dentro de uma caixinha. 3. Organize as caixas em sequência.",
            "como_usar": "Mostre a sequência do dia pela manhã. Ao concluir cada atividade, retire a caixa.",
            "para_que_serve": "Dá previsibilidade ao dia e reduz ansiedade."
        }
    ],
    "interacao_social": [
        {
            "nome": "História Social Ilustrada",
            "materiais": "Papel, canetinhas, grampeador",
            "como_fazer": "1. Crie uma história curta sobre uma situação social específica. 2. Ilustre cada passo. 3. Grampeie como um livrinho.",
            "como_usar": "Leia a história com o aluno antes da situação acontecer.",
            "para_que_serve": "Ensina habilidades sociais de forma visual e previsível."
        }
    ]
}


# ============================================
# BUSCA ONLINE
# ============================================

def buscar_recursos_online(termo_busca: str, max_resultados: int = 3) -> List[Dict]:
    try:
        try:
            from ddgs import DDGS
            version = "ddgs"
            use_timeout = True
        except ImportError:
            from duckduckgo_search import DDGS
            version = "duckduckgo-search"
            use_timeout = False

        query = f"tecnologia assistiva TEA autismo {termo_busca}"
        resultados = []

        if use_timeout:
            with DDGS(timeout=10) as ddgs:
                for r in ddgs.text(query, region="pt-br", max_results=max_resultados):
                    resultados.append({
                        "titulo": r.get("title", ""),
                        "resumo": r.get("body", "")[:300],
                        "link": r.get("href", "")
                    })
        else:
            with DDGS() as ddgs:
                for r in ddgs.text(query, region="pt-br", max_results=max_resultados):
                    resultados.append({
                        "titulo": r.get("title", ""),
                        "resumo": r.get("body", "")[:300],
                        "link": r.get("href", "")
                    })

        print(f"✅ Busca online com {version}: {len(resultados)} resultados")
        return resultados

    except Exception as e:
        print(f"⚠️ Busca online indisponível: {type(e).__name__}: {e}")
        return []


def formatar_resultados_online(resultados: List[Dict]) -> str:
    if not resultados:
        return ""

    texto = "\n### 🌐 RECURSOS ENCONTRADOS NA INTERNET (complementar)\n\n"
    texto += "*Estes resultados foram buscados online e podem trazer ideias adicionais.*\n\n"

    for i, r in enumerate(resultados, 1):
        texto += f"**{i}. {r['titulo']}**\n\n"
        texto += f"{r['resumo']}...\n\n"
        texto += f"🔗 Fonte: {r['link']}\n\n"
        texto += "---\n\n"

    return texto


# ============================================
# FUNÇÕES DO CATÁLOGO
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
        texto += f"**🎯 Para que serve:** {recurso['para_que_serve']}\n\n"
        texto += "---\n\n"
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
        "comunicacao": classificar(
            ["não fala", "não conversa", "não comunica"],
            ["fala", "conversa", "comunica"]
        ),
        "estruturacao": classificar(
            ["sem rotina", "não segue rotina"],
            ["rotina", "organiza", "estrutura"]
        ),
        "regulacao_sensorial": classificar(
            ["crise", "grita", "sobrecarga"],
            ["barulho", "sensorial", "som"]
        ),
        "interacao_social": classificar(
            ["isolado", "não interage"],
            ["social", "colegas", "intera"]
        ),
    }


# ============================================
# FUNÇÃO DE GERAÇÃO
# ============================================

async def gerar_resposta_async(prompt: str, max_tokens: int = 500) -> str:
    try:
        mensagens = [
            {"role": "system",
             "content": "Você é um especialista em Tecnologia Assistiva para TEA. Responda em português de forma clara, prática e direta. Use apenas materiais recicláveis e acessíveis. NUNCA recomende aplicativos, tablets ou tecnologia digital."},
            {"role": "user", "content": prompt}
        ]

        response = llm.create_chat_completion(
            messages=mensagens,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=False,
        )

        resposta = response["choices"][0]["message"]["content"]

        if "Olá" in resposta or "Como posso" in resposta:
            resposta = resposta.replace("Olá", "").replace("Como posso", "")

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

    prompt = f"""Você é um especialista em Tecnologia Assistiva para TEA. NUNCA inicie a resposta com saudações. Vá direto à análise técnica.

    CASO DO ALUNO:
    - Relato: {solicitacao.descricao_professor}
    - Idade: {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}
    - Interesses: {solicitacao.interesses_especificos if solicitacao.interesses_especificos else "Não informados"}
    - Sensibilidades: {solicitacao.sensibilidades_sensoriais if solicitacao.sensibilidades_sensoriais else "Não informadas"}
    - Recursos disponíveis: {solicitacao.recursos_disponiveis if solicitacao.recursos_disponiveis else "Não informados"}

    REGRAS:
    - PROIBIDO recomendar aplicativos, tablets ou tecnologia digital.
    - Use APENAS materiais recicláveis e acessíveis.

    RESPONDA EM PORTUGUÊS:

    1. BARREIRAS: Quais as principais dificuldades?
    2. SOLUÇÃO PRINCIPAL: Qual adaptação você recomenda?
    3. SOLUÇÕES ALTERNATIVAS: Liste 2 outras opções
    4. ADAPTAÇÕES NA ROTINA: Como adaptar o dia a dia?
    5. DICAS PRÁTICAS: O que o professor pode fazer hoje?"""

    try:
        resposta_ia = await gerar_resposta_async(prompt)
        if len(resposta_ia.split()) > 30:
            analise = f"""### ANÁLISE DO CASO

**Baseado no relato:** "{solicitacao.descricao_professor}"

**Idade:** {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}

---

{resposta_ia}

---

### 🎯 RECURSOS RECOMENDADOS

{formatar_catalogo(recursos)}

### 💡 DICAS PARA O PROFESSOR

1. **Use os interesses do aluno** como ponto de partida
2. **Observe** o que funciona e ajuste
3. **Comece com um recurso** de cada vez
4. **Reforce positivamente** qualquer tentativa de participação
5. **Comunique-se com a família** sobre o que funciona em casa
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
        "modelo": "LiquidAI/LFM2.5-350M (GGUF)",
        "dispositivo": "cpu",
        "versao": "Q4_K_M"
    }


def formatar_resposta_fallback(solicitacao: SolicitacaoAnaliseTEA, recursos: List[Dict]) -> str:
    return f"""### ANÁLISE DO CASO

**Baseado no relato:** "{solicitacao.descricao_professor}"

**Idade:** {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}

---

### 🎯 RECURSOS RECOMENDADOS

{formatar_catalogo(recursos)}

### 💡 DICAS PARA O PROFESSOR

1. **Use os interesses do aluno** como ponto de partida
2. **Observe** o que funciona e ajuste
3. **Comece com um recurso** de cada vez
4. **Reforce positivamente** qualquer tentativa de participação
5. **Comunique-se com a família** sobre o que funciona em casa
"""


# ============================================
# INICIALIZAÇÃO
# ============================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🎓 API de Tecnologias Assistivas para TEA - LFM2.5-350M (GGUF)")
    print("=" * 50)
    print(f"📊 Documentação: http://localhost:8000/docs")
    print(f"📝 Endpoint: POST /analisar-aluno-tea/")
    print(f"💻 Modelo: LFM2.5-350M (Q4_K_M)")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)