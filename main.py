from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from llama_cpp import Llama
import gc
import re
import os
import warnings

# SUPRIME WARNINGS
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ============================================
# CONFIGURAÇÃO INICIAL DA API
# ============================================

app = FastAPI(
    title="API de Tecnologias Assistivas para TEA - TCC",
    description="Sistema especializado em recomendar tecnologias assistivas para alunos com TEA",
    version="2.0.0"
)

print("Carregando modelo TinyLlama-1.1B (GGUF)...")

# CAMINHO DO MODELO TINYLLAMA
MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Verifica se o arquivo existe
if not os.path.exists(MODEL_PATH):
    print(f"X Arquivo não encontrado: {MODEL_PATH}")
    exit(1)

# CARREGA O MODELO COM CONFIGURAÇÕES MÍNIMAS
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=128,           # Contexto reduzido
        n_threads=1,         # Apenas 1 thread
        n_gpu_layers=0,      # 0 = apenas CPU
        verbose=False,
        n_batch=32,
        use_mmap=True,
        use_mlock=False,
    )
    print("Modelo TinyLlama-1.1B carregado com sucesso!")
    modelo_ok = True
except Exception as e:
    print(f" X Erro ao carregar modelo: {e}")
    modelo_ok = False
    llm = None

print(f"💻 Modo: {'IA' if modelo_ok else 'Catálogo Fixo'}")

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

    # 🔥 CAMPOS DO FORMULÁRIO
    comunicacao: Optional[str] = None  # Verbal, Não verbal, Limitada, Mista
    motor: Optional[str] = None  # Motor Fino, Motor Grosso
    atencao: Optional[str] = None  # Alta, Média, Baixa
    comportamentos: Optional[str] = None  # Repetitivos, Flexibilidade

    # 🔥 NOVOS CAMPOS DO FORMULÁRIO
    area_principal: Optional[
        str] = None  # Comunicação, Regulação Sensorial, Motor, Cognitivo, Interação Social, Estruturação
    prioridade: Optional[str] = None  # Alta, Média, Baixa
    areas_atencao: Optional[str] = None  # Lista de áreas separadas por vírgula
    interesses: Optional[str] = None  # Lista de interesses + observações
    sensibilidades: Optional[str] = None  # Lista de sensibilidades + observações
    recursos: Optional[str] = None  # Lista de recursos + observações


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
    ],
    "motor": [
        {
            "nome": "Teclado Adaptado com Papelão",
            "materiais": "Papelão, teclas desenhadas, fita adesiva",
            "como_fazer": "1. Desenhe um teclado em papelão. 2. Recorte as teclas. 3. Fixe com fita.",
            "como_usar": "O aluno usa para digitar ou apontar letras.",
            "para_que_serve": "Auxilia na coordenação motora fina."
        }
    ],
    "cognitivo": [
        {
            "nome": "Agenda Visual de Tarefas",
            "materiais": "Papel, canetinhas, velcro",
            "como_fazer": "1. Desenhe as tarefas. 2. Recorte. 3. Cole velcro.",
            "como_usar": "O aluno organiza as tarefas do dia.",
            "para_que_serve": "Ajuda na organização e planejamento."
        }
    ]
}


# ============================================
# BUSCA ONLINE
# ============================================

def buscar_recursos_online(termo_busca: str, max_resultados: int = 5) -> List[Dict]:
    """
    Busca tecnologias assistivas online com mais fontes
    """
    try:
        from ddgs import DDGS

        # 🔥 MÚLTIPLAS QUERIES PARA MAIS RESULTADOS
        queries = [
            f"tecnologia assistiva TEA autismo {termo_busca}",
            f"dicas autismo TEA professor {termo_busca}",
            f"atividades autismo sala de aula {termo_busca}",
            f"recursos pedagógicos autismo TEA {termo_busca}"
        ]

        resultados = []
        for query in queries:
            with DDGS(timeout=10) as ddgs:
                for r in ddgs.text(query, region="pt-br", max_results=2):
                    # 🔥 VERIFICA SE O LINK JÁ FOI ADICIONADO
                    if not any(res.get("link") == r.get("href") for res in resultados):
                        # 🔥 INFORMA A FONTE
                        fonte = "YouTube" if "youtube" in r.get("href", "") else \
                            "Instagram" if "instagram" in r.get("href", "") else \
                                "TikTok" if "tiktok" in r.get("href", "") else \
                                    "Site"
                        resultados.append({
                            "titulo": r.get("title", ""),
                            "resumo": r.get("body", "")[:300],
                            "link": r.get("href", ""),
                            "fonte": fonte
                        })

        return resultados[:5]  # Máximo 5 resultados

    except Exception as e:
        print(f"⚠️ Busca online indisponível: {e}")
        return []


def formatar_resultados_online(resultados: List[Dict]) -> str:
    if not resultados:
        return ""

    texto = "\n### 🌐 RECURSOS ENCONTRADOS NA INTERNET\n\n"

    for i, r in enumerate(resultados, 1):
        # 🔥 ÍCONE DA FONTE
        icone = "▶️" if r.get("fonte") == "YouTube" else \
            "📸" if r.get("fonte") == "Instagram" else \
                "🎵" if r.get("fonte") == "TikTok" else \
                    "🌐"
        texto += f"**{i}. {icone} {r['titulo']}**\n\n"
        texto += f"{r['resumo']}...\n\n"
        texto += f"🔗 Fonte: {r['link']}\n\n"
        texto += "---\n\n"

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
    """
    Classifica as necessidades baseado nos campos da taxonomia
    """
    descricao = solicitacao.descricao_professor.lower()

    #Função auxiliar para classificar com Baixa
    def classificar(termos_alta: list, termos_media: list, valor_fornecido: Optional[str] = None) -> str:
        # Se o professor forneceu um valor específico
        if valor_fornecido is not None:
            # 🔥 Valores que indicam ALTA necessidade
            if valor_fornecido in [
                "Não verbal",
                "Limitada",
                "Repetitivos",
                "Baixa",  # atenção baixa
                "Motor Fino",  # dificuldade motora fina
                "Não verbal",
                "não verbal",
                "Nao fala",
                "não fala"
            ]:
                return "Alta"

            # Valores que indicam MÉDIA necessidade
            elif valor_fornecido in [
                "Mista",
                "Moderada",
                "Motor Grosso",
                "Verbal limitada",
                "Fala pouco",
                "Média",
                "Regular"
            ]:
                return "Média"

            # Valores que indicam BAIXA necessidade
            else:
                return "Baixa"

        # Fallback: analisa a descrição
        if any(termo in descricao for termo in termos_alta):
            return "Alta"
        elif any(termo in descricao for termo in termos_media):
            return "Média"
        else:
            return "Baixa"

    # Classifica cada área
    comunicacao = classificar(
        termos_alta=["não fala", "não conversa", "não verbal"],
        termos_media=["fala pouco", "comunica com gestos", "limitada"],
        valor_fornecido=solicitacao.comunicacao
    )

    motor = classificar(
        termos_alta=["coordenação", "motor fino", "dificuldade motora", "escrever"],
        termos_media=["coordenação média", "motor regular"],
        valor_fornecido=solicitacao.motor
    )

    atencao = classificar(
        termos_alta=["atenção", "concentração", "distração", "hiperativo", "não participa", "não presta atenção"],
        termos_media=["atenção média", "concentração média", "participa pouco"],
        valor_fornecido=solicitacao.atencao
    )

    comportamentos = classificar(
        termos_alta=["repetitivo", "ritualístico", "inflexível", "estereotipia", "grita", "corre", "andando"],
        termos_media=["alguns repetitivos", "flexibilidade média", "agitado"],
        valor_fornecido=solicitacao.comportamentos
    )

    regulacao_sensorial = classificar(
        termos_alta=["barulho", "crise", "sensorial", "auditiva", "sobrecarga", "sensibilidade"],
        termos_media=["sensibilidade média", "desconforto", "incomoda"],
        valor_fornecido=None
    )

    interacao_social = classificar(
        termos_alta=["colegas", "social", "interação", "isolado", "medo"],
        termos_media=["interage pouco", "social média"],
        valor_fornecido=None
    )

    estruturacao = classificar(
        termos_alta=["rotina", "organiza", "estrutura", "ordem", "planejamento"],
        termos_media=["rotina média", "organização média"],
        valor_fornecido=None
    )

    return {
        "comunicacao": comunicacao,
        "motor": motor,
        "atencao": atencao,
        "comportamentos": comportamentos,
        "regulacao_sensorial": regulacao_sensorial,
        "interacao_social": interacao_social,
        "estruturacao": estruturacao,
    }

# ============================================
# FUNÇÃO DE GERAÇÃO
# ============================================

async def gerar_resposta_async(prompt: str, max_tokens: int = 80) -> str:
    """Gera resposta usando TinyLlama via llama-cpp-python"""
    if not modelo_ok or llm is None:
        return "Modelo indisponível. Use o catálogo de recursos."

    try:
        mensagens = [
            {"role": "system",
             "content": "Você é um especialista em Tecnologia Assistiva para TEA. Responda em português de forma clara e prática. PROIBIDO recomendar tecnologia digital."},
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

        del response
        gc.collect()

        return resposta.strip()

    except Exception as e:
        print(f"X Erro na geração: {e}")
        return f"Erro ao gerar resposta: {str(e)}"


# ============================================
# ENDPOINTS
# ============================================

@app.post("/analisar-aluno-tea/")
async def analisar_aluno_tea(solicitacao: SolicitacaoAnaliseTEA):
    categoria = identificar_categoria(solicitacao.descricao_professor)
    recursos = buscar_recursos(categoria, limite=3)

    # 🔥 COMBINA OS CAMPOS DO FORMULÁRIO
    interesses_final = []
    if solicitacao.interesses_especificos:
        interesses_final.append(solicitacao.interesses_especificos)
    if solicitacao.interesses:
        interesses_final.append(solicitacao.interesses)
    interesses_texto = ", ".join(filter(None, interesses_final)) if interesses_final else "Nao informados"

    sensibilidades_final = []
    if solicitacao.sensibilidades_sensoriais:
        sensibilidades_final.append(solicitacao.sensibilidades_sensoriais)
    if solicitacao.sensibilidades:
        sensibilidades_final.append(solicitacao.sensibilidades)
    sensibilidades_texto = ", ".join(filter(None, sensibilidades_final)) if sensibilidades_final else "Nao informadas"

    recursos_final = []
    if solicitacao.recursos_disponiveis:
        recursos_final.append(solicitacao.recursos_disponiveis)
    if solicitacao.recursos:
        recursos_final.append(solicitacao.recursos)
    recursos_texto = ", ".join(filter(None, recursos_final)) if recursos_final else "Nao informados"

    # 🔥 PROMPT COMPLETO COM TODOS OS DADOS
    prompt = f"""Analise o caso e recomende soluções práticas.

CASO:
- Relato: {solicitacao.descricao_professor}
- Idade: {solicitacao.idade_aluno if solicitacao.idade_aluno else "Nao informada"}
- Nível de suporte: {solicitacao.nivel_suporte if solicitacao.nivel_suporte else "Nao informado"}
- Interesses: {interesses_texto}
- Sensibilidades: {sensibilidades_texto}
- Recursos disponíveis: {recursos_texto}

DADOS DO FORMULÁRIO:
- Área Principal: {solicitacao.area_principal if solicitacao.area_principal else "Nao informada"}
- Prioridade: {solicitacao.prioridade if solicitacao.prioridade else "Nao informada"}
- Áreas de Atenção: {solicitacao.areas_atencao if solicitacao.areas_atencao else "Nao informadas"}
- Comunicação: {solicitacao.comunicacao if solicitacao.comunicacao else "Nao informado"}
- Motor: {solicitacao.motor if solicitacao.motor else "Nao informado"}
- Atenção: {solicitacao.atencao if solicitacao.atencao else "Nao informada"}
- Comportamentos: {solicitacao.comportamentos if solicitacao.comportamentos else "Nao informados"}

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
        "categorias_necessidade": categorias,  # Agora inclui motor, atencao, comportamentos
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
        if categorias.get("motor") in ["Alta", "Média"]:
            estruturas.append("teclado_adaptado")
        if categorias.get("atencao") in ["Alta", "Média"]:
            estruturas.append("agenda_visual")
        if categorias.get("comportamentos") in ["Alta", "Média"]:
            estruturas.append("estratégias_comportamentais")

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
        "modelo": "TinyLlama-1.1B (GGUF)",
        "dispositivo": "cpu",
        "parametros": "1.1B"
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