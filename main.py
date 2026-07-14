# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from llama_cpp import Llama
import gc
import re
import os
import warnings
import time

# SUPRIME WARNINGS
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ============================================
# CONFIGURAÇÃO INICIAL DA API
# ============================================

app = FastAPI(
    title="API de Tecnologias Assistivas para TEA",
    description="Sistema especializado em recomendar tecnologias assistivas para alunos com TEA",
    version="3.0.0"
)

print("Carregando modelo SmolLM2-135M (GGUF)...")

# CAMINHO DO MODELO GGUF
MODEL_PATH = "./models/HuggingFaceTB.SmolLM2-135M-Instruct.Q4_K_M.gguf"

# VERIFICAR SE O MODELO EXISTE
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Arquivo não encontrado: {MODEL_PATH}")
    print("⚠️ Baixe o modelo com:")
    print("wget -O models/HuggingFaceTB.SmolLM2-135M-Instruct.Q4_K_M.gguf \\")
    print("  https://huggingface.co/DevQuasar/HuggingFaceTB.SmolLM2-135M-Instruct-GGUF/resolve/main/HuggingFaceTB.SmolLM2-135M-Instruct.Q4_K_M.gguf")
    modelo_ok = False
    llm = None
else:
    try:
        # CARREGAR O MODELO COM LLAMA-CPP
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=1024,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
            n_batch=256,
            use_mmap=True,
            use_mlock=False,
        )
        print("✅ Modelo SmolLM2-135M (GGUF) carregado com sucesso!")
        modelo_ok = True
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
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

    # CAMPOS DO FORMULÁRIO (CHECKBOXES)
    comunicacao: Optional[str] = None
    motor: Optional[str] = None
    atencao: Optional[str] = None
    comportamentos: Optional[str] = None

    # CAMPOS ADICIONAIS
    area_principal: Optional[str] = None
    prioridade: Optional[str] = None
    areas_atencao: Optional[str] = None
    interesses: Optional[str] = None
    sensibilidades: Optional[str] = None
    recursos: Optional[str] = None


class SolicitacaoCatalogoTEA(BaseModel):
    necessidade: str
    tecnologia_recusada: str
    faixa_etaria: Optional[str] = None
    categoria_preferida: Optional[str] = None
    buscar_online: bool = True


# ============================================
# CATÁLOGO FIXO DE RECURSOS (5 POR CATEGORIA)
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
        },
        {
            "nome": "Pasta de Comunicação com Símbolos PECS",
            "materiais": "Pasta, velcro, figuras impressas, plastificador",
            "como_fazer": "1. Imprima figuras. 2. Plastifique. 3. Fixe velcro na pasta e nas figuras.",
            "como_usar": "O aluno entrega a figura para o professor.",
            "para_que_serve": "Sistema de comunicação por troca de figuras."
        },
        {
            "nome": "Cartazes de Rotina de Comunicação",
            "materiais": "Cartolina, canetinhas, figuras, fita adesiva",
            "como_fazer": "1. Divida a cartolina em seções. 2. Desenhe ou cole figuras. 3. Fixe na parede.",
            "como_usar": "O aluno aponta para o cartaz para se comunicar.",
            "para_que_serve": "Facilita a comunicação visual em sala de aula."
        },
        {
            "nome": "Livro de Histórias com Símbolos",
            "materiais": "Caderno, figuras, canetinhas, cola",
            "como_fazer": "1. Crie uma história simples. 2. Desenhe símbolos para palavras-chave. 3. Monte o livro.",
            "como_usar": "Leia a história com o aluno, apontando para os símbolos.",
            "para_que_serve": "Estimula a comunicação e a compreensão de histórias."
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
        },
        {
            "nome": "Tapete Sensorial de Texturas",
            "materiais": "Tapete EVA, tecidos, botões, feltro, cola quente",
            "como_fazer": "1. Corte o tapete em seções. 2. Cole diferentes texturas. 3. Deixe secar.",
            "como_usar": "O aluno caminha ou toca as texturas.",
            "para_que_serve": "Estimula a integração sensorial tátil."
        },
        {
            "nome": "Pote da Ansiedade",
            "materiais": "Pote de vidro, água, glitter, corante, cola quente",
            "como_fazer": "1. Encha com água. 2. Adicione glitter e corante. 3. Feche bem.",
            "como_usar": "Agite e observe o glitter se acalmar.",
            "para_que_serve": "Ajuda na regulação emocional e na ansiedade."
        }
    ],
    "estruturacao": [
        {
            "nome": "Rotina Visual com Caixas",
            "materiais": "Caixas de fósforo, papel, canetinhas",
            "como_fazer": "1. Desenhe as atividades. 2. Coloque em caixas. 3. Organize em sequência.",
            "como_usar": "Mostre a sequência do dia.",
            "para_que_serve": "Dá previsibilidade."
        },
        {
            "nome": "Agenda Visual de Tarefas",
            "materiais": "Papel, canetinhas, velcro",
            "como_fazer": "1. Desenhe as tarefas. 2. Recorte. 3. Cole velcro.",
            "como_usar": "O aluno organiza as tarefas do dia.",
            "para_que_serve": "Ajuda na organização e planejamento."
        },
        {
            "nome": "Calendário Visual de Atividades",
            "materiais": "Papelão, figuras, velcro, canetinhas",
            "como_fazer": "1. Crie um calendário. 2. Desenhe as atividades. 3. Fixe velcro.",
            "como_usar": "O aluno visualiza as atividades do mês.",
            "para_que_serve": "Dá previsibilidade para o mês."
        },
        {
            "nome": "Relógio Visual de Rotina",
            "materiais": "Relógio, canetas coloridas, papel",
            "como_fazer": "1. Desenhe as atividades ao redor do relógio. 2. Associe cores.",
            "como_usar": "O aluno vê a hora da próxima atividade.",
            "para_que_serve": "Ajuda na compreensão do tempo."
        },
        {
            "nome": "Organizador de Tarefas por Cores",
            "materiais": "Caixas coloridas, etiquetas, fita adesiva",
            "como_fazer": "1. Separe cores para cada categoria. 2. Organize as tarefas.",
            "como_usar": "O aluno identifica a cor da tarefa.",
            "para_que_serve": "Facilita a organização e autonomia."
        }
    ],
    "interacao_social": [
        {
            "nome": "História Social Ilustrada",
            "materiais": "Papel, canetinhas, grampeador",
            "como_fazer": "1. Crie uma história. 2. Ilustre. 3. Grampeie.",
            "como_usar": "Leia antes da situação acontecer.",
            "para_que_serve": "Ensina habilidades sociais."
        },
        {
            "nome": "Cartões de Habilidades Sociais",
            "materiais": "Papel cartão, canetinhas, figuras",
            "como_fazer": "1. Desenhe situações sociais. 2. Escreva dicas. 3. Plastifique.",
            "como_usar": "Leia a situação e discuta as respostas.",
            "para_que_serve": "Ensina habilidades sociais e empatia."
        },
        {
            "nome": "Jogo de Faz de Conta Social",
            "materiais": "Fantoches, roupas, cartões com situações",
            "como_fazer": "1. Crie cartões com situações. 2. Prepare os materiais.",
            "como_usar": "Os alunos encenam as situações sociais.",
            "para_que_serve": "Desenvolve habilidades sociais na prática."
        },
        {
            "nome": "Painel de Emoções",
            "materiais": "Papelão, figuras de expressões, velcro",
            "como_fazer": "1. Desenhe expressões faciais. 2. Recorte. 3. Fixe no painel.",
            "como_usar": "O aluno aponta para a expressão que sente.",
            "para_que_serve": "Ensina o reconhecimento de emoções."
        },
        {
            "nome": "Livro de Regras Sociais",
            "materiais": "Caderno, canetinhas, figuras, cola",
            "como_fazer": "1. Crie páginas com regras. 2. Ilustre cada uma. 3. Monte o livro.",
            "como_usar": "Leia e discuta as regras sociais.",
            "para_que_serve": "Ensina regras de convivência."
        }
    ],
    "motor": [
        {
            "nome": "Teclado Adaptado com Papelão",
            "materiais": "Papelão, teclas desenhadas, fita adesiva",
            "como_fazer": "1. Desenhe um teclado em papelão. 2. Recorte as teclas. 3. Fixe com fita.",
            "como_usar": "O aluno usa para digitar ou apontar letras.",
            "para_que_serve": "Auxilia na coordenação motora fina."
        },
        {
            "nome": "Prancha de Atividades Motoras",
            "materiais": "Papelão, clipes, botões, cordas",
            "como_fazer": "1. Crie uma prancha. 2. Inclua abrir clipes, amarrar cordas, encaixar botões.",
            "como_usar": "O aluno realiza as atividades.",
            "para_que_serve": "Desenvolve coordenação motora fina."
        },
        {
            "nome": "Kit de Massinha Caseira",
            "materiais": "Farinha, sal, água, corante",
            "como_fazer": "1. Misture farinha, sal e água. 2. Adicione corante. 3. Amasse.",
            "como_usar": "O aluno modela formas, letras e números.",
            "para_que_serve": "Estimula coordenação motora e criatividade."
        },
        {
            "nome": "Painel de Encaixes",
            "materiais": "Papelão, formas geométricas, tesoura, cola",
            "como_fazer": "1. Desenhe formas no papelão. 2. Recorte os encaixes. 3. Pinte.",
            "como_usar": "O aluno encaixa as formas no lugar correto.",
            "para_que_serve": "Desenvolve coordenação motora e raciocínio lógico."
        },
        {
            "nome": "Prancha de Atividades com Pinças",
            "materiais": "Pinças, objetos pequenos, recipientes, papelão",
            "como_fazer": "1. Crie uma prancha com recipientes. 2. Coloque objetos.",
            "como_usar": "O aluno usa a pinça para transferir objetos.",
            "para_que_serve": "Desenvolve a coordenação motora fina."
        }
    ],
    "cognitivo": [
        {
            "nome": "Jogo da Memória Adaptado",
            "materiais": "Papelão, figuras impressas, cola, plastificador",
            "como_fazer": "1. Imprima pares de figuras. 2. Plastifique. 3. Recorte em cartões.",
            "como_usar": "Encontre os pares.",
            "para_que_serve": "Estimula memória visual e atenção."
        },
        {
            "nome": "Atividades de Raciocínio Lógico",
            "materiais": "Papel, canetinhas, lápis",
            "como_fazer": "1. Crie sequências lógicas. 2. Imprima labirintos.",
            "como_usar": "O aluno realiza as atividades.",
            "para_que_serve": "Desenvolve raciocínio lógico e pensamento crítico."
        },
        {
            "nome": "Painel de Rotina Diária",
            "materiais": "Painel de cortiça, cartões, velcro, canetinhas",
            "como_fazer": "1. Desenhe as atividades em cartões. 2. Plastifique. 3. Organize no painel.",
            "como_usar": "O aluno organiza os cartões no painel.",
            "para_que_serve": "Desenvolve organização e planejamento."
        },
        {
            "nome": "Jogo de Classificação de Objetos",
            "materiais": "Caixas, objetos variados, etiquetas",
            "como_fazer": "1. Separe as caixas por categorias. 2. Cole etiquetas.",
            "como_usar": "O aluno classifica os objetos nas caixas corretas.",
            "para_que_serve": "Desenvolve raciocínio lógico e classificação."
        },
        {
            "nome": "Quebra-Cabeça Adaptado",
            "materiais": "Papelão, figuras, tesoura, cola",
            "como_fazer": "1. Cole uma figura no papelão. 2. Recorte em 4-6 peças.",
            "como_usar": "O aluno monta o quebra-cabeça.",
            "para_que_serve": "Desenvolve percepção visual e coordenação."
        }
    ]
}


# ============================================
# BUSCA ONLINE
# ============================================

def buscar_recursos_online(termo_busca: str, max_resultados: int = 10) -> List[Dict]:
    try:
        from ddgs import DDGS

        queries = [
            f"tecnologia assistiva TEA autismo {termo_busca}",
            f"recursos pedagógicos autismo TEA {termo_busca}",
            f"atividades adaptadas autismo TEA {termo_busca}",
            f"comunicação alternativa autismo TEA {termo_busca}",
            f"materiais adaptados autismo TEA {termo_busca}",
            f"dicas autismo TEA professor {termo_busca}",
            f"tecnologias assistivas baixo custo autismo {termo_busca}",
            f"autismo TEA inclusão escolar recursos {termo_busca}",
            f"apps educativos autismo TEA {termo_busca}",
            f"jogos pedagógicos autismo TEA {termo_busca}"
        ]

        resultados = []
        links_unicos = set()

        for query in queries:
            try:
                with DDGS(timeout=15) as ddgs:
                    for r in ddgs.text(query, region="pt-br", max_results=3):
                        link = r.get("href", "")

                        if not link or link in links_unicos:
                            continue

                        if "youtube" in link or "youtu.be" in link:
                            fonte = "YouTube"
                            icone = "▶️"
                        elif "instagram" in link:
                            fonte = "Instagram"
                            icone = "📸"
                        elif "tiktok" in link:
                            fonte = "TikTok"
                            icone = "🎵"
                        else:
                            fonte = "Site"
                            icone = "🌐"

                        sites_comerciais = ["amazon", "mercadolivre", "shopee", "aliexpress"]
                        if any(site in link.lower() for site in sites_comerciais):
                            continue

                        titulo = r.get("title", "").strip()
                        resumo = r.get("body", "").strip()[:300]

                        palavras_chave = ["autismo", "tea", "inclusão", "pedagógico", "educativo",
                                          "adaptado", "assistiva", "comunicação", "sensorial", "motor"]

                        texto_busca = (titulo + " " + resumo).lower()
                        relevancia = sum(1 for p in palavras_chave if p in texto_busca)

                        if relevancia >= 2:
                            links_unicos.add(link)
                            resultados.append({
                                "titulo": titulo or f"Recurso sobre {termo_busca}",
                                "resumo": resumo or "Recurso encontrado na internet",
                                "link": link,
                                "fonte": fonte,
                                "icone": icone,
                                "relevancia": relevancia
                            })

                            if len(resultados) >= max_resultados:
                                break
            except Exception as e:
                print(f"⚠️ Erro na query '{query}': {e}")
                continue

        resultados.sort(key=lambda x: x.get("relevancia", 0), reverse=True)
        return resultados[:max_resultados]

    except Exception as e:
        print(f"⚠️ Busca online indisponível: {e}")
        return []


def formatar_resultados_online(resultados: List[Dict]) -> str:
    if not resultados:
        return ""

    texto = "\n### 🌐 RECURSOS ENCONTRADOS NA INTERNET\n\n"

    for i, r in enumerate(resultados, 1):
        icone = r.get("icone", "🌐")
        fonte = r.get("fonte", "Site")
        texto += f"**{i}. {icone} {r['titulo']}**\n\n"
        texto += f"{r['resumo']}...\n\n"
        texto += f"🔗 **Fonte:** {fonte}\n"
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
    descricao = solicitacao.descricao_professor.lower()

    def classificar(termos_alta: list, termos_media: list, valor_fornecido: Optional[str] = None) -> str:
        if valor_fornecido is not None:
            if valor_fornecido in [
                "Não verbal", "Limitada", "Repetitivos", "Baixa",
                "Motor Fino", "não verbal", "Nao fala", "não fala"
            ]:
                return "Alta"
            elif valor_fornecido in ["Mista", "Moderada", "Motor Grosso", "Verbal limitada", "Fala pouco", "Média", "Regular"]:
                return "Média"
            else:
                return "Baixa"

        if any(termo in descricao for termo in termos_alta):
            return "Alta"
        elif any(termo in descricao for termo in termos_media):
            return "Média"
        else:
            return "Baixa"

    return {
        "comunicacao": classificar(
            termos_alta=["não fala", "não conversa", "não verbal"],
            termos_media=["fala pouco", "comunica com gestos", "limitada"],
            valor_fornecido=solicitacao.comunicacao
        ),
        "motor": classificar(
            termos_alta=["coordenação", "motor fino", "dificuldade motora", "escrever"],
            termos_media=["coordenação média", "motor regular"],
            valor_fornecido=solicitacao.motor
        ),
        "atencao": classificar(
            termos_alta=["atenção", "concentração", "distração", "hiperativo"],
            termos_media=["atenção média", "concentração média"],
            valor_fornecido=solicitacao.atencao
        ),
        "comportamentos": classificar(
            termos_alta=["repetitivo", "ritualístico", "inflexível", "estereotipia"],
            termos_media=["alguns repetitivos", "flexibilidade média"],
            valor_fornecido=solicitacao.comportamentos
        ),
        "regulacao_sensorial": classificar(
            termos_alta=["barulho", "crise", "sensorial", "sobrecarga"],
            termos_media=["sensibilidade média", "desconforto"],
            valor_fornecido=None
        ),
        "interacao_social": classificar(
            termos_alta=["colegas", "social", "interação", "isolado"],
            termos_media=["interage pouco", "social média"],
            valor_fornecido=None
        ),
        "estruturacao": classificar(
            termos_alta=["rotina", "organiza", "estrutura", "ordem"],
            termos_media=["rotina média", "organização média"],
            valor_fornecido=None
        ),
    }


async def gerar_resposta_llama(prompt: str, max_tokens: int = 200) -> str:
    """Gera resposta usando llama-cpp com GGUF"""
    if not modelo_ok or llm is None:
        return "Modelo indisponível. Use o catálogo de recursos."

    try:
        messages = [
            {"role": "system", "content": "Você é um especialista em Tecnologia Assistiva para TEA no Brasil. Responda APENAS em português do Brasil, de forma clara e prática."},
            {"role": "user", "content": prompt}
        ]

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=False,
        )

        resposta = response["choices"][0]["message"]["content"]
        gc.collect()
        return resposta.strip()

    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        return f"Erro ao gerar resposta: {str(e)}"


def formatar_resposta_fallback(solicitacao: SolicitacaoAnaliseTEA, recursos: List[Dict]) -> str:
    return f"""### ANÁLISE DO CASO

**Baseado no relato:** "{solicitacao.descricao_professor[:100]}..."

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
# ENDPOINTS
# ============================================

@app.post("/analisar-aluno-tea/")
async def analisar_aluno_tea(solicitacao: SolicitacaoAnaliseTEA):
    inicio = time.time()

    categoria = identificar_categoria(solicitacao.descricao_professor)
    recursos = buscar_recursos(categoria, limite=3)

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

    # 🔥 PROMPT MELHORADO - APENAS PORTUGUÊS
    prompt = f"""Você é um especialista em Tecnologias Assistivas para alunos com TEA no Brasil. Responda APENAS em português brasileiro.

Analise o caso abaixo e recomende tecnologias assistivas PRÁTICAS e de BAIXO CUSTO, usando materiais recicláveis.

CASO DO ALUNO:
- Relato: {solicitacao.descricao_professor}
- Idade: {solicitacao.idade_aluno if solicitacao.idade_aluno else "Nao informada"}
- Nível de suporte: {solicitacao.nivel_suporte if solicitacao.nivel_suporte else "Nao informado"}
- Interesses: {interesses_texto}
- Sensibilidades: {sensibilidades_texto}
- Recursos disponíveis: {recursos_texto}

ÁREAS DE ATENÇÃO:
- Comunicação: {solicitacao.comunicacao or 'Nao informado'}
- Motor: {solicitacao.motor or 'Nao informado'}
- Atenção: {solicitacao.atencao or 'Nao informada'}
- Comportamentos: {solicitacao.comportamentos or 'Nao informados'}

REGRAS IMPORTANTES:
1. PROIBIDO recomendar apps, tablets ou tecnologia digital
2. Use APENAS materiais recicláveis e de baixo custo
3. Responda APENAS em português do Brasil

Responda de forma direta e prática:
1. Quais as principais dificuldades?
2. Qual adaptação você recomenda como solução principal?
3. Liste 2 soluções alternativas
4. Dê 3 dicas práticas para o professor"""

    try:
        resposta_ia = await gerar_resposta_llama(prompt, max_tokens=200)
        if len(resposta_ia.split()) > 20 and "Erro" not in resposta_ia:
            analise = f"""### ANÁLISE DO CASO

**Perfil do aluno:** {solicitacao.descricao_professor[:150]}...

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
        "tempo_resposta": f"{round(time.time() - inicio, 2)}s"
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
        "modelo": "SmolLM2-135M (GGUF)",
        "modo": "IA" if modelo_ok else "Catálogo Fixo",
        "arquivo_modelo": MODEL_PATH
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🎓 API de Tecnologias Assistivas para TEA")
    print("=" * 50)
    print(f"📊 Documentação: http://localhost:8000/docs")
    print(f"📝 Endpoint: POST /analisar-aluno-tea/")
    print(f"💻 Modelo: SmolLM2-135M (GGUF)")
    print(f"🔧 Status: {'✅ Carregado' if modelo_ok else '❌ Modo Catálogo'}")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)