from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import gc
import re
# from duckduckgo_search import DDGS

# ============================================
# CONFIGURAÇÃO INICIAL DA API
# ============================================

app = FastAPI(
    title="API de Tecnologias Assistivas para TEA - TCC",
    description="Sistema especializado em recomendar tecnologias assistivas para alunos com TEA",
    version="2.0.0"
)

print("🚀 Carregando modelo Qwen2-0.5B...")

ID_MODELO = "Qwen/Qwen2-0.5B-Instruct"

tokenizador = AutoTokenizer.from_pretrained(ID_MODELO)

if tokenizador.pad_token is None:
    tokenizador.pad_token = tokenizador.eos_token

modelo = AutoModelForCausalLM.from_pretrained(
    ID_MODELO,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_cache=True,
)

modelo = AutoModelForCausalLM.from_pretrained(
    ID_MODELO,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_cache=True,
    trust_remote_code=True
)

print("✅ Modelo Qwen2-0.5B carregado com sucesso!")
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
    buscar_online: bool = True  # NOVO: liga/desliga a busca online


class SolicitacaoCatalogoTEA(BaseModel):
    necessidade: str
    tecnologia_recusada: str
    faixa_etaria: Optional[str] = None
    categoria_preferida: Optional[str] = None
    buscar_online: bool = True  # NOVO


# ============================================
# 📚 CATÁLOGO FIXO DE RECURSOS (EXPANDIDO)
# ============================================

CATALOGO_RECURSOS = {
    "comunicacao": [
        {
            "nome": "Prancha de Comunicação com Figuras Recortadas (PECS adaptado)",
            "materiais": "Revistas velhas, tesoura, cola, papelão, velcro ou fita adesiva",
            "como_fazer": "1. Recorte figuras de revistas que representem necessidades básicas (comer, beber, banheiro, brincar). 2. Cole as figuras em quadrados de papelão. 3. Cole velcro atrás de cada figura e numa prancha base.",
            "como_usar": "O aluno aponta ou entrega a figura do que deseja. O professor verbaliza a palavra ao receber a figura, reforçando a fala.",
            "para_que_serve": "Facilita a comunicação não-verbal e reduz a frustração, base do método PECS."
        },
        {
            "nome": "Cartões de Comunicação com Tampinhas",
            "materiais": "Tampinhas de garrafa, papel, canetinhas, cola",
            "como_fazer": "1. Corte círculos de papel do tamanho das tampinhas. 2. Desenhe símbolos simples (água, comida, sim, não). 3. Cole os papéis nas tampinhas.",
            "como_usar": "O aluno entrega a tampinha com o símbolo que representa o que quer.",
            "para_que_serve": "Permite comunicação simples e direta, ótimo para níveis de suporte mais altos."
        },
        {
            "nome": "Quadro de Comunicação por Categorias",
            "materiais": "Cartolina, EVA, figuras impressas ou desenhadas, plástico contact",
            "como_fazer": "1. Divida a cartolina em categorias (alimentos, sentimentos, atividades). 2. Cole as figuras em cada seção. 3. Plastifique com contact para durar mais.",
            "como_usar": "O aluno navega pelas categorias até encontrar o que quer comunicar.",
            "para_que_serve": "Amplia o vocabulário visual disponível para comunicação."
        },
        {
            "nome": "Aplicativo Gratuito de CAA (Comunicação Aumentativa)",
            "materiais": "Smartphone ou tablet (Letme Talk, ou similar gratuito)",
            "como_fazer": "1. Baixe um app gratuito de CAA na loja de aplicativos. 2. Personalize com fotos reais do aluno e do ambiente dele.",
            "como_usar": "O aluno toca nos ícones na tela para formar frases ou pedidos.",
            "para_que_serve": "Tecnologia de média/alta complexidade, com voz sintetizada, mais motivadora para alguns alunos."
        }
    ],
    "regulacao_sensorial": [
        {
            "nome": "Garrafa da Calma (Sensory Bottle)",
            "materiais": "Garrafa PET, água, glitter, corante, cola quente",
            "como_fazer": "1. Encha a garrafa com água até 3/4. 2. Adicione glitter e corante. 3. Feche com cola quente.",
            "como_usar": "Quando o aluno estiver agitado, agite a garrafa e peça para observar o glitter caindo, contando até 10.",
            "para_que_serve": "Ajuda na regulação emocional e reduz crises, dá um foco visual calmante."
        },
        {
            "nome": "Kit Sensorial com Caixas e Tecidos",
            "materiais": "Caixa de sapato, tecidos variados, botões, fitas, esponjas",
            "como_fazer": "1. Forre a caixa com diferentes tecidos (veludo, lixa, algodão). 2. Cole botões e fitas para criar texturas variadas.",
            "como_usar": "Deixe o aluno explorar as texturas livremente em momentos de transição ou ansiedade.",
            "para_que_serve": "Estimula o tato de forma controlada e ajuda na regulação sensorial."
        },
        {
            "nome": "Fone de Ouvido Caseiro (Redutor de Ruído)",
            "materiais": "Fone de ouvido velho ou protetor auricular tipo concha, espuma, tecido",
            "como_fazer": "1. Retire as almofadas do fone. 2. Encha com espuma. 3. Recubra com tecido macio.",
            "como_usar": "O aluno usa em momentos de muito barulho (recreio, eventos, sirenes).",
            "para_que_serve": "Reduz a sobrecarga auditiva, fundamental para alunos com hipersensibilidade a som."
        },
        {
            "nome": "Almofada ou Colete de Peso Sensorial",
            "materiais": "Tecido grosso, arroz ou areia fina, máquina de costura ou cola de tecido",
            "como_fazer": "1. Costure pequenos bolsos num tecido duplo. 2. Encha cada bolso com arroz. 3. Feche bem as bordas.",
            "como_usar": "O aluno coloca no colo ou veste durante atividades que exigem concentração.",
            "para_que_serve": "Pressão profunda ajuda na autorregulação e reduz a ansiedade corporal."
        },
        {
            "nome": "Caixa de Massinha Sensorial Caseira",
            "materiais": "Farinha, sal, óleo, água, corante alimentício",
            "como_fazer": "1. Misture 2 xícaras de farinha, 1 de sal, 2 colheres de óleo e água até dar liga. 2. Adicione corante.",
            "como_usar": "Atividade livre de manipulação, ótima para alunos que gostam de atividades manuais.",
            "para_que_serve": "Trabalha propriocepção e regulação sensorial tátil de forma lúdica."
        }
    ],
    "estruturacao": [
        {
            "nome": "Rotina Visual com Caixas de Fósforo",
            "materiais": "Caixas de fósforo vazias, papel, canetinhas, cola",
            "como_fazer": "1. Desenhe as atividades do dia em tiras de papel pequenas. 2. Coloque cada uma dentro de uma caixinha. 3. Organize as caixas em sequência numa fita ou caixa maior.",
            "como_usar": "Mostre a sequência do dia pela manhã. Ao concluir cada atividade, o aluno retira ou vira a caixa correspondente.",
            "para_que_serve": "Dá previsibilidade ao dia e reduz ansiedade, baseado nos princípios do método TEACCH."
        },
        {
            "nome": "Timer Visual com Garrafa de Areia",
            "materiais": "Garrafa PET pequena, areia fina, funil, cola quente",
            "como_fazer": "1. Encha a garrafa com a quantidade de areia que leva o tempo desejado para escoar (teste antes). 2. Feche bem com cola quente.",
            "como_usar": "Vire a garrafa no início de uma atividade; o aluno vê visualmente o tempo passando.",
            "para_que_serve": "Ajuda na noção abstrata de tempo, que costuma ser difícil para alunos com TEA."
        },
        {
            "nome": "Quadro de Rotina com Velcro",
            "materiais": "Cartolina ou EVA, velcro, figuras das atividades plastificadas",
            "como_fazer": "1. Monte uma coluna vertical na cartolina com velcro. 2. Cole velcro atrás de cada figura de atividade.",
            "como_usar": "Organize a sequência do dia pela manhã junto com o aluno; ele retira a figura ao terminar cada etapa.",
            "para_que_serve": "Estrutura visual reutilizável da rotina diária completa."
        },
        {
            "nome": "Calendário de Transição Hoje/Amanhã",
            "materiais": "Papel cartão, figuras do sol e da lua, velcro",
            "como_fazer": "1. Crie duas colunas: HOJE e AMANHÃ. 2. Use ícones simples para marcar dias especiais (passeio, prova, feriado).",
            "como_usar": "Revise todo dia com o aluno o que muda de hoje para amanhã.",
            "para_que_serve": "Prepara o aluno para mudanças na rotina, reduzindo crises por imprevisibilidade."
        }
    ],
    "interacao_social": [
        {
            "nome": "História Social Ilustrada",
            "materiais": "Papel, canetinhas, grampeador, fotos do próprio aluno se possível",
            "como_fazer": "1. Crie uma história curta sobre uma situação social específica (ex: pedir para brincar). 2. Ilustre cada passo com desenhos simples. 3. Grampeie como um livrinho.",
            "como_usar": "Leia a história com o aluno antes da situação acontecer, repetindo em momentos calmos.",
            "para_que_serve": "Ensina habilidades sociais de forma visual e previsível, método validado cientificamente (Carol Gray)."
        },
        {
            "nome": "Jogo de Cartas de Emoções",
            "materiais": "Cartolina, canetinhas, papel laminado ou contact",
            "como_fazer": "1. Desenhe rostos com expressões diferentes (feliz, triste, bravo, assustado) em cartões. 2. Plastifique para durar.",
            "como_usar": "Use para o aluno apontar como está se sentindo ou identificar emoções de personagens em histórias.",
            "para_que_serve": "Desenvolve reconhecimento e nomeação de emoções, base para interação social."
        },
        {
            "nome": "Roteiro de Brincadeira Estruturada",
            "materiais": "Papel com passos numerados e ilustrados de uma brincadeira simples",
            "como_fazer": "1. Escolha uma brincadeira com regras claras (ex: boliche com garrafas). 2. Desenhe o passo a passo numerado.",
            "como_usar": "Use o roteiro para guiar o aluno e um colega na brincadeira, seguindo a sequência visual.",
            "para_que_serve": "Cria oportunidades estruturadas e previsíveis de interação com colegas."
        }
    ]
}


# ============================================
# 🌐 BUSCA ONLINE COMPLEMENTAR
# ============================================

def buscar_recursos_online(termo_busca: str, max_resultados: int = 3) -> List[Dict]:
    """
    Busca tecnologias assistivas online - compatível com ddgs e duckduckgo-search.
    """
    try:
        # Tenta importar do ddgs primeiro (mais novo)
        try:
            from ddgs import DDGS
            version = "ddgs"
            # 🔥 Para ddgs 9.14.4, podemos usar timeout no construtor
            use_timeout = True
        except ImportError:
            # Fallback para duckduckgo-search
            from duckduckgo_search import DDGS
            version = "duckduckgo-search"
            use_timeout = False

        query = f"tecnologia assistiva TEA autismo {termo_busca}"
        resultados = []

        # 🔥 Se for ddgs, usa timeout; se não, não usa
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

        print(f"Busca online com {version}: {len(resultados)} resultados")
        return resultados

    except ImportError as e:
        print(f"⚠️ Pacote de busca não instalado: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Busca online indisponível: {type(e).__name__}: {e}")
        return []

def formatar_resultados_online(resultados: List[Dict]) -> str:
    """Formata os resultados da busca online em texto legível"""
    if not resultados:
        return ""

    texto = "\n### 🌐 RECURSOS ENCONTRADOS NA INTERNET (complementar)\n\n"
    texto += "*Estes resultados foram buscados online e podem trazer ideias adicionais. Avalie a fonte antes de aplicar.*\n\n"

    for i, r in enumerate(resultados, 1):
        texto += f"**{i}. {r['titulo']}**\n\n"
        texto += f"{r['resumo']}...\n\n"
        texto += f"🔗 Fonte: {r['link']}\n\n"
        texto += "---\n\n"

    return texto


# ============================================
# FUNÇÕES DO CATÁLOGO FIXO
# ============================================

def identificar_categoria(descricao: str) -> str:
    descricao = descricao.lower()

    if any(p in descricao for p in ["comunica", "fala", "conversa", "expressar"]):
        return "comunicacao"
    elif any(p in descricao for p in ["barulho", "auditiva", "sensorial", "som", "crise"]):
        return "regulacao_sensorial"
    elif any(p in descricao for p in ["rotina", "organiza", "estrutura", "ordem"]):
        return "estruturacao"
    elif any(p in descricao for p in ["social", "intera", "colegas", "amigos"]):
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
            termos_alta=["não fala", "não conversa", "não comunica", "sem fala"],
            termos_media=["fala", "conversa", "comunica", "expressar"]
        ),
        "estruturacao": classificar(
            termos_alta=["sem rotina", "não segue rotina", "resiste a mudança"],
            termos_media=["rotina", "organiza", "estrutura", "ordem"]
        ),
        "regulacao_sensorial": classificar(
            termos_alta=["crise", "grita", "sobrecarga", "colapso"],
            termos_media=["barulho", "sensorial", "som", "incomoda"]
        ),
        "interacao_social": classificar(
            termos_alta=["isolado", "não interage", "evita colegas"],
            termos_media=["social", "colegas", "intera", "amigos"]
        ),
    }


# ============================================
# ENDPOINTS
# ============================================

@app.post("/analisar-aluno-tea/")
async def analisar_aluno_tea(solicitacao: SolicitacaoAnaliseTEA):
    """Endpoint para análise de alunos com TEA"""

    categoria = identificar_categoria(solicitacao.descricao_professor)
    recursos = buscar_recursos(categoria, limite=3)

    prompt = f"""Você é um especialista em Tecnologia Assistiva para TEA. NUNCA inicie a resposta com saudações como "Olá" ou "Bom dia". Vá direto à análise técnica.

    CASO DO ALUNO:
    - Relato: {solicitacao.descricao_professor}
    - Idade: {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}
    - Interesses: {solicitacao.interesses_especificos if solicitacao.interesses_especificos else "Não informados"}
    - Sensibilidades: {solicitacao.sensibilidades_sensoriais if solicitacao.sensibilidades_sensoriais else "Não informadas"}
    - Recursos disponíveis na escola/casa: {solicitacao.recursos_disponiveis if solicitacao.recursos_disponiveis else "Não informados"}

    REGRAS IMPORTANTES:
    - Recomende jogos (de tabuleiro, manuais ou digitais) SOMENTE se ajudarem diretamente no desenvolvimento do aluno com TEA, ligados aos interesses dele.
    - Recomende aplicativos de tablet/celular SOMENTE SE os recursos disponíveis mencionarem tablet, celular ou computador. Se não houver menção a esses recursos, NÃO cite nenhum aplicativo.
    - Baseie a solução principal nos materiais que o professor já tem disponíveis.
    - O professor NÃO tem tablet, celular ou computador disponível. Está PROIBIDO sugerir aplicativos, apps ou qualquer tecnologia digital.
    - Use APENAS os materiais físicos listados em "Recursos disponíveis".

    RESPONDA EM PORTUGUÊS, de forma direta, sem introdução, seguindo esta estrutura:

    1. BARREIRAS: Quais as principais dificuldades?

    2. SOLUÇÃO PRINCIPAL: Qual adaptação você recomenda? (priorize os materiais disponíveis)

    3. SOLUÇÕES ALTERNATIVAS: Liste 2 outras opções, incluindo jogos se forem pertinentes

    4. ADAPTAÇÕES NA ROTINA: Como o professor pode adaptar o dia a dia?

    5. DICAS PRÁTICAS: O que o professor pode fazer hoje?"""

    try:
        resposta_ia = await gerar_resposta_async(prompt)
        if len(resposta_ia.split()) > 30:
            analise = formatar_resposta_ia(resposta_ia, solicitacao, recursos)
        else:
            analise = formatar_resposta_fallback(solicitacao, recursos)
    except Exception as e:
        print(f"❌ Erro na IA: {e}")
        analise = formatar_resposta_fallback(solicitacao, recursos)

    # 🌐 Busca online complementar (baseada nos interesses/recursos citados pelo professor)
    if solicitacao.buscar_online:
        termo = solicitacao.interesses_especificos or solicitacao.descricao_professor[:60]
        resultados_online = buscar_recursos_online(termo)
        if resultados_online:
            analise += formatar_resultados_online(resultados_online)

    categorias = classificar_necessidades(solicitacao)

    if "grita" in solicitacao.descricao_professor.lower():
        categorias["regulacao_sensorial"] = "Alta"
    if "não gosta" in solicitacao.descricao_professor.lower():
        categorias["estruturacao"] = "Alta"

    resultado = {
        "analise": analise,
        "categorias_necessidade": categorias,
    }

    if solicitacao.incluir_estruturas:
        estruturas = []
        if categorias["estruturacao"] in ["Alta", "Média"]:
            estruturas.append("rotina_visual")
        if categorias["regulacao_sensorial"] in ["Alta", "Média"]:
            estruturas.append("kit_sensorial")
        if categorias["comunicacao"] in ["Alta", "Média"]:
            estruturas.append("prancha_comunicacao")
        if categorias["interacao_social"] in ["Alta", "Média"]:
            estruturas.append("historias_sociais")

        if "terra" in solicitacao.descricao_professor.lower() or "grama" in solicitacao.descricao_professor.lower():
            estruturas.append("caixa_sensorial_natureza")

        resultado["estruturas_recomendadas"] = estruturas

    return resultado


@app.post("/catalogo-alternativas-tea/")
async def catalogo_tecnologias_tea(solicitacao: SolicitacaoCatalogoTEA):
    """Catálogo de Tecnologias Assistivas para TEA"""

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

    # 🌐 Busca online complementar
    if solicitacao.buscar_online:
        resultados_online = buscar_recursos_online(solicitacao.necessidade)
        if resultados_online:
            catalogo += formatar_resultados_online(resultados_online)

    return {"catalogo": catalogo}


@app.get("/saude/")
async def verificar_saude():
    return {"status": "saudavel", "modelo": ID_MODELO, "dispositivo": str(modelo.device)}


# ============================================
# FUNÇÕES AUXILIARES DE FORMATAÇÃO
# ============================================

def formatar_resposta_ia(resposta: str, solicitacao: SolicitacaoAnaliseTEA, recursos: List[Dict]) -> str:
    if "Olá" in resposta or "Como posso" in resposta:
        return formatar_resposta_fallback(solicitacao, recursos)

    recursos_personalizados = ""
    if solicitacao.interesses_especificos:
        if "terra" in solicitacao.interesses_especificos.lower() or "grama" in solicitacao.interesses_especificos.lower():
            recursos_personalizados += """
#### 💡 RECOMENDAÇÃO ESPECIAL (Baseada nos interesses do aluno)

**Caixa Sensorial de Terra e Natureza**

**📦 Materiais:** Terra, grama, pedras, folhas, recipiente plástico

**🔧 Como fazer:**
1. Coloque terra em um recipiente grande
2. Adicione grama, pedras e folhas
3. Esconda objetos pequenos na terra

**👩‍🏫 Como usar:**
1. Deixe o aluno explorar livremente
2. Peça para ele encontrar objetos escondidos
3. Use como atividade de transição

**🎯 Para que serve:** Aproveita o interesse natural do aluno e promove regulação sensorial
"""

    if solicitacao.recursos_disponiveis:
        recursos_personalizados += f"""
**📋 Recursos disponíveis:** {solicitacao.recursos_disponiveis}
**💡 Dica:** Use o que você já tem em casa ou na escola!
"""

    return f"""### ANÁLISE DO CASO

**Baseado no relato:** "{solicitacao.descricao_professor}"

**Idade:** {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}

---

{resposta}

---

### 🎯 RECURSOS RECOMENDADOS

{formatar_catalogo(recursos)}

{recursos_personalizados}

---

### 💡 DICAS PARA O PROFESSOR

1. **Use os interesses do aluno** como ponto de partida
2. **Observe** o que funciona e ajuste
3. **Comece com um recurso** de cada vez
4. **Reforce positivamente** qualquer tentativa de participação
5. **Comunique-se com a família** sobre o que funciona em casa
"""


def formatar_resposta_fallback(solicitacao: SolicitacaoAnaliseTEA, recursos: List[Dict]) -> str:
    recursos_personalizados = ""
    if solicitacao.interesses_especificos:
        if "terra" in solicitacao.interesses_especificos.lower() or "grama" in solicitacao.interesses_especificos.lower():
            recursos_personalizados += """
#### 💡 RECOMENDAÇÃO ESPECIAL (Baseada nos interesses do aluno)

**Caixa Sensorial de Terra e Natureza**

**📦 Materiais:** Terra, grama, pedras, folhas, recipiente plástico

**🔧 Como fazer:**
1. Coloque terra em um recipiente grande
2. Adicione grama, pedras e folhas
3. Esconda objetos pequenos na terra

**👩‍🏫 Como usar:**
1. Deixe o aluno explorar livremente
2. Peça para ele encontrar objetos escondidos
3. Use como atividade de transição

**🎯 Para que serve:** Aproveita o interesse natural do aluno e promove regulação sensorial
"""

    return f"""### ANÁLISE DO CASO

**Baseado no relato:** "{solicitacao.descricao_professor}"

**Idade:** {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}

---

### 🎯 RECURSOS RECOMENDADOS

{formatar_catalogo(recursos)}

{recursos_personalizados}

---

### 💡 DICAS PARA O PROFESSOR

1. **Use os interesses do aluno** como ponto de partida
2. **Observe** o que funciona e ajuste
3. **Comece com um recurso** de cada vez
4. **Reforce positivamente** qualquer tentativa de participação
5. **Comunique-se com a família** sobre o que funciona em casa
"""


async def gerar_resposta_async(prompt: str, max_tokens: int = 500) -> str:
    """Gera resposta usando o Qwen2.5-0.5B"""
    try:
        mensagens = [
            {"role": "system", "content": "Você é um especialista em Tecnologia Assistiva para TEA. Responda em português de forma clara, prática e direta. Use apenas materiais recicláveis e acessíveis."},
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
            max_length=1024
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
        return ""


# ============================================
# INICIALIZAÇÃO
# ============================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("🎓 API de Tecnologias Assistivas para TEA - Qwen")
    print("=" * 50)
    print(f"📊 Documentação: http://localhost:8000/docs")
    print(f"📝 Endpoint: POST /analisar-aluno-tea/")
    print(f"📚 Catálogo: POST /catalogo-alternativas-tea/")
    print(f"💻 Modelo: {ID_MODELO}")
    print(f"🌐 Busca online: DuckDuckGo (gratuita, sem API key)")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)