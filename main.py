from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import gc
import re

# ============================================
# CONFIGURAÇÃO INICIAL DA API
# ============================================

app = FastAPI(
    title="API de Tecnologias Assistivas para TEA - TCC",
    description="Sistema especializado em recomendar tecnologias assistivas para alunos com TEA",
    version="1.0.0"
)

print("🚀 Carregando modelo Qwen2.5-0.5B...")

ID_MODELO = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizador = AutoTokenizer.from_pretrained(ID_MODELO)

if tokenizador.pad_token is None:
    tokenizador.pad_token = tokenizador.eos_token

modelo = AutoModelForCausalLM.from_pretrained(
    ID_MODELO,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

print("✅ Modelo Qwen2.5-0.5B carregado com sucesso!")
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


class SolicitacaoCatalogoTEA(BaseModel):
    necessidade: str
    tecnologia_recusada: str
    faixa_etaria: Optional[str] = None
    categoria_preferida: Optional[str] = None


# ============================================
# 📚 CATÁLOGO FIXO DE RECURSOS (Plano B)
# ============================================

CATALOGO_RECURSOS = {
    "comunicacao": [
        {
            "nome": "Prancha de Comunicação com Figuras Recortadas",
            "materiais": "Revistas velhas, tesoura, cola, papelão, velcro ou fita adesiva",
            "como_fazer": "1. Recorte figuras de revistas que representem necessidades básicas (comida, água, banheiro, atividades). 2. Cole as figuras em um papelão. 3. Se possível, cole velcro para fixar as figuras.",
            "como_usar": "O aluno aponta para a figura do que deseja. O professor pergunta: 'Você quer X?' e aguarda a resposta (aceno de cabeça, olhar, toque).",
            "para_que_serve": "Facilita a comunicação não-verbal e reduz a frustração por não conseguir se expressar."
        },
        {
            "nome": "Cartões de Comunicação com Tampinhas",
            "materiais": "Tampinhas de garrafa, papel, canetinhas, cola",
            "como_fazer": "1. Corte círculos de papel do tamanho das tampinhas. 2. Desenhe símbolos simples (coração = amor, X = não, check = sim). 3. Cole os papéis nas tampinhas.",
            "como_usar": "O aluno entrega a tampinha com o símbolo que representa o que quer. O professor responde ao símbolo.",
            "para_que_serve": "Permite comunicação simples e direta para crianças que não falam."
        }
    ],
    "regulacao_sensorial": [
        {
            "nome": "Garrafa da Calma (Sensory Bottle)",
            "materiais": "Garrafa PET de 500ml, água, glitter, corante alimentício, cola quente",
            "como_fazer": "1. Encha a garrafa com água até 3/4. 2. Adicione glitter e algumas gotas de corante. 3. Complete com água e feche com cola quente para não abrir.",
            "como_usar": "Quando o aluno estiver agitado, agite a garrafa e peça para ele observar o glitter caindo lentamente. Respirem juntos enquanto observam.",
            "para_que_serve": "Ajuda na regulação emocional, reduz ansiedade e crises."
        },
        {
            "nome": "Kit Sensorial com Caixas e Tecidos",
            "materiais": "Caixa de sapato, tecidos variados (algodão, lã, jeans), botões, fitas",
            "como_fazer": "1. Forre a caixa com diferentes tecidos. 2. Cole botões e fitas para criar texturas. 3. Crie diferentes 'zonas' na caixa (lisa, áspera, macia).",
            "como_usar": "Deixe o aluno explorar as texturas livremente. Pergunte como cada textura faz ele se sentir.",
            "para_que_serve": "Estimula o tato e ajuda na regulação sensorial."
        },
        {
            "nome": "Fone de Ouvido Caseiro (Redutor de Ruído)",
            "materiais": "Fone de ouvido velho, espuma ou algodão, tecido, agulha e linha",
            "como_fazer": "1. Retire as almofadas do fone. 2. Encha com espuma ou algodão para abafar o som. 3. Recubra com tecido e costure.",
            "como_usar": "O aluno usa o fone em momentos de muito barulho (recreio, corredor, atividades barulhentas).",
            "para_que_serve": "Reduz a sobrecarga auditiva e previne crises."
        }
    ],
    "estruturacao": [
        {
            "nome": "Rotina Visual com Caixas de Fósforo",
            "materiais": "Caixas de fósforo vazias, papel, canetinhas, cola",
            "como_fazer": "1. Desenhe ou escreva as atividades do dia em tiras de papel. 2. Coloque cada tira em uma caixa de fósforo. 3. Organize as caixas em sequência.",
            "como_usar": "Mostre ao aluno a sequência do dia. Ao final de cada atividade, retire a caixa e coloque em um 'pote de concluídos'.",
            "para_que_serve": "Dá previsibilidade, reduz ansiedade e ajuda na transição entre atividades."
        },
        {
            "nome": "Timer Visual com Garrafa de Areia Caseira",
            "materiais": "Garrafa PET pequena, areia ou sal, funil, cola quente",
            "como_fazer": "1. Encha a garrafa com areia ou sal. 2. Cronometre quanto tempo leva para cair toda a areia. 3. Feche com cola quente.",
            "como_usar": "Use para marcar o tempo de atividades. 'Vamos fazer essa atividade até a areia acabar.'",
            "para_que_serve": "Ajuda na noção de tempo e na transição entre atividades."
        }
    ],
    "interacao_social": [
        {
            "nome": "História Social Ilustrada",
            "materiais": "Papel, canetinhas, grampeador",
            "como_fazer": "1. Crie uma história simples sobre uma situação social (ex: 'Como brincar com os colegas'). 2. Ilustre cada passo. 3. Grampeie como um livrinho.",
            "como_usar": "Leia a história com o aluno antes da situação acontecer. Use para ensinar comportamentos sociais.",
            "para_que_serve": "Ensina habilidades sociais de forma visual e previsível."
        }
    ]
}


def buscar_recursos_por_categoria(categoria: str, limite: int = 3) -> List[Dict]:
    """Busca recursos no catálogo fixo por categoria"""
    recursos = []

    if categoria.lower() in ["audição", "auditiva", "regulação sensorial", "sensorial"]:
        recursos = CATALOGO_RECURSOS.get("regulacao_sensorial", [])
    elif categoria.lower() in ["comunicação", "comunicacao", "fala"]:
        recursos = CATALOGO_RECURSOS.get("comunicacao", [])
    elif categoria.lower() in ["estruturação", "estruturacao", "rotina"]:
        recursos = CATALOGO_RECURSOS.get("estruturacao", [])
    elif categoria.lower() in ["social", "interação", "interacao"]:
        recursos = CATALOGO_RECURSOS.get("interacao_social", [])
    else:
        # Se não encontrar categoria, mistura todas
        todas = []
        for cat in CATALOGO_RECURSOS.values():
            todas.extend(cat)
        recursos = todas

    return recursos[:limite]


def formatar_catalogo(recursos: List[Dict]) -> str:
    """Formata os recursos em texto legível"""
    if not recursos:
        return "Nenhum recurso encontrado para esta categoria."

    texto = "### RECURSOS PRÁTICOS E ACESSÍVEIS\n\n"

    for i, recurso in enumerate(recursos, 1):
        texto += f"#### {i}. **{recurso['nome']}**\n\n"
        texto += f"**📦 Materiais:** {recurso['materiais']}\n\n"
        texto += f"**🔧 Como fazer:**\n{recurso['como_fazer']}\n\n"
        texto += f"**👆 Como usar:**\n{recurso['como_usar']}\n\n"
        texto += f"**💡 Para que serve:** {recurso['para_que_serve']}\n\n"
        texto += "---\n\n"

    return texto


# ============================================
# ENDPOINTS
# ============================================

@app.post("/analisar-aluno-tea/")
async def analisar_aluno_tea(solicitacao: SolicitacaoAnaliseTEA):
    """
    Endpoint especializado para análise de alunos com TEA
    """

    prompt = f"""Analise o caso abaixo e recomende soluções PRÁTICAS e ACESSÍVEIS para um professor.

⚠️ REGRAS OBRIGATÓRIAS:
- NÃO recomende APPs, softwares, tablets, computadores ou qualquer tecnologia digital
- Use APENAS materiais recicláveis e acessíveis
- Seja específico: diga COMO FAZER passo a passo

CASO DO ALUNO:
- Relato: {solicitacao.descricao_professor}
- Idade: {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}
- Interesses: {solicitacao.interesses_especificos if solicitacao.interesses_especificos else "Não informados"}
- Sensibilidades: {solicitacao.sensibilidades_sensoriais if solicitacao.sensibilidades_sensoriais else "Não informadas"}

RESPONDA DIRETAMENTE:

1. BARRREIRAS: Quais as principais dificuldades?

2. SOLUÇÃO PRINCIPAL: Qual adaptação você recomenda? (Use materiais recicláveis, explique como fazer)

3. SOLUÇÕES ALTERNATIVAS: Liste 2 outras opções com materiais diferentes

4. ADAPTAÇÕES NA ROTINA: Como o professor pode adaptar o dia a dia?

5. DICAS PRÁTICAS: O que o professor pode fazer hoje?"""

    resposta = await gerar_resposta_async(prompt)

    if len(resposta.split()) < 10 or "Olá" in resposta or "Como posso" in resposta:
        prompt_direto = f"""Responda diretamente sobre este caso de aluno com TEA.

⚠️ PROIBIDO recomendar tecnologia digital.
Use APENAS materiais recicláveis: papel, caixas, tampinhas, garrafas, tecidos.

Relato: {solicitacao.descricao_professor}
Idade: {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}

1. Barreiras principais
2. Solução com materiais recicláveis (como fazer passo a passo)
3. Alternativas
4. Adaptações na rotina
5. Dicas práticas"""
        resposta = await gerar_resposta_async(prompt_direto)

    resposta = limpar_resposta_portugues(resposta)

    categorias = classificar_necessidade_tea_avancado(resposta, solicitacao)

    resultado = {
        "analise": resposta,
        "categorias_necessidade": categorias,
    }

    if solicitacao.incluir_estruturas:
        estruturas = []
        if categorias["estruturacao"] in ["Alta", "Média"]:
            estruturas.append("rotina_visual_com_materiais_reciclaveis")
        if categorias["interacao_social"] in ["Alta", "Média"]:
            estruturas.append("historias_sociais_ilustradas")
        if categorias["regulacao_sensorial"] in ["Alta", "Média"]:
            estruturas.append("kit_sensorial_reciclavel")
        if categorias["comunicacao"] in ["Alta", "Média"]:
            estruturas.append("prancha_de_comunicacao_casera")

        resultado["estruturas_recomendadas"] = estruturas

    return resultado


@app.post("/catalogo-alternativas-tea/")
async def catalogo_tecnologias_tea(solicitacao: SolicitacaoCatalogoTEA):
    """
    Catálogo especializado de Tecnologias Assistivas para TEA
    🔥 USANDO CATÁLOGO FIXO (Plano B)
    """

    # 🔥 BUSCA NO CATÁLOGO FIXO
    categoria = solicitacao.categoria_preferida if solicitacao.categoria_preferida else ""

    recursos = buscar_recursos_por_categoria(categoria, limite=3)

    # Se não encontrou recursos, busca da categoria "regulacao_sensorial" (mais relevante)
    if not recursos:
        recursos = buscar_recursos_por_categoria("regulacao_sensorial", limite=3)

    # Se ainda não encontrou, pega todos
    if not recursos:
        todos = []
        for cat in CATALOGO_RECURSOS.values():
            todos.extend(cat)
        recursos = todos[:3]

    catalogo_formatado = formatar_catalogo(recursos)

    return {"catalogo": catalogo_formatado}


@app.get("/saude/")
async def verificar_saude():
    return {
        "status": "saudavel",
        "modelo": ID_MODELO,
        "dispositivo": str(modelo.device)
    }


# ============================================
# FUNÇÃO DE GERAÇÃO - QWEN
# ============================================

async def gerar_resposta_async(prompt: str, max_tokens: int = 900, temperature: float = 0.3) -> str:
    """
    Função para o Qwen2.5-0.5B
    """
    try:
        mensagens = [
            {"role": "system",
             "content": "Você é um especialista em Tecnologia Assistiva para TEA. Responda de forma prática, usando APENAS materiais recicláveis e acessíveis. NUNCA recomende apps, tablets, computadores ou tecnologia digital. Foque em soluções de baixa tecnologia que o professor pode fazer agora. Seja ESPECÍFICO e DETALHADO."},
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
            max_length=2048
        ).to(modelo.device)

        config_geracao = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
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
# FUNÇÃO DE LIMPEZA DE PORTUGUÊS
# ============================================

def limpar_resposta_portugues(texto: str) -> str:
    """
    Limpa e corrige possíveis erros de português na resposta
    """
    if not texto:
        return texto

    # Remove saudações genéricas
    if "Olá!" in texto or "Como posso auxiliar" in texto:
        texto = texto.replace("Olá! Como posso auxiliar você hoje?", "")
        texto = texto.strip()

    # Remove qualquer menção a tecnologia digital
    palavras_remover = [
        r'\bapp\b', r'\bapps\b', r'\bsoftware\b', r'\bdispositivo\b',
        r'\bdispositivos\b', r'\baplicativo\b', r'\baplicativos\b',
        r'\btablet\b', r'\bcomputador\b', r'\bcelular\b', r'\bsmartphone\b',
        r'\bdigital\b', r'\btecnológico\b', r'\btecnológica\b',
        r'\bIA\b', r'\bInteligência Artificial\b'
    ]

    for palavra in palavras_remover:
        texto = re.sub(palavra, '', texto, flags=re.IGNORECASE)

    # Corrige acentuação
    correcoes_acentuacao = {
        r'\bcomunicacao\b': 'comunicação',
        r'\bestruturacao\b': 'estruturação',
        r'\bregulacao\b': 'regulação',
        r'\binteracao\b': 'interação',
        r'\bnecessidade\b': 'necessidade',
        r'\btecnologia\b': 'tecnologia',
        r'\bassistiva\b': 'assistiva',
        r'\bpedagogicas\b': 'pedagógicas',
        r'\badaptacoes\b': 'adaptações',
        r'\bimplementacao\b': 'implementação',
        r'\brecomendacao\b': 'recomendação',
        r'\bavaliacao\b': 'avaliação',
        r'\bsensoriais\b': 'sensoriais',
        r'\bcolegas\b': 'colegas',
        r'\bprofessor\b': 'professor',
        r'\baluno\b': 'aluno',
        r'\brotina\b': 'rotina',
        r'\bvisual\b': 'visual',
        r'\bsocial\b': 'social',
        r'\bbarreiras\b': 'barreiras',
        r'\breciclaveis\b': 'recicláveis',
        r'\breciclavel\b': 'reciclável',
        r'\bmassinha\b': 'massinha',
        r'\bcrise\b': 'crise',
        r'\bcrises\b': 'crises',
        r'\btampinhas\b': 'tampinhas',
        r'\bgarrafas\b': 'garrafas',
        r'\bcaixas\b': 'caixas',
        r'\btecidos\b': 'tecidos',
        r'\bbarbante\b': 'barbante',
        r'\belasticos\b': 'elásticos',
        r'\bpregadores\b': 'pregadores',
    }

    for errado, certo in correcoes_acentuacao.items():
        texto = re.sub(errado, certo, texto, flags=re.IGNORECASE)

    # Remove repetições excessivas
    linhas = texto.split('\n')
    linhas_limpas = []
    ultima_linha = ""

    for linha in linhas:
        if linha.strip() and linha.strip() != ultima_linha:
            linhas_limpas.append(linha)
            ultima_linha = linha.strip()
        elif linha.strip() and len(linha.strip()) > 30:
            linhas_limpas.append(linha)

    texto = '\n'.join(linhas_limpas)
    texto = re.sub(r'\n\s*\n', '\n', texto)

    return texto.strip()


# ============================================
# FUNÇÃO DE CLASSIFICAÇÃO AVANÇADA
# ============================================

def classificar_necessidade_tea_avancado(resposta: str, solicitacao: SolicitacaoAnaliseTEA) -> Dict[str, str]:
    """
    Classifica as áreas de necessidade do aluno com ALTA, MÉDIA ou BAIXA
    """
    resposta_minuscula = resposta.lower()

    # ==========================================
    # 1. COMUNICAÇÃO
    # ==========================================
    comunicacao_score = 0

    palavras_comunicacao = [
        "não conversa", "não fala", "isolado", "isolamento",
        "comunicação", "expressar", "verbal", "diálogo",
        "não interage", "dificuldade de comunicação", "pouca fala",
        "prancha", "gestos"
    ]

    for palavra in palavras_comunicacao:
        if palavra in resposta_minuscula:
            comunicacao_score += 1

    if "não conversa" in solicitacao.descricao_professor.lower():
        comunicacao_score += 2
    if "não fala" in solicitacao.descricao_professor.lower():
        comunicacao_score += 2
    if "não interage" in solicitacao.descricao_professor.lower():
        comunicacao_score += 1

    comunicacao = "Alta" if comunicacao_score >= 4 else "Média" if comunicacao_score >= 2 else "Baixa"

    # ==========================================
    # 2. ESTRUTURAÇÃO
    # ==========================================
    estruturacao_score = 0

    palavras_estruturacao = [
        "rotina", "estrutura", "organização", "previsibilidade",
        "ordem", "planejamento", "visual", "tabela", "agenda",
        "timer", "passo a passo", "sequência"
    ]

    for palavra in palavras_estruturacao:
        if palavra in resposta_minuscula:
            estruturacao_score += 1

    if solicitacao.idade_aluno and solicitacao.idade_aluno <= 6:
        estruturacao_score += 1
    if "crise" in solicitacao.descricao_professor.lower():
        estruturacao_score += 1
    if "rotina" in solicitacao.descricao_professor.lower():
        estruturacao_score += 1

    estruturacao = "Alta" if estruturacao_score >= 4 else "Média" if estruturacao_score >= 2 else "Baixa"

    # ==========================================
    # 3. REGULAÇÃO SENSORIAL
    # ==========================================
    regulacao_score = 0

    palavras_regulacao = [
        "sensorial", "auditiva", "visual", "tátil", "olfativa",
        "barulho", "luz", "toque", "hipersensibilidade", "sobrecarga",
        "fone", "ruído", "sensitivo", "regulação"
    ]

    for palavra in palavras_regulacao:
        if palavra in resposta_minuscula:
            regulacao_score += 1

    if solicitacao.sensibilidades_sensoriais:
        sensibilidades = solicitacao.sensibilidades_sensoriais.lower()
        if "auditiva" in sensibilidades or "barulho" in sensibilidades:
            regulacao_score += 2
        if "visual" in sensibilidades or "luz" in sensibilidades:
            regulacao_score += 1
        if "tátil" in sensibilidades or "toque" in sensibilidades:
            regulacao_score += 1

    if "barulho" in solicitacao.descricao_professor.lower():
        regulacao_score += 2
    if "crise" in solicitacao.descricao_professor.lower():
        regulacao_score += 1

    regulacao = "Alta" if regulacao_score >= 4 else "Média" if regulacao_score >= 2 else "Baixa"

    # ==========================================
    # 4. INTERAÇÃO SOCIAL
    # ==========================================
    interacao_score = 0

    palavras_interacao = [
        "social", "interação", "colegas", "amigos", "grupo",
        "isolamento", "sozinho", "retraído", "timidez",
        "brincar", "compartilhar"
    ]

    for palavra in palavras_interacao:
        if palavra in resposta_minuscula:
            interacao_score += 1

    if "não conversa" in solicitacao.descricao_professor.lower():
        interacao_score += 2
    if "colegas" in solicitacao.descricao_professor.lower() and "não" in solicitacao.descricao_professor.lower():
        interacao_score += 2

    interacao = "Alta" if interacao_score >= 4 else "Média" if interacao_score >= 2 else "Baixa"

    return {
        "comunicacao": comunicacao,
        "estruturacao": estruturacao,
        "regulacao_sensorial": regulacao,
        "interacao_social": interacao
    }


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
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)