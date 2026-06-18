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

print("🚀 Carregando modelo leve para o Render Free...")

# 🔥 MODELO MAIS LEVE PARA O RENDER FREE
ID_MODELO = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizador = AutoTokenizer.from_pretrained(ID_MODELO)

if tokenizador.pad_token is None:
    tokenizador.pad_token = tokenizador.eos_token

modelo = AutoModelForCausalLM.from_pretrained(
    ID_MODELO,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

print("✅ Modelo carregado com sucesso!")
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
# 📚 CATÁLOGO FIXO DE RECURSOS (PLANO B)
# ============================================

CATALOGO_RECURSOS = {
    "comunicacao": [
        {
            "nome": "Prancha de Comunicação com Figuras Recortadas",
            "materiais": "Revistas velhas, tesoura, cola, papelão, velcro ou fita adesiva",
            "como_fazer": "1. Recorte figuras de revistas que representem necessidades básicas. 2. Cole as figuras em um papelão. 3. Se possível, cole velcro para fixar.",
            "como_usar": "O aluno aponta para a figura do que deseja. O professor pergunta e aguarda a resposta.",
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
            "como_fazer": "1. Retire as almofadas do fone. 2. Encha com espuma. 3. Recubra com tecido.",
            "como_usar": "O aluno usa em momentos de muito barulho.",
            "para_que_serve": "Reduz a sobrecarga auditiva."
        }
    ],
    "estruturacao": [
        {
            "nome": "Rotina Visual com Caixas de Fósforo",
            "materiais": "Caixas de fósforo vazias, papel, canetinhas, cola",
            "como_fazer": "1. Desenhe as atividades em tiras de papel. 2. Coloque em caixas. 3. Organize em sequência.",
            "como_usar": "Mostre a sequência do dia. Ao final de cada atividade, retire a caixa.",
            "para_que_serve": "Dá previsibilidade e reduz ansiedade."
        },
        {
            "nome": "Timer Visual com Garrafa de Areia",
            "materiais": "Garrafa PET, areia, funil, cola quente",
            "como_fazer": "1. Encha a garrafa com areia. 2. Cronometre o tempo. 3. Feche com cola quente.",
            "como_usar": "Use para marcar o tempo de atividades.",
            "para_que_serve": "Ajuda na noção de tempo."
        }
    ],
    "interacao_social": [
        {
            "nome": "História Social Ilustrada",
            "materiais": "Papel, canetinhas, grampeador",
            "como_fazer": "1. Crie uma história sobre uma situação social. 2. Ilustre cada passo. 3. Grampeie como um livrinho.",
            "como_usar": "Leia a história com o aluno antes da situação acontecer.",
            "para_que_serve": "Ensina habilidades sociais de forma visual."
        }
    ]
}


# ============================================
# FUNÇÕES DO CATÁLOGO
# ============================================

def identificar_categoria(descricao: str) -> str:
    """Identifica a categoria principal baseado na descrição"""
    descricao = descricao.lower()

    if any(p in descricao for p in ["comunica", "fala", "conversa", "expressar"]):
        return "comunicacao"
    elif any(p in descricao for p in ["barulho", "auditiva", "sensorial", "som", "ruido"]):
        return "regulacao_sensorial"
    elif any(p in descricao for p in ["rotina", "organiza", "estrutura", "ordem"]):
        return "estruturacao"
    elif any(p in descricao for p in ["social", "intera", "colegas", "amigos"]):
        return "interacao_social"
    else:
        return "regulacao_sensorial"  # Fallback


def buscar_recursos_por_categoria(categoria: str, limite: int = 3) -> List[Dict]:
    """Busca recursos no catálogo fixo por categoria"""
    if categoria in CATALOGO_RECURSOS:
        return CATALOGO_RECURSOS[categoria][:limite]

    # Fallback: pega de todas as categorias
    todos = []
    for cat in CATALOGO_RECURSOS.values():
        todos.extend(cat)
    return todos[:limite]


def formatar_catalogo(recursos: List[Dict]) -> str:
    """Formata os recursos em texto legível"""
    if not recursos:
        return "Nenhum recurso encontrado para esta categoria."

    texto = "### RECURSOS PRÁTICOS E ACESSÍVEIS\n\n"

    for i, recurso in enumerate(recursos, 1):
        texto += f"#### {i}. **{recurso['nome']}**\n\n"
        texto += f"**📦 Materiais:** {recurso['materiais']}\n\n"
        texto += f"**🔧 Como fazer:**\n{recurso['como_fazer']}\n\n"
        texto += f"**👩‍🏫 Como usar:**\n{recurso['como_usar']}\n\n"
        texto += f"**🎯 Para que serve:** {recurso['para_que_serve']}\n\n"
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

    # 🔥 IDENTIFICA A CATEGORIA PRINCIPAL
    categoria = identificar_categoria(solicitacao.descricao_professor)

    # 🔥 BUSCA RECURSOS DO CATÁLOGO FIXO
    recursos = buscar_recursos_por_categoria(categoria, limite=3)

    # 🔥 USA O MODELO APENAS PARA COMPLEMENTAR (se possível)
    prompt = f"""Analise o caso e classifique as necessidades do aluno.

CASO:
- {solicitacao.descricao_professor}
- Idade: {solicitacao.idade_aluno if solicitacao.idade_aluno else "Não informada"}

Classifique cada área como ALTA, MÉDIA ou BAIXA:

1. Comunicação
2. Estruturação
3. Regulação Sensorial
4. Interação Social

Responda apenas com as classificações."""

    try:
        classificacao = await gerar_resposta_async(prompt, max_tokens=100)
    except:
        classificacao = ""

    # 🔥 CRIA A RESPOSTA FINAL COMBINANDO CATÁLOGO + CLASSIFICAÇÃO
    analise = formatar_catalogo(recursos)

    # Adiciona uma introdução personalizada
    introducao = f"""### ANÁLISE DO CASO

**Baseado no relato:** "{solicitacao.descricao_professor}"

**Categoria identificada:** {categoria.replace('_', ' ').title()}

**Recursos recomendados para este caso:**

"""

    # Se o modelo conseguiu classificar, adiciona
    if classificacao and len(classificacao) > 10:
        analise = introducao + analise + f"\n\n### CLASSIFICAÇÃO DO MODELO\n{classificacao}"
    else:
        analise = introducao + analise

    # Classificação baseada na categoria
    categorias = {
        "comunicacao": "Média",
        "estruturacao": "Média",
        "regulacao_sensorial": "Média",
        "interacao_social": "Média"
    }

    # Ajusta com base na categoria identificada
    if categoria == "comunicacao":
        categorias["comunicacao"] = "Alta"
    elif categoria == "regulacao_sensorial":
        categorias["regulacao_sensorial"] = "Alta"
    elif categoria == "estruturacao":
        categorias["estruturacao"] = "Alta"
    elif categoria == "interacao_social":
        categorias["interacao_social"] = "Alta"

    resultado = {
        "analise": analise,
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
    """Catálogo especializado de Tecnologias Assistivas para TEA"""

    categoria = solicitacao.categoria_preferida if solicitacao.categoria_preferida else ""

    if categoria:
        recursos = buscar_recursos_por_categoria(categoria, limite=3)
    else:
        # Busca por palavra-chave na necessidade
        if "comunica" in solicitacao.necessidade.lower():
            recursos = buscar_recursos_por_categoria("comunicacao", limite=3)
        elif "barulho" in solicitacao.necessidade.lower() or "sensorial" in solicitacao.necessidade.lower():
            recursos = buscar_recursos_por_categoria("regulacao_sensorial", limite=3)
        elif "rotina" in solicitacao.necessidade.lower():
            recursos = buscar_recursos_por_categoria("estruturacao", limite=3)
        else:
            recursos = buscar_recursos_por_categoria("regulacao_sensorial", limite=3)

    catalogo_formatado = formatar_catalogo(recursos)

    # Adiciona cabeçalho
    catalogo_completo = f"""### CATÁLOGO DE TECNOLOGIAS ASSISTIVAS PARA TEA

**Necessidade:** {solicitacao.necessidade}

{catalogo_formatado}

---
*Recursos práticos que o professor pode implementar imediatamente com materiais recicláveis.*
"""

    return {"catalogo": catalogo_completo}


@app.get("/saude/")
async def verificar_saude():
    return {
        "status": "saudavel",
        "modelo": ID_MODELO,
        "dispositivo": str(modelo.device)
    }


# ============================================
# FUNÇÃO DE GERAÇÃO
# ============================================

async def gerar_resposta_async(prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
    """
    Função para o SmolLM2-135M
    """
    try:
        mensagens = [
            {"role": "system", "content": "Você é um especialista em TEA. Responda de forma curta e direta."},
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
            temperature=temperature,
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
    print("🎓 API de Tecnologias Assistivas para TEA - Render Free")
    print("=" * 50)
    print(f"📊 Documentação: http://localhost:8000/docs")
    print(f"📝 Endpoint: POST /analisar-aluno-tea/")
    print(f"📚 Catálogo: POST /catalogo-alternativas-tea/")
    print(f"💻 Modelo: {ID_MODELO}")
    print(f"📦 Modo: Catálogo Fixo + IA Leve")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)