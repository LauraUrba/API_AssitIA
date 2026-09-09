from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv
import sys
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import do prompt dinâmico
from prompt_assistia import (
    montar_prompt_atividade_unica,
    escolher_interesse_para_area,
    AREA_PRINCIPAL_PARA_AREA,
    _normalizar_lista,
    _normalizar,
)

# CONFIGURAÇÕES INICIAIS

warnings.filterwarnings("ignore")
os.environ['OLLAMA_DEBUG'] = '0'

# Configuração de encoding para evitar problemas com caracteres especiais
if sys.stdin.encoding != 'utf-8':
    sys.stdin.reconfigure(encoding='utf-8', errors='ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

load_dotenv()

CAMINHO_DB = "db"


# MODELOS PYDANTIC PARA A API

class AlunoRequest(BaseModel):
    """Dados do aluno vindos do formulário do site"""
    descricao_aluno: str
    nivel_dsm5: str
    pergunta: str

    # Áreas de atenção (checkboxes)
    areas: List[str] = []  # ex: ["comunicacao", "social", "estrutura"]
    area_principal: Optional[str] = None  # ex: "interacao_social"
    prioridade: str = "media"  # "alta", "media" ou "baixa"

    # Interesses (checkboxes)
    interesses: List[str] = []
    interesses_observacao: str = ""

    # Sensibilidades (checkboxes)
    sensibilidades: List[str] = []
    sensibilidades_observacao: str = ""

    # Recursos (checkboxes)
    recursos: List[str] = []
    recursos_observacao: str = ""


class RespostaAPI(BaseModel):
    """Resposta da API"""
    atividades: str
    tamanho_prompt: int = 0
    sucesso: bool = True
    mensagem: str = ""



# FUNÇÕES AUXILIARES

def get_embedding_function():
    """Retorna a função de embedding"""
    ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=ollama_host
    )


def get_vector_store():
    """Retorna o vector store configurado"""
    return Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=get_embedding_function()
    )


def get_model():
    """Retorna o modelo configurado"""
    ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    return ChatOllama(
        model="llama3.2:latest",
        base_url=ollama_host,
        temperature=0.3,
        num_predict=500,
        #num_ctx=2048
    )


def buscar_base_conhecimento(pergunta: str, limite: float = 0.5, k: int = 5) -> str:
    try:
        # Suprimir saída do Ollama durante a busca
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull), redirect_stdout(devnull):
                db = get_vector_store()
                resultados = db.similarity_search_with_relevance_scores(pergunta, k=k)

        # Filtrar por relevância
        resultados_filtrados = [(doc, score) for doc, score in resultados if score >= limite]

        if not resultados_filtrados:
            return "Nenhum documento específico encontrado. Use conhecimento geral sobre TEA."

        # Pegar os top 3 mais relevantes
        textos = [doc.page_content for doc, _ in resultados_filtrados[:3]]
        return "\n\n----------\n\n".join(textos)

    except Exception as e:
        print(f"[ERRO] Falha na busca: {e}")
        return "Erro ao buscar base de conhecimento. Use conhecimento geral sobre TEA."


def gerar_atividades(request: AlunoRequest) -> RespostaAPI:
    """
    Gera atividades adaptadas para o aluno com TEA

    Args:
        request: Dados do aluno e configurações

    Returns:
        RespostaAPI com as atividades geradas
    """
    try:
        # 1. Buscar base de conhecimento
        base_conhecimento = buscar_base_conhecimento(request.pergunta)

        # 2. Preparar áreas e interesses (normalizados) para o loop
        areas_normalizadas = _normalizar_lista(request.areas)
        area_principal_normalizada = (
            AREA_PRINCIPAL_PARA_AREA.get(_normalizar(request.area_principal))
            if request.area_principal else None
        )

        if not areas_normalizadas:
            return RespostaAPI(
                atividades="Nenhuma área de atenção foi selecionada.",
                tamanho_prompt=0,
                sucesso=False,
                mensagem="areas vazio"
            )

        '''Isso evita que o modelo precise segurar várias áreas/regras ao
        mesmo tempo numa única geração — cada chamada é bem mais curta
        e focada, o que aumenta muito a confiabilidade em modelos
        pequenos rodando em CPU.'''
        modelo = get_model()
        atividades_geradas = []
        atividades_anteriores = []  # NOVO: armazena atividades já geradas para evitar repetição
        tamanho_prompt_total = 0

        for i, area in enumerate(areas_normalizadas):
            interesse_desta_atividade = escolher_interesse_para_area(request.interesses, i)

            # NOVO: passar índices e atividades anteriores para o prompt
            prompt_area = montar_prompt_atividade_unica(
                descricao_aluno=request.descricao_aluno,
                nivel_dsm5=request.nivel_dsm5,
                pergunta=request.pergunta,
                base_conhecimento=base_conhecimento,
                area=area,
                eh_area_principal=(area == area_principal_normalizada),
                prioridade=request.prioridade,
                interesse=interesse_desta_atividade,
                sensibilidades_list=request.sensibilidades,
                sensibilidades_observacao=request.sensibilidades_observacao,
                recursos_list=request.recursos,
                recursos_observacao=request.recursos_observacao,
                indice_atividade=i,  # NOVO
                total_atividades=len(areas_normalizadas),  # NOVO
                atividades_anteriores=atividades_anteriores  # NOVO
            )
            tamanho_prompt_total += len(prompt_area)

            with open(os.devnull, 'w') as devnull:
                with redirect_stderr(devnull), redirect_stdout(devnull):
                    resposta = modelo.invoke(prompt_area)

            texto_atividade = (resposta.content or "").strip()
            if texto_atividade:
                atividades_geradas.append(texto_atividade)
                atividades_anteriores.append(texto_atividade)  # NOVO: guarda para o próximo loop
            else:
                print(f"[AVISO] Área '{area}' retornou resposta vazia — pulando.")

        # NOVO: Validar se as atividades são realmente diferentes
        if len(atividades_geradas) > 1:
            problemas = _validar_atividades_distintas(atividades_geradas)
            if problemas:
                print(f"[AVISO] Possíveis problemas nas atividades: {problemas}")
                # Não falha, apenas avisa - mas podemos tentar regenerar se for muito grave

        atividades = "\n\n---\n\n".join(atividades_geradas)  # Separador mais claro

        # Verificar se a resposta está vazia ou muito curta
        if not atividades or len(atividades.strip()) < 50:
            atividades = (
                "Não foi possível gerar atividades. Tente reformular sua "
                "pergunta ou verifique se o modelo está rodando."
            )
            return RespostaAPI(
                atividades=atividades,
                tamanho_prompt=tamanho_prompt_total,
                sucesso=False,
                mensagem="Resposta gerada com tamanho insuficiente"
            )

        return RespostaAPI(
            atividades=atividades,
            tamanho_prompt=tamanho_prompt_total,
            sucesso=True,
            mensagem=f"{len(atividades_geradas)} atividade(s) geradas com sucesso"
        )

    except Exception as e:
        print(f"[ERRO] Falha na geração: {e}")
        return RespostaAPI(
            atividades=f"Erro ao gerar atividades: {str(e)}",
            tamanho_prompt=0,
            sucesso=False,
            mensagem=str(e)
        )


# NOVA FUNÇÃO: Validar se as atividades são diferentes
def _validar_atividades_distintas(atividades_geradas):
    """
    Verifica se as atividades são realmente diferentes entre si.
    Retorna uma string com os problemas encontrados, ou None se tudo ok.
    """
    if len(atividades_geradas) < 2:
        return None

    problemas = []

    for i in range(len(atividades_geradas)):
        for j in range(i + 1, len(atividades_geradas)):
            # Extrair nomes
            nome_i = _extrair_campo(atividades_geradas[i], 'NOME DA ATIVIDADE')
            nome_j = _extrair_campo(atividades_geradas[j], 'NOME DA ATIVIDADE')

            if nome_i and nome_j:
                # Verificar se os nomes são iguais ou muito parecidos
                nome_i_clean = nome_i.lower().strip()
                nome_j_clean = nome_j.lower().strip()
                if nome_i_clean == nome_j_clean:
                    problemas.append(f"Atividades {i + 1} e {j + 1} têm o MESMO NOME: '{nome_i}'")
                elif len(nome_i_clean) > 3 and len(nome_j_clean) > 3:
                    # Verificar similaridade (se compartilham palavras principais)
                    palavras_i = set(nome_i_clean.split())
                    palavras_j = set(nome_j_clean.split())
                    palavras_comuns = palavras_i & palavras_j
                    if len(palavras_comuns) >= 2:
                        problemas.append(
                            f"Atividades {i + 1} e {j + 1} têm nomes muito parecidos: '{nome_i}' e '{nome_j}'")

            # Extrair materiais
            materiais_i = _extrair_campo(atividades_geradas[i], 'MATERIAIS')
            materiais_j = _extrair_campo(atividades_geradas[j], 'MATERIAIS')

            if materiais_i and materiais_j:
                # Verificar se compartilham muitos materiais
                itens_i = set([item.strip().lower() for item in materiais_i.split(',') if item.strip()])
                itens_j = set([item.strip().lower() for item in materiais_j.split(',') if item.strip()])

                if itens_i and itens_j:
                    intersecao = itens_i & itens_j
                    # Se compartilham mais de 50% dos itens
                    if len(intersecao) / min(len(itens_i), len(itens_j)) > 0.5:
                        problemas.append(f"Atividades {i + 1} e {j + 1} têm materiais muito parecidos")

    if problemas:
        return "; ".join(problemas)
    return None


def _extrair_campo(texto, campo):
    """
    Extrai o valor de um campo no formato "CAMPO: valor" de um texto.
    """
    import re
    # Padrão para capturar o campo até a próxima quebra de linha ou próximo campo
    padrao = rf'{campo}:\s*([^\n]+(?:\n\s+[^\n]+)*)'
    match = re.search(padrao, texto, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# CONFIGURAÇÃO DA API FASTAPI

app = FastAPI(
    title="AssistIA - Assistente para Educação Inclusiva",
    description="API para gerar atividades adaptadas para alunos com TEA",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir requisições do site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ENDPOINTS DA API

@app.post("/gerar-atividades", response_model=RespostaAPI)
async def gerar_atividades_endpoint(request: AlunoRequest):
    """
    Endpoint para gerar atividades adaptadas

    Exemplo de request:
    ```json
    {
        "descricao_aluno": "Idade: 4 anos\\nComunicação: não verbal\\nInteração social: não interage",
        "nivel_dsm5": "2",
        "pergunta": "Como posso ajudar na comunicação e interação social?",
        "areas": ["comunicacao", "social"],
        "area_principal": "interacao_social",
        "prioridade": "alta",
        "interesses": ["massinha"],
        "sensibilidades": ["sons_altos"],
        "recursos": ["reciclaveis"]
    }
    ```
    """
    try:
        # Validações básicas
        if not request.descricao_aluno or len(request.descricao_aluno.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Descrição do aluno é obrigatória e deve ter pelo menos 10 caracteres"
            )

        if not request.pergunta or len(request.pergunta.strip()) < 5:
            raise HTTPException(
                status_code=400,
                detail="Pergunta do professor é obrigatória e deve ter pelo menos 5 caracteres"
            )

        if request.nivel_dsm5 not in ["1", "2", "3"]:
            raise HTTPException(
                status_code=400,
                detail="Nível DSM-5 deve ser 1, 2 ou 3"
            )

        # Gerar atividades
        resultado = gerar_atividades(request)
        return resultado

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERRO] {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde do serviço"""
    return {
        "status": "ok",
        "versao": "2.0.0",
        "servico": "AssistIA"
    }


@app.get("/")
async def root():
    """Endpoint raiz com informações do serviço"""
    return {
        "mensagem": "Bem-vindo ao AssistIA!",
        "documentacao": "/docs",
        "endpoints": {
            "gerar_atividades": "/gerar-atividades (POST)",
            "health": "/health (GET)"
        }
    }


# MODO DE LINHA DE COMANDO (PARA TESTES)

def modo_cli():
    """Modo de linha de comando para testes rápidos"""
    print("\n" + "=" * 60)
    print("ASSISTIA - ASSISTENTE PARA EDUCAÇÃO INCLUSIVA (CLI)")
    print("=" * 60)

    # Coletar dados do usuário de forma interativa
    print("\n[1] DESCRIÇÃO DO ALUNO")
    print("    Exemplo: 'Idade: 4 anos, Comunicação: não verbal, Dificuldades: isolamento'")
    descricao_aluno = input("Descrição: ")

    print("\n[2] NÍVEL DSM-5")
    print("    1 - Nível 1 (Requer suporte)")
    print("    2 - Nível 2 (Requer suporte substancial)")
    print("    3 - Nível 3 (Requer suporte muito substancial)")
    nivel_dsm5 = input("Nível (1/2/3): ")

    print("\n[3] PERGUNTA DO PROFESSOR")
    pergunta = input("Pergunta: ")

    print("\n[4] ÁREAS DE ATENÇÃO (separadas por vírgula)")
    print("    Opções: comunicacao, sensorial, motor, cognitivo, social, estrutura")
    areas_input = input("Áreas: ")
    areas = [a.strip() for a in areas_input.split(",") if a.strip()]

    print("\n[5] INTERESSES DO ALUNO (separados por vírgula)")
    print("    Opções: musica, instrumentos, desenhos, pintura, massinha, dinossauros, "
          "animais, natureza, leitura, jogos, tecnologia, esportes")
    interesses_input = input("Interesses: ")
    interesses = [i.strip() for i in interesses_input.split(",") if i.strip()]

    print("\n[6] SENSIBILIDADES (separadas por vírgula)")
    print("    Opções: sons_altos, luzes_fortes, texturas, multidoes, cheiros, "
          "toque, movimento, comidas, rotina, temperatura")
    sensibilidades_input = input("Sensibilidades: ")
    sensibilidades = [s.strip() for s in sensibilidades_input.split(",") if s.strip()]

    print("\n[7] RECURSOS DISPONÍVEIS (separados por vírgula)")
    print("    Opções: tablet, computador, impressora, lousa_digital, reciclaveis, "
          "livros, jogos_educativos, instrumentos_musicais, arte, esporte")
    recursos_input = input("Recursos: ")
    recursos = [r.strip() for r in recursos_input.split(",") if r.strip()]

    # Criar request
    request = AlunoRequest(
        descricao_aluno=descricao_aluno,
        nivel_dsm5=nivel_dsm5,
        pergunta=pergunta,
        areas=areas,
        area_principal=areas[0] if areas else None,
        prioridade="media",
        interesses=interesses,
        sensibilidades=sensibilidades,
        recursos=recursos
    )

    # Gerar atividades
    print("\n" + "=" * 60)
    print(f"GERANDO ATIVIDADES... ({len(areas)} atividade(s), uma chamada por área)")
    print("=" * 60 + "\n")

    resultado = gerar_atividades(request)

    print("=" * 60)
    print("ATIVIDADES GERADAS:")
    print("=" * 60)
    print(resultado.atividades)
    print("\n" + "=" * 60)
    print(f"STATUS: {'✅ Sucesso' if resultado.sucesso else '❌ Falha'}")
    print(f"MENSAGEM: {resultado.mensagem}")
    print(f"TAMANHO DO PROMPT: {resultado.tamanho_prompt} caracteres")
    print("=" * 60)


# PONTO DE ENTRADA

if __name__ == "__main__":
    # Verificar se quer rodar como API ou CLI
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        print("=" * 60)
        print("INICIANDO API ASSISTIA")
        print("=" * 60)
        print("Documentação: http://localhost:8000/docs")
        print("Health Check: http://localhost:8000/health")
        print("=" * 60)
        uvicorn.run(app, host="0.0.0.0", port=8003)
    else:
        modo_cli()