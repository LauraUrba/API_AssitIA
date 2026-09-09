from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
import os
import glob

load_dotenv()

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_BASE = os.path.join(DIR_ATUAL, "..", "dados/pdfs")
os.environ['OLLAMA_MODELS'] = os.path.expanduser('~/assistIA_api_tcc/models/ollama')


def limpar_metadados(metadata):
    """Remove metadados complexos - mantém APENAS tipos simples"""
    limpos = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            limpos[key] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            limpos[key] = value
        elif value is None:
            limpos[key] = None
    return limpos


def limpar_lista_documentos(documentos):
    for doc in documentos:
        doc.metadata = limpar_metadados(doc.metadata)
    return documentos


def criar_db():
    """Recria o banco do ZERO com todos os documentos (PDFs + CSVs). Demorado — use só quando necessário."""
    documentos = carregar_documentos()
    print(documentos)
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)


def carregar_documentos(apenas_csv=False):
    """
    Carrega documentos da PASTA_BASE.
    Se apenas_csv=True, carrega só os CSVs (usado na adição incremental).
    """
    arquivos_pdf = [] if apenas_csv else glob.glob(os.path.join(PASTA_BASE, "*.pdf"))
    arquivos_csv = glob.glob(os.path.join(PASTA_BASE, "*.csv"))
    arquivos = arquivos_pdf + arquivos_csv

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {PASTA_BASE}")

    print(f"Carregando {len(arquivos_pdf)} PDFs e {len(arquivos_csv)} CSVs...")

    carregador_pdf = UnstructuredLoader(arquivos_pdf, mode="single", languages=["por", "eng"]) if arquivos_pdf else None
    documentos = carregador_pdf.load() if carregador_pdf else []

    if arquivos_csv:
        for caminho_csv in arquivos_csv:
            documentos.extend(CSVLoader(caminho_csv, encoding="utf-8").load())

    for doc in documentos:
        doc.metadata = limpar_metadados(doc.metadata)
        if 'languages' not in doc.metadata:
            doc.metadata['languages'] = ['por', 'eng']

    print(f"{len(documentos)} documentos carregados")
    return documentos


def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    for chunk in chunks:
        chunk.metadata = limpar_metadados(chunk.metadata)
    print(len(chunks))
    return chunks


def vetorizar_chunks(chunks):
    """Cria um banco NOVO do zero com os chunks recebidos (sobrescreve o persist_directory)."""
    for chunk in chunks:
        chunk.metadata = limpar_metadados(chunk.metadata)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434"
    )
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="db"
    )
    print(f"Banco salvo com {db._collection.count()} vetores")
    return db


def adicionar_ao_db(apenas_csv=True):
    """
    Adiciona documentos a um banco JÁ EXISTENTE, sem recriar do zero.
    Por padrão só carrega CSVs (caso de uso: você já tinha os PDFs indexados
    e só precisava incluir os CSVs que faltaram).

    Se quiser adicionar PDFs novos também, chame com apenas_csv=False —
    mas nesse caso vai reprocessar TODOS os PDFs da pasta, o que pode
    gerar chunks duplicados se eles já estavam no banco. Prefira colocar
    só os PDFs novos numa pasta separada nesse caso, ou adaptar o filtro.
    """
    documentos = carregar_documentos(apenas_csv=apenas_csv)
    chunks = dividir_chunks(documentos)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434"
    )
    db = Chroma(persist_directory="db", embedding_function=embeddings)  # abre o banco EXISTENTE

    db.add_documents(chunks)
    print(f"Banco atualizado. Total de vetores agora: {db._collection.count()}")
    return db


if __name__ == "__main__":
    import sys

    if "--adicionar" in sys.argv:
        adicionar_ao_db(apenas_csv=True)
    else:
        criar_db()