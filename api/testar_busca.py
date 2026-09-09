from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

os.environ['OLLAMA_MODELS'] = os.path.expanduser('~/assistIA_api_tcc/models/ollama')

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

print(f"Banco conectado com {db._collection.count()} vetores")

resultados = db.similarity_search("estratégias para alfabetização de autistas", k=3)

print(f"\nEncontrados {len(resultados)} resultados:")
for i, doc in enumerate(resultados):
    print(f"\nResultado {i+1}:")
    print(f"Arquivo: {doc.metadata.get('filename', 'N/A')}")
    print(f"Pagina: {doc.metadata.get('page_number', 'N/A')}")
    print(f"Texto: {doc.page_content[:200]}...")