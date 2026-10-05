from pathlib import Path
import gc
import shutil
import time

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PASTA_BASE = Path(__file__).resolve().parent / "base"
CAMINHO_DB = PASTA_BASE / "db"
MODELO_EMBEDDING = "nomic-embed-text"
TAMANHO_LOTE = 25


def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)


def carregar_documentos():
    documentos = []

    for arquivo_pdf in PASTA_BASE.rglob("*.pdf"):
        carregador_pdf = PyPDFLoader(str(arquivo_pdf))
        documentos.extend(carregador_pdf.load())

    for arquivo_md in PASTA_BASE.rglob("*.md"):
        if arquivo_md.name.lower() == "readme.md":
            continue
        carregador_md = TextLoader(str(arquivo_md), encoding="utf-8")
        documentos.extend(carregador_md.load())

    print(f"Quantidade de documentos carregados: {len(documentos)}")
    return documentos


def dividir_chunks(documentos):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Quantidade de chunks: {len(chunks)}")
    return chunks


def vetorizar_chunks(chunks):
    if CAMINHO_DB.exists():
        shutil.rmtree(CAMINHO_DB)

    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    db = Chroma(
        persist_directory=str(CAMINHO_DB),
        embedding_function=embeddings,
    )

    for inicio in range(0, len(chunks), TAMANHO_LOTE):
        fim = inicio + TAMANHO_LOTE
        lote = chunks[inicio:fim]
        db.add_documents(lote)
        print(f"Vetorizados {min(fim, len(chunks))} de {len(chunks)} chunks")

    del db
    gc.collect()
    time.sleep(1)

    print("Banco de dados criado com sucesso!")


if __name__ == "__main__":
    criar_db()
