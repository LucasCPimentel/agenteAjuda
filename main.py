from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

PASTA_BASE = Path(__file__).resolve().parent / "base"
CAMINHO_DB = PASTA_BASE / "db"
MODELO_EMBEDDING = "nomic-embed-text"
MODELO_CHAT = "llama3.2"
QUANTIDADE_RESULTADOS = 5
SCORE_MINIMO = 0.4
MAX_HISTORICO = 6

prompt_template = """
Voce e um assistente de estudos especializado em IA, RAG, arquitetura e n8n.

Responda usando somente a base de conhecimento abaixo.
Use o historico da conversa apenas para entender perguntas de continuacao.
Se a base estiver em ingles, traduza e sintetize a resposta em portugues.
Se a base realmente nao tiver informacao suficiente, diga:
"Desculpe, nao encontrei essa resposta na base."

Historico da conversa:
{historico}

Pergunta atual:
{pergunta}

Base de conhecimento:
{base_de_conhecimento}
"""


def formatar_historico(historico):
    if not historico:
        return "Sem historico ainda."

    mensagens = []
    for item in historico[-MAX_HISTORICO:]:
        mensagens.append(f"Usuario: {item['pergunta']}")
        mensagens.append(f"Assistente: {item['resposta']}")
    return "\n".join(mensagens)


def montar_consulta_para_busca(pergunta, historico):
    if not historico:
        return pergunta

    ultimas_perguntas = [item["pergunta"] for item in historico[-2:]]
    contexto_curto = " ".join(ultimas_perguntas)
    return f"{contexto_curto} {pergunta}"


def buscar_contexto(db, pergunta, historico):
    consulta = montar_consulta_para_busca(pergunta, historico)
    resultados = db.similarity_search_with_relevance_scores(
        consulta,
        k=QUANTIDADE_RESULTADOS,
    )
    return resultados


def montar_base_de_conhecimento(resultados):
    textos_resultados = []
    for documento, _score in resultados:
        textos_resultados.append(documento.page_content)
    return "\n\n----\n\n".join(textos_resultados)


def mostrar_fontes(resultados):
    print("\nFontes usadas:")
    for documento, score in resultados:
        fonte = Path(documento.metadata.get("source", "fonte desconhecida")).name
        pagina = documento.metadata.get("page")
        pagina_texto = f", pagina {pagina + 1}" if pagina is not None else ""
        print(f"- {fonte}{pagina_texto} | score: {score:.3f}")


def responder(pergunta, historico, db, modelo):
    resultados = buscar_contexto(db, pergunta, historico)
    if len(resultados) == 0 or resultados[0][1] < SCORE_MINIMO:
        resposta = "Desculpe, nao encontrei essa resposta na base."
        print(resposta)
        mostrar_fontes(resultados)
        historico.append({"pergunta": pergunta, "resposta": resposta})
        return

    base_de_conhecimento = montar_base_de_conhecimento(resultados)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_formatado = prompt.invoke(
        {
            "historico": formatar_historico(historico),
            "pergunta": pergunta,
            "base_de_conhecimento": base_de_conhecimento,
        }
    )

    resposta = modelo.invoke(prompt_formatado).content
    print(resposta)
    mostrar_fontes(resultados)
    historico.append({"pergunta": pergunta, "resposta": resposta})


def iniciar_agente():
    funcao_embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    db = Chroma(
        persist_directory=str(CAMINHO_DB),
        embedding_function=funcao_embeddings,
    )
    modelo = ChatOllama(model=MODELO_CHAT)
    historico = []

    print("Agente iniciado. Digite 'sair' para encerrar ou 'limpar' para apagar a memoria.")

    while True:
        pergunta = input("\nDigite sua pergunta: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Conversa encerrada.")
            break
        if pergunta.lower() == "limpar":
            historico.clear()
            print("Memoria da conversa apagada.")
            continue
        if not pergunta:
            continue

        responder(pergunta, historico, db, modelo)


if __name__ == "__main__":
    iniciar_agente()
