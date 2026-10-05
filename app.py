from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

PASTA_BASE = Path(__file__).resolve().parent / "base"
CAMINHO_DB = PASTA_BASE / "db"
MODELO_EMBEDDING = "nomic-embed-text"
MODELO_CHAT = "llama3.2"
MAX_HISTORICO = 6

PROMPT_TEMPLATE = """
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


@st.cache_resource
def carregar_db():
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    return Chroma(
        persist_directory=str(CAMINHO_DB),
        embedding_function=embeddings,
    )


@st.cache_resource
def carregar_modelo():
    return ChatOllama(model=MODELO_CHAT)


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


def montar_base_de_conhecimento(resultados):
    return "\n\n----\n\n".join(documento.page_content for documento, _score in resultados)


def formatar_fontes(resultados):
    fontes = []
    for documento, score in resultados:
        fonte = Path(documento.metadata.get("source", "fonte desconhecida")).name
        pagina = documento.metadata.get("page")
        pagina_texto = f", pagina {pagina + 1}" if pagina is not None else ""
        fontes.append(
            {
                "fonte": f"{fonte}{pagina_texto}",
                "score": score,
            }
        )
    return fontes


def gerar_resposta(pergunta, historico, quantidade_resultados, score_minimo):
    db = carregar_db()
    modelo = carregar_modelo()

    consulta = montar_consulta_para_busca(pergunta, historico)
    resultados = db.similarity_search_with_relevance_scores(
        consulta,
        k=quantidade_resultados,
    )

    fontes = formatar_fontes(resultados)
    if len(resultados) == 0 or resultados[0][1] < score_minimo:
        return "Desculpe, nao encontrei essa resposta na base.", fontes

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_formatado = prompt.invoke(
        {
            "historico": formatar_historico(historico),
            "pergunta": pergunta,
            "base_de_conhecimento": montar_base_de_conhecimento(resultados),
        }
    )
    resposta = modelo.invoke(prompt_formatado).content
    return resposta, fontes


st.set_page_config(page_title="Agente RAG", page_icon="AI", layout="wide")

st.title("Agente RAG de Estudos")
st.caption("Base local com RAG, IA geral, arquitetura e n8n.")

with st.sidebar:
    st.header("Configuracoes")
    quantidade_resultados = st.slider("Resultados recuperados", 3, 10, 5)
    score_minimo = st.slider("Score minimo", 0.0, 1.0, 0.4, 0.05)

    if st.button("Limpar memoria"):
        st.session_state.mensagens = []
        st.session_state.historico = []
        st.rerun()

    st.divider()
    st.subheader("Base")
    st.write(f"Modelo chat: `{MODELO_CHAT}`")
    st.write(f"Embeddings: `{MODELO_EMBEDDING}`")
    st.write(f"Banco: `{CAMINHO_DB}`")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "historico" not in st.session_state:
    st.session_state.historico = []

for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])
        if mensagem.get("fontes"):
            with st.expander("Fontes usadas"):
                for item in mensagem["fontes"]:
                    st.write(f"- {item['fonte']} | score: {item['score']:.3f}")

pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a base e gerando resposta..."):
            resposta, fontes = gerar_resposta(
                pergunta,
                st.session_state.historico,
                quantidade_resultados,
                score_minimo,
            )
        st.markdown(resposta)
        with st.expander("Fontes usadas"):
            for item in fontes:
                st.write(f"- {item['fonte']} | score: {item['score']:.3f}")

    st.session_state.mensagens.append(
        {
            "role": "assistant",
            "content": resposta,
            "fontes": fontes,
        }
    )
    st.session_state.historico.append(
        {
            "pergunta": pergunta,
            "resposta": resposta,
        }
    )
