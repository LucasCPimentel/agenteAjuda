# Agente RAG de Estudos

Projeto de estudos para criar um agente com RAG usando Python, LangChain, Chroma,
Ollama e Streamlit.

## Base de conhecimento

A pasta `base/` esta separada por temas:

- `rag/`
- `ia-geral/`
- `arquitetura/`
- `n8n/`

O banco vetorial fica em `base/db/`, mas ele nao vai para o GitHub porque e gerado
localmente com `criar_db.py`.

## Requisitos

- Python 3.14+
- Ollama instalado
- Modelos Ollama:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Instalar dependencias

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Criar banco vetorial

```powershell
python criar_db.py
```

## Rodar no terminal

```powershell
python main.py
```

## Rodar interface Streamlit

```powershell
python -m streamlit run app.py
```

Se o Streamlit pedir email:

```powershell
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
python -m streamlit run app.py
```
