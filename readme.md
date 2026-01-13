# 🛰️ CubeDocs: Assistente de Engenharia Aeroespacial

Uma API baseada em **RAG (Retrieval-Augmented Generation)** projetada para auxiliar engenheiros na consulta de manuais técnicos complexos (como o *CubeSat Design Specification* ou datasheets de componentes). 

O sistema utiliza uma **arquitetura híbrida** para o custo-benefício: processamento vetorial local e inferência generativa na nuvem.

## Arquitetura

Este projeto resolve o problema de alucinação de LLMs em engenharia, restringindo as respostas ao contexto técnico fornecido pelos documentos.

1.  **Ingestão:** O PDF é processado e fragmentado (*Chunking*) usando `RecursiveCharacterTextSplitter`.
2.  **Vetorização (Local):** Utilizamos o modelo `sentence-transformers/all-MiniLM-L6-v2` rodando na CPU (via HuggingFace) para criar embeddings sem custo de API e com privacidade.
3.  **Armazenamento:** Banco vetorial **FAISS** para busca semântica de alta performance na memória RAM.
4.  **Geração (Cloud):** Os fragmentos relevantes são enviados para o **Google Gemini 2.5 Flash Lite** via LangChain LCEL para gerar a resposta final didática.
5.  **Interface:** API RESTful assíncrona construída com **FastAPI**.

## Pra Rodar

### Pré-requisitos
* Python 3.10 ou superior
* Uma chave de API do Google AI Studio

### Instalação

1.  Clone este repositório:
    ```bash
    git clone https://github.com/renanaqn/chat_cube_docs.git
    cd chat_cube_docs
    ```

2.  Crie um ambiente virtual e instale as dependências:
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    
    pip install -r requirements.txt
    ```

3.  Configure a chave de API:
    Crie um arquivo `.env` na raiz do projeto e adicione:
    ```env
    GOOGLE_API_KEY="Sua_Chave_Aqui"
    ```

4.  Inicie o servidor:
    ```bash
    uvicorn main:app --reload
    ```

## Endpoints da API

Acesse a documentação interativa automática (Swagger UI) em `http://localhost:8000/docs`.

* `POST /upload`: Envia um documento técnico para indexação vetorial local.
* `POST /ask`: Envia uma pergunta técnica em linguagem natural e recebe uma resposta contextualizada.

## Stack

* **Linguagem:** Python
* **Framework Web:** FastAPI (Uvicorn)
* **Arquitetura:** LangChain (Sintaxe Moderna LCEL)
* **LLM:** Google Gemini 2.5 Flash
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local)
* **Vector Store:** FAISS CPU

---
