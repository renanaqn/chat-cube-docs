import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class CubeRAG:
    def __init__(self):
        self.vector_store = None
        
        # LLM (Google Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0.5,
            convert_system_message_to_human=True
        )

        # Embeddings locais
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Modelo de embeddings pronto!")

    def ingest_pdf(self, pdf_paths: list):
        """Processa múltiplos PDFs e salva na memória"""
        all_splits = []
        
        for path in pdf_paths:
            print(f"Lendo arquivo: {path}")
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Adição na pilha total
                all_splits.extend(splits)
            except Exception as e:
                print(f"Erro ao ler {path}: {e}")

        if not all_splits:
            return {"status": "Erro", "msg": "Nenhum documento válido processado."}

        print(f"Criando banco vetorial com {len(all_splits)} fragmentos totais...")
        
        # Cria a memória vetorial com tudo junto
        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        
        return {
            "status": "Sucesso", 
            "arquivos_processados": len(pdf_paths), 
            "total_chunks": len(all_splits)
        }


    def ask(self, question: str):
        if not self.vector_store:
            return "Erro: Nenhum documento carregado."

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        template = """Você é um Engenheiro Aeroespacial Sênior. 
        Responda à pergunta técnica, de forma didática, usando apenas o contexto abaixo.
        Se não encontrar a resposta, diga "Não encontrei no manual".

        CONTEXTO:
        {context}

        PERGUNTA:
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)