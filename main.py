from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List 
import shutil
import os
from rag_engine import CubeRAG

app = FastAPI(title="CubeDocs API", description="RAG Aeroespacial")

bot = CubeRAG()

os.makedirs("docs", exist_ok=True)

@app.get("/")
def home():
    return {"status": "Online", "model": "Google Gemini"}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Recebe múltiplos PDFs e treina a IA com eles"""
    saved_paths = []
    
    for file in files:
        file_path = f"docs/{file.filename}"
        
        # Salva cada arquivo
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        saved_paths.append(file_path)
    
    # Envia a lista de caminhos para o motor
    try:
        result = bot.ingest_pdf(saved_paths)
        return {"filenames": [f.filename for f in files], "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(question: str):
    answer = bot.ask(question)
    return {"pergunta": question, "resposta": answer}