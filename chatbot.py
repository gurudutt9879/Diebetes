from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ollama 
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from fastapi.responses import HTMLResponse

def search(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = PyPDFLoader("Diabetes.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

    chunks = splitter.split_documents(docs)
    db = Chroma.from_documents(chunks, embeddings)
    print("Chunks:", len(chunks))

    results = db.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])

def ask(question):
    context = search(question)
    prompt = (
    "you need to respond to greetings and You are a professional medical assistant chatbot.\n\n"
    "Rules:\n"
    "- Answer clearly in simple language\n"
    "- Keep it short (3–5 lines max)\n"
    "- Use bullet points if helpful\n"
    "- Only use the given context\n"
    "- If not found, say exactly: Not found in document\n\n"
    "Context:\n" + context + "\n\n"
    "Question:\n" + question + "\n\n"
    "Answer:\n"
    )   

    response = ollama.chat(model="gpt-oss:120b-cloud", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
def chat(data: dict):
    question = data["question"]
    return {"answer": ask(question)}

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def guru():
   with open("index.html", "r") as f:
        return f.read()