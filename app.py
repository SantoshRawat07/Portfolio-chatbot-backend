import os

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq import Groq

from memory import (
    load_memory,
    add_message
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==================================================
# ENV
# ==================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY not found")

client = Groq(api_key=GROQ_API_KEY)

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(title="Personal AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# VECTOR DB
# ==================================================

VECTOR_DB_PATH = "vector_db"

if not os.path.exists(f"{VECTOR_DB_PATH}/index.faiss"):
    raise Exception(
        "vector_db not found. Run ingest.py first."
    )

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Loading FAISS index...")

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

print("✅ Retriever Ready")

# ==================================================
# REQUEST MODEL
# ==================================================

class ChatRequest(BaseModel):
    message: str

# ==================================================
# ROUTES
# ==================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Personal AI Backend Ready"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/chat")
def chat(req: ChatRequest):
    try:

        docs = retriever.invoke(req.message)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        memory = load_memory()

        history = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in memory
        )

        prompt = f"""
You are Santosh's Personal AI Assistant.

Conversation History:
{history}

Knowledge:
{context}

Question:
{req.message}

Rules:
1. Use retrieved knowledge.
2. Give concise answers.
3. Do not invent facts.
4. If information is unavailable, say:
   I don't have that information.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        add_message("user", req.message)
        add_message("assistant", answer)

        return {
            "answer": answer
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}"
        }

print("✅ Backend Started")