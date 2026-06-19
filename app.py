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

print("\n==============================")
print("Environment Check")
print("==============================")
print("GROQ KEY EXISTS:", bool(GROQ_API_KEY))

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="Personal AI Assistant"
)

# ==================================================
# CORS
# FIX: was pointing to backend URL instead of frontend
# FIX: allow_credentials must be False when allow_origins=["*"]
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# GROQ
# ==================================================

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY not found")

client = Groq(api_key=GROQ_API_KEY)

print("✅ Groq Client Ready")

# ==================================================
# VECTOR DB SETTINGS
# ==================================================

VECTOR_DB_PATH = "vector_db"

_embeddings = None
_retriever = None

# ==================================================
# LAZY LOAD VECTOR DB
# ==================================================
def get_retriever():
    global _embeddings, _retriever

    if _retriever:
        return _retriever

    print("Loading embeddings...")

    _embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )

    emb_dim = len(_embeddings.embed_query("hello"))

    print(f"Embedding Dimension: {emb_dim}")
    print(f"FAISS Dimension: {db.index.d}")

    if emb_dim != db.index.d:
        raise Exception(
            f"FAISS dimension mismatch. "
            f"Embedding={emb_dim}, "
            f"Index={db.index.d}. "
            f"Rebuild vector_db."
        )

    _retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    print("✅ Retriever Ready")

    return _retriever
# ==================================================
# REQUEST MODEL
# ==================================================

class ChatRequest(BaseModel):
    message: str

# ==================================================
# ROOT
# ==================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Personal AI Backend Ready"
    }

# ==================================================
# HEALTH
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq": bool(GROQ_API_KEY)
    }

# ==================================================
# CHAT
# ==================================================

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        ret = get_retriever()

        docs = ret.invoke(req.message)

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
1. Use knowledge when available.
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

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        add_message("user", req.message)
        add_message("assistant", answer)

        return {"answer": answer}

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print("\n❌ CHAT ERROR")
        print(error_msg)
        return {"answer": f"⚠️ Server error: {error_msg}"}

# ==================================================
# STARTUP LOG
# ==================================================

print("\n==============================")
print("Backend Started")
print("==============================")