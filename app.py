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

from huggingface_hub import login

from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

# ==================================================
# ENV
# ==================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

print("\n==============================")
print("Environment Check")
print("==============================")

print("GROQ KEY EXISTS:", bool(GROQ_API_KEY))
print("HF TOKEN EXISTS:", bool(HF_TOKEN))

# ==================================================
# HUGGING FACE LOGIN
# ==================================================

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)

        print("✅ HuggingFace Login Success")

    except Exception as e:
        print(
            f"⚠️ HuggingFace Login Failed: {e}"
        )
else:
    print(
        "⚠️ HF_TOKEN not found"
    )

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="Personal AI Assistant"
)

# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# GROQ
# ==================================================

if not GROQ_API_KEY:
    raise Exception(
        "GROQ_API_KEY not found"
    )

client = Groq(
    api_key=GROQ_API_KEY
)

print("✅ Groq Client Ready")

# ==================================================
# EMBEDDINGS
# ==================================================

print("\nLoading Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

print("✅ Embedding Model Loaded")

# ==================================================
# LOAD VECTOR DB
# ==================================================

VECTOR_DB_PATH = "vector_db"

print("\nLoading Vector DB...")

if not os.path.exists(VECTOR_DB_PATH):

    raise Exception(
        f"Vector DB not found: {VECTOR_DB_PATH}"
    )

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_kwargs={
        "k": 10
    }
)

print("✅ Vector DB Loaded")

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
        "groq": bool(GROQ_API_KEY),
        "hf_token": bool(HF_TOKEN)
    }

# ==================================================
# CHAT
# ==================================================

@app.post("/chat")
def chat(req: ChatRequest):

    try:

        # --------------------------------------
        # VECTOR SEARCH
        # --------------------------------------

        docs = retriever.invoke(
            req.message
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # --------------------------------------
        # MEMORY
        # --------------------------------------

        memory = load_memory()

        history = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in memory
        )

        # --------------------------------------
        # PROMPT
        # --------------------------------------

        prompt = f"""
You are Santosh's Personal AI Assistant.

Use the supplied knowledge.

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

        # --------------------------------------
        # GROQ
        # --------------------------------------

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1500
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        # --------------------------------------
        # SAVE MEMORY
        # --------------------------------------

        add_message(
            "user",
            req.message
        )

        add_message(
            "assistant",
            answer
        )

        return {
            "answer": answer
        }

    except Exception as e:

        print(
            "\n❌ CHAT ERROR:"
        )

        print(e)

        return {
            "answer": str(e)
        }

# ==================================================
# STARTUP LOG
# ==================================================

print("\n==============================")
print("Backend Ready")
print("==============================")