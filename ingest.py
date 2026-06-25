import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

DATA_DIR = "Data"
VECTOR_DIR = "vector_db"



faiss_file = os.path.join(VECTOR_DIR, "index.faiss")
pkl_file = os.path.join(VECTOR_DIR, "index.pkl")

if os.path.exists(faiss_file) and os.path.exists(pkl_file):
    print("Vector DB already exists.")
    print("Skipping creation.")
    exit()

# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("📦 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# --------------------------------------------------
# Load documents
# --------------------------------------------------

documents = []

print("Loading documents...")

for file in os.listdir(DATA_DIR):

    path = os.path.join(DATA_DIR, file)

    if file.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())

    elif file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

# --------------------------------------------------
# Split documents
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("⚡ Creating vector database...")

db = FAISS.from_documents(
    chunks,
    embeddings
)

# --------------------------------------------------
# Save FAISS DB
# --------------------------------------------------

os.makedirs(VECTOR_DIR, exist_ok=True)

db.save_local(VECTOR_DIR)

print(" Vector DB created successfully!")
print(f"Saved to: {VECTOR_DIR}")