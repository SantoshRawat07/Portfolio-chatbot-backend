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

DATA_DIR = "data"
VECTOR_DIR = "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = []

for file in os.listdir(DATA_DIR):

    path = os.path.join(DATA_DIR, file)

    if file.endswith(".txt"):

        loader = TextLoader(path)
        documents.extend(loader.load())

    elif file.endswith(".pdf"):

        loader = PyPDFLoader(path)
        documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

db = FAISS.from_documents(
    chunks,
    embeddings
)

db.save_local(VECTOR_DIR)

print("Vector DB created")