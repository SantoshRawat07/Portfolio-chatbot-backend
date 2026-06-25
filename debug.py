from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

# print the dimension stored in the index
print("FAISS index dimension:", db.index.d)

# print the dimension your current embedding model produces
test = embeddings.embed_query("test")
print("Embedding model dimension:", len(test))