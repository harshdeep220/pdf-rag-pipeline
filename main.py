import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import ollama
import numpy as np


# Configs
PDF_PATH = os.environ["PDF_PATH"]
INDEX_NAME = os.environ["INDEX_NAME"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]


# Load PDF
print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()


# Chunking
print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,   
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]


# Embeddings with EmbeddingGemma-300M
print("Encoding chunks with Ollama EmbeddingGemma...")
embeddings = []

for text in texts:
    response = ollama.embeddings(model="embeddinggemma", prompt=text)
    embeddings.append(response["embedding"])

embeddings = np.array(embeddings)



# Pinecone Setup
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=embeddings.shape[1],  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


# Upsert Embeddings into Pinecone
print("Upserting embeddings into Pinecone...")
to_upsert = [
    (str(i), emb.tolist(), {"text": texts[i]})
    for i, emb in enumerate(embeddings)
]
index.upsert(to_upsert)
print("Finished indexing!")


# Query Pipeline
def query_rag(query: str, top_k=5):

    response = ollama.embeddings(model="embeddinggemma", prompt=query)
    query_emb = response["embedding"]

    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)

    context = " ".join([match["metadata"]["text"] for match in results["matches"]])

    prompt = f"Answer the following question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = ollama.chat(model="gemma3:4b", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

# Test a Query
if __name__ == "__main__":
    user_query = "Summarize the main findings of the PDF."
    answer = query_rag(user_query, top_k=5)
    print("\n=== Final Answer ===\n")
    print(answer)
