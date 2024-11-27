import faiss
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

#Load .env file
load_dotenv()

perma_context = os.getenv("FIXED_CONTEXT")
print(perma_context)
# Load FAISS index and documents
def load_faiss_index():
    index = faiss.read_index("faiss_index.index")
    documents = np.load("documents.npy", allow_pickle=True)
    return index, documents

# Search FAISS for top-k results
def search_faiss(index, documents, query, top_k=3):
    # Embed the query
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode([query])

    # Perform FAISS search
    distances, indices = index.search(query_embedding, k=top_k)

    # Retrieve the top documents
    top_documents = [documents[idx] for idx in indices[0]]
    return top_documents, distances[0]

# Ollama integration for LLM
def query_ollama(prompt, model_name="openhermes"):
    output = ollama.generate(
        model=model_name,
        prompt = prompt
    )
    return output["response"]

# Main QA Workflow
def run_query():
    # Load FAISS index and documents
    index, documents = load_faiss_index()

    # Get user query
    user_query = input("Enter your query: ")

    # Search FAISS for relevant documents
    top_docs, distances = search_faiss(index, documents, user_query, top_k=3)
    top_docs.append(perma_context)

    """
    # Print retrieved documents
    print("\nTop Retrieved Documents:")
    for i, doc in enumerate(top_docs):
        print(f"Document {i + 1} (Distance: {distances[i]:.2f}): {doc}")
    """
    # Create the context for the LLM
    context = " ".join(top_docs)
    prompt = f"Context: {context}\nUser Query: {user_query}\nResponse:"

    # Query the LLM via Ollama
    response = query_ollama(prompt)
    print("\nGenerated Response:")
    print(response)

# Run the query
if __name__ == "__main__":
    while True:
        run_query()
