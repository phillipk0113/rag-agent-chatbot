import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Load the FAISS index and documents
index = faiss.read_index("faiss_index.index")
documents = np.load("documents.npy", allow_pickle=True)

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to send a query to the local Ollama server
def generate_response_with_ollama(prompt):
    url = "http://localhost:11434/generate"  # The local endpoint for Ollama
    payload = {
        "model": "Llama-3.2",
        "prompt": prompt
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raises an error for HTTP status codes 4xx/5xx

        # Return the generated text if the response is successful
        return response.json().get('text', 'No response generated.')
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "Failed to generate a response."

def search_query(query, top_k=3):
    # Embed the user's query
    query_embedding = embedding_model.encode([query])

    # Search the FAISS index
    distances, indices = index.search(query_embedding, k=top_k)

    # Retrieve the top documents
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return retrieved_docs, distances[0]

def generate_response(context, query):
    # Concatenate the context for Llama 3.2
    context_text = " ".join(context)
    prompt = f"Context: {context_text}\nUser query: {query}\nResponse:"

    # Generate a response using Ollama's local service
    return generate_response_with_ollama(prompt)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    top_docs, distances = search_query(user_query)

    print("\nTop Retrieved Documents:")
    for i, doc in enumerate(top_docs):
        print(f"Document {i + 1} (Distance: {distances[i]:.2f}): {doc}")

    # Generate a response with the retrieved context
    response = generate_response(top_docs, user_query)
    print("\nGenerated Response:")
    print(response)
