import faiss
import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain  # Updated import
import requests

# Directly load FAISS index
index = faiss.read_index("faiss_index.index")
documents = np.load("documents.npy", allow_pickle=True)

# Initialize LangChain FAISS wrapper with the loaded index
embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
vector_store = FAISS(embedding_model=embeddings, index=index, documents=documents)

# Define a custom LLM class to integrate with Ollama
class CustomLLM:
    def __init__(self, model_name="Llama-3.2", server_url="http://localhost:11434"):
        self.model_name = model_name
        self.server_url = server_url

    def generate_response(self, prompt):
        url = f"{self.server_url}/generate"
        payload = {"model": self.model_name, "prompt": prompt}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get('text', 'No response generated.')
        else:
            raise ValueError(f"Error: {response.status_code} - {response.text}")

    def __call__(self, prompt):
        return self.generate_response(prompt)

# Initialize the custom LLM
custom_llm = CustomLLM()

# Create a LangChain QA chain with FAISS as the retriever
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=custom_llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# Run a query
query = input("Enter your query: ")
response = qa_chain.run(query)
print("\nGenerated Response:")
print(response)
