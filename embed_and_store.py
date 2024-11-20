from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from extract_website_content_playwright import extract_dynamic_content


url = "https://www.meridianexecutivelimo.com/"
extracted_texts = extract_dynamic_content(url)

#initializes the embedding model for vector database
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

#Creating vector embeddings
embeddings = embed_model.encode(list(extracted_texts))

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.index")
np.save("documents.npy", np.array(list(extracted_texts)))


print("FAISS index created and saved with website content.")