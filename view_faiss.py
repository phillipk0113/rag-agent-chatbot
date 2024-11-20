import faiss
import numpy as np

# Load the FAISS index and document content
index = faiss.read_index("faiss_index.index")
documents = np.load("documents.npy", allow_pickle=True)

# Display some information about the FAISS index and sample document content
index_info = {
    "Number of Vectors": index.ntotal,
    "Vector Dimension": index.d
}

print("FAISS Index Info:", index_info)

# Print a few sample documents
print("\nSample Documents:")
for doc in documents[:5]:  # Adjust the number as needed
    print("-", doc)
