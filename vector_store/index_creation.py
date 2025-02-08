from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np

# Directory containing the text chunks
CHUNKS_DIR = "../database_population/roman_chunks/"
NUM_FILES = 1000  # chunk_0.txt to chunk_999.txt

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read text files and generate embeddings
texts = []
filenames = []

for i in range(NUM_FILES):
    file_path = os.path.join(CHUNKS_DIR, f"chunk_{i}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            texts.append(text)
            filenames.append(file_path)
    else:
        print(f"Warning: {file_path} not found.")

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# Create a FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 (Euclidean distance)

# Add embeddings to the index
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, "faiss_roman_chunks.index")

# Save metadata (optional)
with open("metadata.txt", "w", encoding="utf-8") as meta_file:
    for i, filename in enumerate(filenames):
        meta_file.write(f"{i}\t{filename}\n")

print("FAISS index successfully created and saved.")