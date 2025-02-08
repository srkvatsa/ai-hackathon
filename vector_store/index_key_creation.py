from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np
import json

# Load the MiniLM model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Path to JSON file containing key-value pairs (keys: comma-separated keywords, values: chunks)
JSON_FILE = "../database_population/doc_inclusion.json"

# Read JSON file and extract keys (keywords) and values (chunks)
keys = []
chunks = []
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load JSON as a dictionary

    for key, chunk in data.items():
        keys.append(key)  # Store the comma-separated keyword string
        chunks.append(chunk)  # Store the corresponding text chunk

# Generate embeddings for the keys
print("Generating embeddings for keywords...")
embeddings = model.encode(keys, convert_to_numpy=True, normalize_embeddings=True)

# Create a FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 (Euclidean distance)

# Add embeddings to the index
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, "faiss_roman_keywords.index")

# Save metadata mapping (Index -> Key -> Chunk)
metadata_file = "metadata.json"
metadata = {str(i): {"keywords": keys[i], "chunk": chunks[i]} for i in range(len(keys))}

with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print("FAISS index and metadata successfully created and saved.")
