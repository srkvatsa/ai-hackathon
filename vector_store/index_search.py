from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np

# Load the MiniLM model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_index(index_path="faiss_roman_chunks.index"):
    """Loads the FAISS index from the given file."""
    return faiss.read_index(index_path)

def search_faiss_index(index, query, top_k=5):
    """
    Searches the FAISS index for the top_k most relevant vectors for a given query.

    Args:
        index (faiss.IndexFlatL2): The FAISS index.
        query (str): The input query.
        top_k (int): Number of most similar vectors to return.

    Returns:
        List of tuples (index_id, similarity_score).
    """
    # Generate the query embedding
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Perform the search
    distances, indices = index.search(query_embedding, top_k)

    # Return results as a list of (index_id, similarity_score)
    return list(zip(indices[0], distances[0]))

def get_chunk_text(chunk_id, metadata_path="metadata.txt"):
    with open(metadata_path, "r", encoding="utf-8") as meta_file:
        lines = meta_file.readlines()
        chunk_file = dict(line.strip().split("\t") for line in lines).get(str(chunk_id), None)

    if chunk_file and os.path.exists(chunk_file):
        with open(chunk_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return f"Chunk {chunk_id} not found."

# Example Usage
if __name__ == "__main__":
    index = load_faiss_index()
    user_query = input("Enter your query: ")
    results = search_faiss_index(index, user_query, top_k=5)

    print("Top relevant results:")
    for idx, score in results:
        print(f"Chunk ID: {idx}, Distance: {score}")
        print(get_chunk_text(idx))

