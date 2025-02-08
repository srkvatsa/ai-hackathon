from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np
from keybert import KeyBERT
import json

# Load the MiniLM model
model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT()

def load_faiss_index(index_path="../vector_store/faiss_roman_concats.index"):
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

def get_chunk_text(chunk_id, metadata_path="../vector_store/metadata.json"):
    """
    Retrieves the chunk text corresponding to the given chunk_id from the metadata JSON file.

    Args:
        chunk_id (int or str): The chunk ID to look up.
        metadata_path (str): Path to the JSON metadata file.

    Returns:
        str: The chunk text if found, otherwise an error message.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    chunk_data = metadata.get(str(chunk_id))  # Ensure we look for chunk_id as a string
    if chunk_data and "chunk" in chunk_data:
        return chunk_data["chunk"]
    else:
        return f"Chunk {chunk_id} not found."
    

def gen_keywords_from_prompt(prompt):
    res = kw_model.extract_keywords(prompt, top_n=10)
    print(f"res: {res}")
    kws = ", ".join([keyword[0] for keyword in res])
    return kws

# Example Usage
if __name__ == "__main__":
    index = load_faiss_index()
    user_query = input("Enter your query: ")
    kws = gen_keywords_from_prompt(user_query)
    print(kws)
    results = search_faiss_index(index, kws, top_k=5)

    print("Top relevant results:")
    for idx, score in results:
        print(f"Chunk ID: {idx}, Distance: {score}")
        print(get_chunk_text(idx))

