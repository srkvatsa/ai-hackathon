import index_search

def augment_prompt(query: str, source_knowledge):
    #augment_prompt = f'''[INST] You are an expert assistant providing detailed and accurate responses. Use the provided context to answer the user's question as accurately as possible. If the context is insufficient, indicate that you do not have enough information rather than making up an answer. 
    augment_prompt = f'''
    [INST] 
Using only the following historal context, answer the query from the User
<Historical Textbook Background> 
{source_knowledge}
<Historical Textbook Background> 

## User Question: {query}

## Response:
[/INST]
    '''

    return augment_prompt

import requests

def query(prompt):
    # user_query = input("Enter your query: ")
    kws = index_search.gen_keywords_from_prompt(prompt)
    print(kws)
    results = index_search.search_faiss_index(index, kws, top_k=5)
    print(results)
    context = ""
    for result in results:
        doc = index_search.get_chunk_text(result[0])
        context += '\n\n' + doc
    
    final_prompt = augment_prompt(prompt, context)
    print(f"{final_prompt}\n\n")
    return query_llama(final_prompt)

    # print("Top relevant results:")
    # for idx, score in results:
    #     print(f"Chunk ID: {idx}, Distance: {score}")
    #     print(get_chunk_text(idx))


def query_llama(query: str, host: str = "http://localhost:8080") -> str:
    """
    Sends a user query to a llama.cpp server running on localhost:8080
    and returns the response.

    Args:
        query (str): The user query.
        host (str): The base URL of the llama.cpp server.

    Returns:
        str: The response from llama.cpp.
    """
    url = f"{host}/v1/chat/completions"
    payload = {
        "model": "llama",  # Adjust based on the model name running in llama.cpp
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with llama.cpp: {e}"


messages = []
index = index_search.load_faiss_index()

if __name__ == "__main__":
    prompt = input("Enter your query: ")
    print(query(prompt))
