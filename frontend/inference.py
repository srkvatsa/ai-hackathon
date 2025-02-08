def add_human_prompt(content):
    pass

def get_response(prompt):
    pass
    # augment prompt
    # augmented_prompt = custom_prompt(qdrant_tao, prompt)
    
    # # process prompt
    # add_human_prompt(augmented_prompt)
    
    # # send to TinyLlama
    # res = chat.invoke(messages)

    # # return response
    # return res.content

def augment_prompt(client, query: str):
    pass
    # kws = keyword_search.search(query);
    # results = vec_db.similarity_search(kws, k=3)
    # source_knowledge = "\n".join([x.page_content for x in results])
    # augment_prompt = f"""Using the contexts below, answer the query:

    # Contexts:
    # {source_knowledge}

    # Query: {query}"""
    # return augment_prompt

import requests

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