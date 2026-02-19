import requests
from functools import lru_cache
from typing import List, Optional
from config import OLLAMA_BASE_URL, OLLAMA_INFERENCE_MODEL, OLLAMA_EMBEDDING_MODEL, OLLAMA_VECTOR_DIMENSION

# Кэшированный вызов чата
@lru_cache(maxsize=128)
def cached_ollama_call(prompt: str, model: Optional[str] = None) -> str:
    """Cached call to Ollama chat completion."""
    if model is None:
        model = OLLAMA_INFERENCE_MODEL

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise graph relationship extractor. Extract all relationships from the text "
                    "and format them as a JSON object with this exact structure:\n"
                    '{ "graph": [ {"node": "Person/Entity", "target_node": "Related Entity", "relationship": "Type of Relationship"}, ... ] }\n'
                    "Include ALL relationships mentioned in the text, including implicit ones. Be thorough and precise."
                )
            },
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        result = response.json()
        if 'message' in result and 'content' in result['message']:
            return result['message']['content']
        else:
            print(f"Warning: Unexpected API response format: {result}")
            return '{"graph": []}'
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama API at {OLLAMA_BASE_URL}")
        return '{"graph": []}'
    except Exception as e:
        print(f"Error in Ollama API call: {str(e)}")
        return '{"graph": []}'


# Получение эмбеддингов
def ollama_embeddings(text: str) -> List[float]:
    """Generate embeddings for a single text string using Ollama."""
    try:
        payload = {
            "model": OLLAMA_EMBEDDING_MODEL,
            "prompt": text
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("embedding", [0] * OLLAMA_VECTOR_DIMENSION)
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return [0] * OLLAMA_VECTOR_DIMENSION


if __name__ == "__main__":
    test_prompt = "Извлеки отношения: Алиса работает в ТехКорпе."
    print("Тестовый вызов cached_ollama_call:")
    response = cached_ollama_call(test_prompt)
    print(response)

    print("\nТест эмбеддингов:")
    emb = ollama_embeddings("тестовый текст")
    print(f"Размер эмбеддинга: {len(emb)}")