from llm.ollama_client import get_embedding
import numpy as np

from config import TOP_K

def retrieve(query: str, index: list, k: int = TOP_K) -> str:
    query_embedding = get_embedding(query)
    results = _search(query_embedding, index, k)

    return "\n\n".join(
        f"Fuente: {item['source']} (chunk {item['chunk_id']})\n{item['text']}"
        for item in results
    )

def _search(
    query_embedding: list[float],
    index: list[dict],
    k: int,
) -> list[dict]:
    
    query = np.asarray(query_embedding, dtype=float)
    query /= np.linalg.norm(query)

    embeddings = np.asarray([item["embedding"] for item in index], dtype=float)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    scores = embeddings @ query

    idx = np.argsort(scores)[::-1][:k]

    return [index[i] for i in idx]