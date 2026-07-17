import os
import json

from config import DOCS_PATH, INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from llm.ollama_client import get_embeddings

def build_index() -> list[dict]:
    chunk_list = []
    sources = []

    for file in os.listdir(DOCS_PATH):
        text = _read_document(file)

        for i, chunk_text in enumerate(_chunk(text)):
            chunk_list.append(chunk_text)
            sources.append({
                "source": file,
                "chunk_id": i
            })

    embeddings = get_embeddings(chunk_list)

    return [
        {
            "source": source["source"],
            "chunk_id": source["chunk_id"],
            "text": chunk_text,
            "embedding": embedding
        }
        for source, chunk_text, embedding in zip(sources, chunk_list, embeddings)
    ]
    
def write_index(file: str, index: list[dict]) -> None:
    path = os.path.join(INDEX_PATH, file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=4)

def read_index(file: str) -> list[dict]:
    path = os.path.join(INDEX_PATH, file)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def _chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    return chunks

def _read_document(file: str) -> str:
    path = os.path.join(DOCS_PATH, file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}")
    with open (path, "r", encoding="utf-8") as f:
        return f.read() 