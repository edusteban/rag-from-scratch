import requests
from config import GEN_MODEL, EMBED_MODEL, GEN_URL, EMBED_URL

def ask_ollama_with_context(
    question: str,
    context: str | None = None,
    model: str = GEN_MODEL,
) -> str:

    instructions = """
    Eres un asistente útil.
    Responde priorizando la información del contexto.
    Si el contexto no contiene la respuesta, no la saques de tu conocimiento general.
    Si no tienes la respuesta. no te inventes información, simplemente contesta diciendo que no lo sabes.
    """
    
    prompt = f"Instrucciones: {instructions}\nContexto: {context if context else 'Ninguno'}\nPregunta: {question}"
    return _ask_ollama(prompt, model)

def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]

def get_embeddings(
    texts: list[str],
    model: str = EMBED_MODEL,
) -> list[list[float]]:
    
    payload = {
        "model": model,
        "input": texts
    }
    response = requests.post(EMBED_URL, json=payload)
    response.raise_for_status()

    return response.json()["embeddings"]


def _ask_ollama(
    prompt: str,
    model: str = GEN_MODEL,
    stream: bool = False,
) -> str:
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

    response = requests.post(GEN_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"]