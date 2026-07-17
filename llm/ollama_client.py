import requests

URL = "http://localhost:11434/api/generate"


def ask_ollama(question, model = "qwen3:8b", stream = False):
    payload = {
        "model": model,
        "prompt": question,
        "stream": stream
    }

    response = requests.post(URL, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}") 
    return response.json()["response"]

def ask_ollama_with_context(question, context = None, model = "qwen3:8b"):
    instructions = "Responde utilizando únicamente la información del contexto. Si el contexto no contiene la respuesta, di 'No lo sé'."
    prompt = f"Instrucciones: {instructions}\nContexto: {context if context else 'Ninguno'}\nPregunta: {question}"
    return ask_ollama(prompt, model)