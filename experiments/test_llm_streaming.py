import requests
import json

URL = "http://localhost:11434/api/generate"

data = {
    "model": "qwen3:8b",
    "prompt": "¿Qué es un RAG?",
    "stream": True
}

response = requests.post(URL, json=data, stream=True)
if response.status_code == 200:
    for line in response.iter_lines():
        if not line:
            continue
        print(json.loads(line)["response"], end="", flush=True)

