
from llm.ollama_client import ask_ollama, ask_ollama_with_context

#print(ask_ollama("Que es un RAG?"))
print(ask_ollama_with_context("Que es un RAG?", "Un rag es un tipo de comida china"))