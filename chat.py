from llm.ollama_client import ask_ollama_with_context
from retrieval.indexing import read_index
from retrieval.retriever import retrieve

DEBUG = True
index = read_index("index.json")
question = ""

while True:
    question = input("> ")

    context = retrieve(question, index)
    if DEBUG:
        print(context)
        print("--" * 10)

    print(ask_ollama_with_context(question, context))