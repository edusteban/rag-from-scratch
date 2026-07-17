# RAG from Scratch

A Retrieval-Augmented Generation (RAG) pipeline implemented from scratch using Python and Ollama.

---

## Overview

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline without relying on frameworks such as LangChain or ChromaDB.

The goal is to understand how a RAG system works internally by implementing every component manually.

---

## Features

- Local LLM inference with Ollama
- Local embeddings using `nomic-embed-text`
- Character-based document chunking
- Source-aware retrieval (document name and chunk ID included in the retrieved context)
- Cosine similarity retrieval
- Top-k context selection
- Manual vector index construction
- Modular Python project structure

---

## Architecture

```text
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Cosine Similarity Search
      │
      ▼
Retrieve Top-k Chunks
      │
      ▼
Prompt Construction
      │
      ▼
Ollama (Qwen3)
      │
      ▼
Answer
```

---

## Project Structure

```text
rag-from-scratch/
│
├── documents/
├── index/
├── llm/
├── retrieval/
│
├── build_index.py
├── chat.py
├── config.py
├── requirements.txt
└── README.md
```

---

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Download the required Ollama models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Build the embedding index:

```bash
python build_index.py
```

Start the chat application:

```bash
python chat.py
```

---

## Configuration

The project is designed to be easily customizable.

- Place your own `.txt` documents inside the `documents/` directory.
- Rebuild the vector index after modifying the documents:

```bash
python build_index.py
```

- Models, chunk size, chunk overlap, and retrieval parameters can be configured in `config.py`.

---

## How It Works

1. Documents are split into overlapping chunks.
2. Each chunk is converted into an embedding.
3. Chunk embeddings are stored in a local JSON vector index.
4. The user's query is embedded.
5. Cosine similarity retrieves the most relevant chunks.
6. Retrieved chunks are injected into the LLM prompt.
7. The LLM generates an answer using only the retrieved context.

---

## Design Decisions

This implementation uses **character-based chunking** instead of token-based chunking.

While token-based chunking is generally preferred because it aligns with the model's context window, Ollama does not currently expose a public tokenization API. Using an external tokenizer would introduce a dependency tied to a specific model, so character-based chunking was chosen to keep the project lightweight and focused on the core retrieval pipeline.

---

## Technologies

- Python
- NumPy
- Requests
- Ollama
- Qwen3
- Nomic Embed

---

## Future Improvements

- ChromaDB integration
- FAISS vector search
- Token-based chunking
- Metadata filtering
- Hybrid search
- Query reranking