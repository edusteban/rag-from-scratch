from retrieval.indexing import build_index, write_index

index = build_index()
write_index("index.json", index)