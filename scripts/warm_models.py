import os
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

# Point HF cache into the project before importing transformers
MODELS_DIR = Path("./data/models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODELS_DIR)


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Downloading embedding model: BAAI/bge-large-en-v1.5 ...")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
v = embedder.encode("hello world")
print(f"  embedding dim: {len(v)}")

print("Downloading reranker: BAAI/bge-reranker-v2-m3 ...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
score = reranker.predict([("query", "passage")])
print(f"  reranker score: {score}")

print("Models cached under", MODELS_DIR)
