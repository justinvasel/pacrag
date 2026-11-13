# scripts/smoke_test.py
"""One-document end-to-end check: parse -> chunk -> embed -> index -> query -> generate."""

import json
import os
import uuid
from pathlib import Path

import torch
from anthropic import Anthropic
from pac_pipeline.config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

os.environ["HF_HOME"] = str(settings.models_dir.resolve())

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# 1. Load LiteParse output and chunk crudely (we'll do this properly later)
parsed = json.loads(Path("data/parsed/2025/25-FC-118.json").read_text())
# LiteParse JSON shape: pages with text. Handle both common shapes.
pages = parsed.get("pages") or parsed.get("data", {}).get("pages") or []
text = "\n\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in pages)
chunks = [text[i : i + 1500] for i in range(0, len(text), 1200)]  # ~1500 char chunks, 300 overlap
print(f"Chunks: {len(chunks)}")

# 2. Embed
embedder = SentenceTransformer(settings.embedding_model, device=device)
vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()
dim = len(vectors[0])
print(f"Embedding dim: {dim}")

# 3. Index in Qdrant (embedded mode)
settings.qdrant_path.mkdir(parents=True, exist_ok=True)
qdrant = QdrantClient(path=str(settings.qdrant_path))

if not qdrant.collection_exists("pac_smoke"):
    qdrant.create_collection(
        collection_name="pac_smoke",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

qdrant.upsert(
    collection_name="pac_smoke",
    points=[
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"opinion_id": "25-FC-118", "chunk_index": i, "text": chunks[i]},
        )
        for i, vec in enumerate(vectors)
    ],
)
print("Indexed.")

# 4. Query
question = "What did the City of Bloomington do that was alleged to violate APRA?"
qvec = embedder.encode(question, normalize_embeddings=True).tolist()
hits = qdrant.query_points(
    collection_name="pac_smoke",
    query=qvec,
    limit=4,
).points
print(f"Top {len(hits)} chunks retrieved.")

# 5. Generate
context = "\n\n---\n\n".join(
    f"[Opinion {h.payload['opinion_id']}, chunk {h.payload['chunk_index']}]\n{h.payload['text']}"
    for h in hits
)
client = Anthropic(api_key=settings.anthropic_api_key)
prompt = f"""You are answering a question about Indiana PAC advisory opinions.
Use ONLY the excerpts below. Cite each claim as (opinion_id).

Excerpts:
{context}

Question: {question}

Answer:"""
msg = client.messages.create(
    model=settings.generation_model,
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}],
)
print("\n=== Answer ===")
print(msg.content[0].text)
