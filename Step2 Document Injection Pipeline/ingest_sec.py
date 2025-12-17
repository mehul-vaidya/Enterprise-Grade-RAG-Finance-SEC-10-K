# I built a lightweight ingestion pipeline that extracts and cleans SEC HTML filings, deduplicates content,
# chunks it using token-aware segmentation, embeds with a sentence transformer, and stores vectors in FAISS
# for fast similarity search.

# import os
# import re
# from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import tiktoken
# import hashlib
#
# # =====================
# # CONFIG
# # =====================
# DATA_DIR = "sec_10k_fast"
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# EMBED_MODEL = "all-MiniLM-L6-v2"
#
# # =====================
# # LOAD MODEL
# # =====================
# model = SentenceTransformer(EMBED_MODEL)
# tokenizer = tiktoken.get_encoding("cl100k_base")
#
# # =====================
# # CLEAN HTML → TEXT
# # =====================
# def extract_text(html):
#     soup = BeautifulSoup(html, "html.parser")
#
#     # remove scripts/styles
#     for tag in soup(["script", "style", "table"]):
#         tag.decompose()
#
#     text = soup.get_text(separator=" ")
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()
#
# # =====================
# # DEDUPE USING HASH
# # =====================
# def is_duplicate(text, seen_hashes):
#     h = hashlib.md5(text.encode()).hexdigest()
#     if h in seen_hashes:
#         return True
#     seen_hashes.add(h)
#     return False
#
# # =====================
# # CHUNKING
# # =====================
# def chunk_text(text):
#     tokens = tokenizer.encode(text)
#     chunks = []
#
#     for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
#         chunk = tokens[i:i + CHUNK_SIZE]
#         chunks.append(tokenizer.decode(chunk))
#
#     return chunks
#
# # =====================
# # INGEST
# # =====================
# texts = []
# metadata = []
# seen_hashes = set()
#
# for fname in os.listdir(DATA_DIR):
#     if not fname.endswith(".htm") and not fname.endswith(".html"):
#         continue
#
#     path = os.path.join(DATA_DIR, fname)
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         html = f.read()
#
#     text = extract_text(html)
#     if len(text) < 1000:
#         continue
#
#     if is_duplicate(text, seen_hashes):
#         continue
#
#     chunks = chunk_text(text)
#
#     for i, chunk in enumerate(chunks):
#         texts.append(chunk)
#         metadata.append({
#             "source": fname,
#             "chunk_id": i
#         })
#
# print(f"Total chunks: {len(texts)}")
#
# # =====================
# # EMBEDDINGS
# # =====================
# embeddings = model.encode(texts, show_progress_bar=True)
# embeddings = np.array(embeddings).astype("float32")
#
# # =====================
# # FAISS VECTOR STORE
# # =====================
# dim = embeddings.shape[1]
# index = faiss.IndexFlatL2(dim)
# index.add(embeddings)
#
# faiss.write_index(index, "sec_index.faiss")
#
# print("✅ Ingestion complete. Vector index saved.")

import os
import re
import hashlib
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import tiktoken
import pickle
from tqdm import tqdm

# =====================
# CONFIG
# =====================
DATA_DIR = "sec_10k_fast"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "sec_index.faiss"
METADATA_PATH = "sec_metadata.pkl"
BATCH_SIZE = 64  # batch for embedding generation

# =====================
# LOAD MODEL & TOKENIZER
# =====================
model = SentenceTransformer(EMBED_MODEL)
tokenizer = tiktoken.get_encoding("cl100k_base")

# =====================
# CLEAN HTML → TEXT
# =====================
def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================
# DEDUPE USING HASH
# =====================
def is_duplicate(text, seen_hashes):
    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False

# =====================
# CHUNKING
# =====================
def chunk_text(text):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = tokens[i:i + CHUNK_SIZE]
        chunks.append(tokenizer.decode(chunk))
    return chunks

# =====================
# INGEST
# =====================
texts = []
metadata = []
seen_hashes = set()

print("📝 Processing documents...")
for fname in tqdm(os.listdir(DATA_DIR)):
    if not fname.endswith(".htm") and not fname.endswith(".html"):
        continue

    path = os.path.join(DATA_DIR, fname)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    text = extract_text(html)
    if len(text) < 1000:
        continue

    if is_duplicate(text, seen_hashes):
        continue

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadata.append({
            "source": fname,
            "chunk_id": i,
            "text": chunk
        })

print(f"Total chunks to embed: {len(texts)}")

# =====================
# EMBEDDINGS (BATCH)
# =====================
embeddings = []
print("🔹 Generating embeddings in batches...")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings).astype("float32")

# =====================
# FAISS VECTOR STORE
# =====================
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)

# =====================
# SAVE METADATA
# =====================
with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("✅ Ingestion complete. FAISS index and metadata saved.")
