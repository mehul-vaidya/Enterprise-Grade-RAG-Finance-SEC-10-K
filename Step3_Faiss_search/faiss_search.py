# faiss_search_with_reranker.py
# FAISS retrieval (Top-K) + Cross-Encoder reranking (Top-N)

# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pickle
#
# # =========================
# # CONFIG
# # =========================
# FAISS_INDEX_PATH = "sec_index.faiss"
# METADATA_PATH = "sec_metadata.pkl"   # must match ingestion step
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# TOP_K = 5
#
# # =========================
# # LOAD MODEL
# # =========================
# model = SentenceTransformer(MODEL_NAME)
#
# # =========================
# # LOAD FAISS INDEX
# # =========================
# index = faiss.read_index(FAISS_INDEX_PATH)
#
# # =========================
# # LOAD METADATA
# # =========================
# with open(METADATA_PATH, "rb") as f:
#     metadata = pickle.load(f)
#
# # =========================
# # SEARCH FUNCTION
# # =========================
# def search(query):
#     query_embedding = model.encode([query])
#     query_embedding = np.array(query_embedding).astype("float32")
#
#     distances, indices = index.search(query_embedding, TOP_K)
#
#     results = []
#     for idx in indices[0]:
#         results.append(metadata[idx])
#
#     return results
#
#
# # =========================
# # TEST
# # =========================
# if __name__ == "__main__":
#     query = "What are the main risk factors mentioned by Apple?"
#     results = search(query)
#
#     for i, r in enumerate(results, 1):
#         print(f"\nResult {i}")
#         print("-" * 40)
#         print(r["text"][:500])

#
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pickle
# from tqdm import tqdm
#
# # =========================
# # CONFIG
# # =========================
# FAISS_INDEX_PATH = "sec_index.faiss"
# METADATA_PATH = "sec_metadata.pkl"
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# TOP_K = 5
#
# # =========================
# # LOAD MODEL
# # =========================
# model = SentenceTransformer(MODEL_NAME)
#
# # =========================
# # LOAD FAISS INDEX
# # =========================
# index = faiss.read_index(FAISS_INDEX_PATH)
#
# # =========================
# # LOAD METADATA
# # =========================
# with open(METADATA_PATH, "rb") as f:
#     metadata = pickle.load(f)
#
# assert len(metadata) == index.ntotal, "Metadata length and FAISS index size must match!"
#
# # =========================
# # SEARCH FUNCTION
# # =========================
# def search(query, top_k=TOP_K):
#     query_embedding = model.encode([query], show_progress_bar=False)
#     query_embedding = np.array(query_embedding).astype("float32")
#
#     distances, indices = index.search(query_embedding, top_k)
#
#     results = []
#     for idx, dist in zip(indices[0], distances[0]):
#         chunk_meta = metadata[idx]
#         results.append({
#             "source": chunk_meta["source"],
#             "chunk_id": chunk_meta["chunk_id"],
#             "text": chunk_meta["text"],
#             "distance": float(dist)
#         })
#     return results
#
# # =========================
# # TEST
# # =========================
# if __name__ == "__main__":
#     query = "What are the main risk factors mentioned by Apple?"
#     results = search(query)
#
#     print(f"\nTop {TOP_K} results for query:\n'{query}'")
#     for i, r in enumerate(results, 1):
#         print("\n" + "="*50)
#         print(f"Result {i}")
#         print(f"Source File : {r['source']}")
#         print(f"Chunk ID    : {r['chunk_id']}")
#         print(f"Distance    : {r['distance']:.4f}")
#         print(f"Text Preview: {r['text'][:500]}...")  # first 500 chars


# FAISS retrieval - just return top N
# Cross-Encoder - actually finds meaningful matching
# LLM GENERATION (ESSENTIAL) - takes the reranked chunks and produces the final answer.
# FAISS retrieval (Top-K) + Cross-Encoder reranking (Top-N)


import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# CONFIG
# =========================
FAISS_INDEX_PATH = "sec_index.faiss"
METADATA_PATH = "sec_metadata.pkl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 10   # FAISS retrieval
TOP_N = 3    # After reranking

# =========================
# LOAD MODELS
# =========================
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

# =========================
# LOAD FAISS INDEX
# =========================
index = faiss.read_index(FAISS_INDEX_PATH)

# =========================
# LOAD METADATA
# =========================
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

assert index.ntotal == len(metadata), "FAISS index and metadata size mismatch"

# =========================
# SEARCH + RERANK
# =========================
def search(query, top_k=TOP_K, top_n=TOP_N):
    # ---- FAISS SEARCH ----
    query_emb = embedder.encode([query])
    query_emb = np.array(query_emb).astype("float32")

    distances, indices = index.search(query_emb, top_k)

    retrieved = []
    for idx, dist in zip(indices[0], distances[0]):
        m = metadata[idx]
        retrieved.append({
            "source": m["source"],
            "chunk_id": m["chunk_id"],
            "text": m["text"],
            "distance": float(dist)
        })

    # ---- RERANK ----
    pairs = [(query, r["text"]) for r in retrieved]
    scores = reranker.predict(pairs)

    for r, s in zip(retrieved, scores):
        r["rerank_score"] = float(s)

    reranked = sorted(
        retrieved,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_n]

# =========================
# TEST
# =========================
if __name__ == "__main__":
    query = "What are the main risk factors mentioned by Apple?"
    results = search(query)

    for i, r in enumerate(results, 1):
        print("\n" + "=" * 50)
        print(f"Result {i}")
        print(f"Source     : {r['source']}")
        print(f"Chunk ID   : {r['chunk_id']}")
        print(f"FAISS Dist : {r['distance']:.4f}")
        print(f"Rerank Scr : {r['rerank_score']:.4f}")
        print(f"Text       : {r['text'][:500]}...")

