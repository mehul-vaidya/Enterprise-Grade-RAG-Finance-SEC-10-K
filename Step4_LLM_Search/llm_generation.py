#step 3 gives 3-4 best matching result
#this step selects bests one

import os
from openai import OpenAI

# =========================
# CONFIG
# =========================
MODEL_NAME = "gpt-4o-mini"   # or any available chat model
MAX_TOKENS = 500
TEMPERATURE = 0.2

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# PROMPT BUILDER
# =========================
def build_prompt(query, chunks):
    context_blocks = []
    citations = []

    for i, c in enumerate(chunks, 1):
        context_blocks.append(
            f"[{i}] Source: {c['source']} | Chunk: {c['chunk_id']}\n{c['text']}"
        )
        citations.append(f"[{i}] {c['source']} (chunk {c['chunk_id']})")

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a financial analyst assistant.\n"
        "Answer the question strictly using the provided context.\n"
        "Do not add external knowledge.\n"
        "Cite sources using [number] notation."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer with clear explanations and citations."
    )

    return system_prompt, user_prompt, citations

# =========================
# GENERATE ANSWER
# =========================
def generate_answer(query, reranked_chunks):
    system_prompt, user_prompt, citations = build_prompt(query, reranked_chunks)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "citations": citations
    }

# =========================
# TEST
# =========================
if __name__ == "__main__":
    dummy_chunks = [
        {
            "source": "AAPL_10K.html",
            "chunk_id": 12,
            "text": "Apple faces risks related to supply chain disruptions...",
        },
        {
            "source": "AAPL_10K.html",
            "chunk_id": 15,
            "text": "Market competition and pricing pressure may affect margins...",
        }
    ]

    q = "What are the main risk factors mentioned by Apple?"
    result = generate_answer(q, dummy_chunks)

    print("\nANSWER:\n")
    print(result["answer"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(c)