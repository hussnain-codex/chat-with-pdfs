"""Turns text into embeddings (lists of numbers that capture meaning).

Texts with similar meaning get vectors that point in a similar direction,
which is what lets us find the chunks that match a question.
"""

from sentence_transformers import SentenceTransformer

# Load a small embedding model that runs locally (no API key needed).
# all-MiniLM-L6-v2 turns any text into a vector of 384 numbers.
# It is downloaded once and loaded when this file is imported.
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    """Convert a list of texts into a NumPy array of vectors, one row per text.

    The vectors are normalized to length 1. That makes the dot product of two
    vectors equal to their cosine similarity (1 = same meaning, 0 = unrelated),
    so the retriever can compare them with a single np.dot.
    """
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,       # return a NumPy array instead of a list
        normalize_embeddings=True,   # scale every vector to length 1
        show_progress_bar=True,      # show progress while embedding many chunks
    )

    return embeddings


# Quick self-test: `python src/embeddings.py` shows what an embedding looks like.
if __name__ == "__main__":
    sample_chunks = [
        "Python is a programming language.",
        "Machine learning helps computers learn from data.",
        "Python is commonly used in artificial intelligence.",
    ]

    vectors = generate_embeddings(sample_chunks)

    print("Number of chunks:", len(sample_chunks))
    print("Embedding shape:", vectors.shape)          # (3 texts, 384 numbers each)
    print("First vector dimensions:", len(vectors[0]))
    print("First 5 values:", vectors[0][:5])
