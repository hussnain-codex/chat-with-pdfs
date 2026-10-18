
from sentence_transformers import SentenceTransformer

# Load a small, locally running embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    """
    Convert a list of text chunks into numerical vectors.
    """
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


if __name__ == "__main__":
    sample_chunks = [
        "Python is a programming language.",
        "Machine learning helps computers learn from data.",
        "Python is commonly used in artificial intelligence.",
    ]

    vectors = generate_embeddings(sample_chunks)

    print("Number of chunks:", len(sample_chunks))
    print("Embedding shape:", vectors.shape)
    print("First vector dimensions:", len(vectors[0]))
    print("First 5 values:", vectors[0][:5])