
import numpy as np
from embeddings import generate_embeddings


def retrieve_chunks(question, chunks, embeddings, top_k=2):
    # Convert the user's question into an embedding
    question_embedding = generate_embeddings([question])[0]

    # Calculate cosine similarity (embeddings are normalized)
    similarities = np.dot(embeddings, question_embedding)

    # Get the indexes of the most relevant chunks
    top_indexes = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indexes:
        results.append({
            "text": chunks[index],
            "similarity": float(similarities[index])
        })

    return results


if __name__ == "__main__":
    chunks = [
        "Python is a popular programming language.",
        "Machine learning allows computers to learn from data.",
        "Python is widely used for artificial intelligence."
    ]

    embeddings = generate_embeddings(chunks)

    question = "How is Python used in AI?"
    results = retrieve_chunks(question, chunks, embeddings)

    print("\nQuestion:", question)
    print("\nRelevant chunks:")

    for result in results:
        print(f"\nSimilarity: {result['similarity']:.4f}")
        print(result["text"])