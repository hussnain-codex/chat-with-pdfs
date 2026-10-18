
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from embeddings import generate_embeddings


PDF_DIR = Path(__file__).resolve().parent.parent / "pdfs"


def load_and_chunk_pdfs(chunk_size=500, overlap=50):
    chunks = []

    for pdf_path in PDF_DIR.glob("*.pdf"):
        # Verify that the file is actually a PDF
        with open(pdf_path, "rb") as file:
            if file.read(5) != b"%PDF-":
                print(f"Skipping invalid PDF: {pdf_path.name}")
                continue

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as error:
            print(f"Could not read {pdf_path.name}: {error}")
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            words = text.split()

            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_text = " ".join(words[start:end])

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "source": pdf_path.name,
                        "page": page_number,
                    })

                if end >= len(words):
                    break

                start += chunk_size - overlap

    return chunks


def retrieve_chunks(question, chunks, embeddings, top_k=3):
    question_embedding = generate_embeddings([question])[0]

    # Embeddings are normalized, so dot product gives cosine similarity
    similarities = np.dot(embeddings, question_embedding)
    top_indexes = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            **chunks[int(index)],
            "similarity": float(similarities[index]),
        }
        for index in top_indexes
    ]


if __name__ == "__main__":
    chunks = load_and_chunk_pdfs()

    if not chunks:
        print("No PDF text found. Check the files inside the pdfs folder.")
        raise SystemExit(1)

    print(f"Total chunks created: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    question = input("\nAsk a question about your PDFs: ").strip()

    if not question:
        print("Please enter a question.")
        raise SystemExit(1)

    results = retrieve_chunks(question, chunks, embeddings)

    print("\nMost relevant passages:\n")

    for result in results:
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(result["text"])
        print("-" * 60)