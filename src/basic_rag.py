"""From-scratch RAG: pypdf -> hand-written splitter -> MiniLM embeddings in NumPy -> cosine top-k -> Groq."""

import argparse
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from answer_generator import generate_answer
from embeddings import generate_embeddings


PDF_DIR = Path(__file__).resolve().parent.parent / "pdfs"


def load_pdf_words(pdf_dir=PDF_DIR):
    """Return {file name: [(word, page_number), ...]} for every readable PDF."""
    documents = {}

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
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

        words = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            words.extend((word, page_number) for word in text.split())

        documents[pdf_path.name] = words

    return documents


def split_words(words, chunk_size=500, overlap=50):
    """Sliding window over (word, page) pairs. Chunks may cross page boundaries."""
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        first_page, last_page = window[0][1], window[-1][1]
        chunks.append({
            "text": " ".join(word for word, _ in window),
            "page": str(first_page) if first_page == last_page else f"{first_page}-{last_page}",
        })
        if start + chunk_size >= len(words):
            break

    return chunks


def load_and_chunk_pdfs(chunk_size=500, overlap=50):
    chunks = []

    for file_name, words in load_pdf_words().items():
        for chunk in split_words(words, chunk_size, overlap):
            chunks.append({**chunk, "source": file_name})

    return chunks


def retrieve_chunks(question, chunks, embeddings, top_k=4):
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


def print_sources(results):
    print("\nSources:")
    for result in results:
        print(f"  - {result['source']}, page {result['page']} (similarity {result['similarity']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="From-scratch PDF question answering")
    parser.add_argument("--chunk-size", type=int, default=500, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="words of overlap")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    chunks = load_and_chunk_pdfs(args.chunk_size, args.overlap)

    if not chunks:
        print("No PDF text found. Check the files inside the pdfs folder.")
        raise SystemExit(1)

    print(f"Total chunks created: {len(chunks)}")

    embeddings = generate_embeddings([chunk["text"] for chunk in chunks])

    print("\nAsk questions about your PDFs (empty line to quit).")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            break

        results = retrieve_chunks(question, chunks, embeddings, args.top_k)
        print("\nAnswer:\n" + generate_answer(question, results))
        print_sources(results)
