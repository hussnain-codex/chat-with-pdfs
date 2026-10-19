"""From-scratch RAG: pypdf -> hand-written splitter -> MiniLM embeddings in NumPy -> cosine top-k -> Groq.

Pipeline: load the PDFs -> cut them into chunks -> embed every chunk ->
for each question, find the most similar chunks -> ask the LLM to answer from them.
"""

import argparse
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from answer_generator import generate_answer
from embeddings import generate_embeddings


# The pdfs/ folder next to src/
PDF_DIR = Path(__file__).resolve().parent.parent / "pdfs"


def load_pdf_words(pdf_dir=PDF_DIR):
    """Step 1 (load): read every PDF and return its words with their page numbers.

    Returns {file name: [(word, page_number), ...]}. Keeping the page number
    for every word lets each chunk report which page(s) it came from.
    Files that are not real PDFs or can't be read are skipped.
    """
    documents = {}

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        # A real PDF file always starts with the bytes "%PDF-"
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
            # extract_text() returns None for image-only pages, so fall back to ""
            text = page.extract_text() or ""
            words.extend((word, page_number) for word in text.split())

        documents[pdf_path.name] = words

    return documents


def split_words(words, chunk_size=500, overlap=50):
    """Step 2 (chunk): our own splitter, a sliding window over the words.

    Each chunk is `chunk_size` words long. The window moves forward by
    (chunk_size - overlap) words, so neighbouring chunks share `overlap` words
    and a sentence cut at a boundary still appears whole in one of them.
    A chunk can cross a page break; its page is then shown as e.g. "7-8".
    """
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        first_page, last_page = window[0][1], window[-1][1]
        chunks.append({
            "text": " ".join(word for word, _ in window),
            "page": str(first_page) if first_page == last_page else f"{first_page}-{last_page}",
        })
        # Stop once the window has reached the last word of the document
        if start + chunk_size >= len(words):
            break

    return chunks


def load_and_chunk_pdfs(chunk_size=500, overlap=50):
    """Steps 1 + 2: load all PDFs and split them into chunks.

    Returns a flat list of dicts: {"text", "page", "source" (file name)}.
    """
    chunks = []

    for file_name, words in load_pdf_words().items():
        for chunk in split_words(words, chunk_size, overlap):
            chunks.append({**chunk, "source": file_name})

    return chunks


def retrieve_chunks(question, chunks, embeddings, top_k=4):
    """Step 4 (retrieve): find the `top_k` chunks most similar to the question.

    The question is embedded with the same model as the chunks, then compared
    against every chunk vector at once. Returns the best chunks, each with
    its similarity score added.
    """
    question_embedding = generate_embeddings([question])[0]

    # Embeddings are normalized, so dot product gives cosine similarity.
    # `embeddings` is (n_chunks x 384), so this gives one score per chunk.
    similarities = np.dot(embeddings, question_embedding)

    # argsort sorts ascending, so reverse it and take the first top_k indexes
    top_indexes = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            **chunks[int(index)],
            "similarity": float(similarities[index]),
        }
        for index in top_indexes
    ]


def print_sources(results):
    """Print the file name, page and similarity score of each chunk used."""
    print("\nSources:")
    for result in results:
        print(f"  - {result['source']}, page {result['page']} (similarity {result['similarity']:.3f})")


# Interactive question loop: `python src/basic_rag.py [--chunk-size 200]`
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="From-scratch PDF question answering")
    parser.add_argument("--chunk-size", type=int, default=500, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="words of overlap")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    # Load and chunk the PDFs
    chunks = load_and_chunk_pdfs(args.chunk_size, args.overlap)

    if not chunks:
        print("No PDF text found. Check the files inside the pdfs folder.")
        raise SystemExit(1)

    print(f"Total chunks created: {len(chunks)}")

    # Step 3 (embed + store): one vector per chunk, kept in a NumPy array in memory
    embeddings = generate_embeddings([chunk["text"] for chunk in chunks])

    print("\nAsk questions about your PDFs (empty line to quit).")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            break

        # Retrieve, then generate (step 5), then show where the answer came from
        results = retrieve_chunks(question, chunks, embeddings, args.top_k)
        print("\nAnswer:\n" + generate_answer(question, results))
        print_sources(results)
