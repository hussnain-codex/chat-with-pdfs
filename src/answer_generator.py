"""The "generate" step of the from-scratch RAG: sends the retrieved chunks and
the question to a Groq LLM and returns its answer."""

import os

from dotenv import load_dotenv
from groq import Groq

# Read GROQ_API_KEY (and optional GROQ_MODEL) from the .env file.
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

# One Groq client, reused for every question.
client = Groq(api_key=api_key)

# Which LLM to use. It can be changed in .env without touching the code.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# The rules the model must follow. This is what makes it say "I don't know"
# instead of making up an answer when the PDFs don't contain it.
SYSTEM_PROMPT = (
    "You are a PDF question-answering assistant. "
    "Answer only from the context below. "
    "If the answer isn't there, say you don't know. "
    "Never invent facts, sources or page numbers."
)


def generate_answer(question, retrieved_chunks):
    """Ask the LLM to answer `question` using only `retrieved_chunks`.

    Each chunk is a dict with "text", "source" (file name) and "page".
    The chunks are joined into one CONTEXT block, each labelled with its file
    and page, and sent with the question. Returns the answer text.
    """
    # Label every chunk with where it came from so the model can cite it.
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}, Page: {chunk['page']}\n"
            f"{chunk['text']}"
        )

    # Separate the chunks with "---" so the model sees where one ends.
    context = "\n\n---\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION: {question}"
                ),
            },
        ],
        temperature=0,  # 0 = same answer every time, no creative guessing
    )

    # The answer text is in the first (and only) choice.
    return response.choices[0].message.content


# Quick self-test: one question the context answers, one it doesn't.
if __name__ == "__main__":
    sample_chunks = [
        {
            "source": "document1.pdf",
            "page": 1,
            "text": "Ubuntu is an operating system based on Linux."
        }
    ]

    print(generate_answer("What is Ubuntu?", sample_chunks))
    print(generate_answer("What is the capital of France?", sample_chunks))  # should say "I don't know"
