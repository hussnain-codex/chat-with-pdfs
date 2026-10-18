
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

client = Groq(api_key=api_key)

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = (
    "You are a PDF question-answering assistant. "
    "Answer only from the context below. "
    "If the answer isn't there, say you don't know. "
    "Never invent facts, sources or page numbers."
)


def generate_answer(question, retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}, Page: {chunk['page']}\n"
            f"{chunk['text']}"
        )

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
        temperature=0,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    sample_chunks = [
        {
            "source": "document1.pdf",
            "page": 1,
            "text": "Ubuntu is an operating system based on Linux."
        }
    ]

    print(generate_answer("What is Ubuntu?", sample_chunks))
    print(generate_answer("What is the capital of France?", sample_chunks))
