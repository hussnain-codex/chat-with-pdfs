
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

client = Groq(api_key=api_key)


def generate_answer(question, retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}, Page: {chunk['page']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    response = client.chat.completions.create(
       model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a PDF question-answering assistant. "
                    "Answer using only the supplied context. "
                    "If the answer is not supported by the context, "
                    "say: I don't know based on the provided PDF content. "
                    "Cite supporting sources using the filename and page number. "
                    "Never invent a source or page number."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PDF CONTEXT:\n{context}\n\n"
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

    question = "What is Ubuntu?"
    print(generate_answer(question, sample_chunks))