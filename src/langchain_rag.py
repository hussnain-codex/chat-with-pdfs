"""LangChain RAG: PyPDFLoader -> RecursiveCharacterTextSplitter -> Chroma (on disk) -> LCEL chain -> Groq.

Every LLM/retriever call is traced in Langfuse when LANGFUSE_* keys are set in .env.
Run `python src/langchain_rag.py` for chat mode (follow-up questions are rewritten
into standalone questions using the chat history before retrieval).
"""

import argparse
import bisect
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "pdfs"
CHROMA_DIR = ROOT / "chroma_db"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


class MiniLMEmbeddings(Embeddings):
    """LangChain wrapper around the same all-MiniLM-L6-v2 model the from-scratch version uses."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


# ---------- indexing ----------

def word_count(text):
    return len(text.split())


def load_and_split(chunk_size=500, overlap=50):
    """Load every PDF and split it into ~chunk_size-word chunks.

    Pages of one file are joined before splitting so chunks can cross page
    boundaries; each chunk keeps the page it starts on (and ends on).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=word_count,
    )
    chunks = []

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        pages = PyPDFLoader(str(pdf_path)).load()

        # Join pages, remembering where each one starts in the joined text
        page_starts, parts, offset = [], [], 0
        for page in pages:
            page_starts.append(offset)
            parts.append(page.page_content)
            offset += len(page.page_content) + 2
        full_text = "\n\n".join(parts)

        # Locate each chunk ourselves: the splitter's start_index is unreliable
        # when length_function counts words instead of characters.
        search_from = 0
        for text in splitter.split_text(full_text):
            start = full_text.find(text, search_from)
            if start == -1:
                start = search_from
            search_from = start + 1
            end = start + len(text)
            first_page = bisect.bisect_right(page_starts, start)
            last_page = bisect.bisect_right(page_starts, max(start, end - 1))
            chunks.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": first_page,
                    "page_end": last_page,
                },
            ))

    return chunks


def get_vectorstore(chunk_size=500, overlap=50, rebuild=False, embeddings=None):
    """Open (or build) the Chroma collection for this chunk size, persisted to chroma_db/."""
    vectorstore = Chroma(
        collection_name=f"pdfs_{chunk_size}w",
        embedding_function=embeddings or MiniLMEmbeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    if rebuild:
        vectorstore.reset_collection()

    if not vectorstore.get(limit=1)["ids"]:
        chunks = load_and_split(chunk_size, overlap)
        if not chunks:
            raise SystemExit("No PDF text found. Check the files inside the pdfs folder.")
        print(f"Indexing {len(chunks)} chunks of ~{chunk_size} words into Chroma...")
        vectorstore.add_documents(chunks)

    return vectorstore


# ---------- chains ----------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a PDF question-answering assistant. "
     "Answer only from the context below. If the answer isn't there, say you don't know. "
     "Never invent facts, sources or page numbers.\n\nCONTEXT:\n{context}"),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rewrite the follow-up into a "
     "standalone question that can be understood without the history. "
     "Do NOT answer it. If it is already standalone, return it unchanged. "
     "Return only the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def page_label(metadata):
    first, last = metadata["page"], metadata.get("page_end", metadata["page"])
    return str(first) if first == last else f"{first}-{last}"


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata['source']}, Page: {page_label(doc.metadata)}\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(retriever, llm):
    """{context: retriever | format_docs, question: passthrough} | prompt | llm | parser

    The retrieved docs are also kept in the output so sources can be shown.
    """
    answer_chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(x["docs"])),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(),
    ).assign(answer=answer_chain)


def build_rewrite_chain(llm):
    return REWRITE_PROMPT | llm | StrOutputParser()


def build_chains(chunk_size=500, overlap=50, rebuild=False, k=4):
    vectorstore = get_vectorstore(chunk_size, overlap, rebuild)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(model=MODEL, temperature=0, max_tokens=1024, max_retries=8)
    return build_rag_chain(retriever, llm), build_rewrite_chain(llm)


def sources_of(docs):
    """Unique 'file, page' labels in retrieval order."""
    seen = []
    for doc in docs:
        label = f"{doc.metadata['source']}, page {page_label(doc.metadata)}"
        if label not in seen:
            seen.append(label)
    return seen


# ---------- Langfuse ----------

def langfuse_handler():
    """Return a Langfuse CallbackHandler, or None if keys are not configured."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("Langfuse keys not set in .env - tracing disabled.")
        return None
    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def flush_langfuse():
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse import get_client
        get_client().flush()


def run_config(handler, run_name, session_id=None, tags=None):
    config = {"run_name": run_name, "callbacks": [handler] if handler else []}
    metadata = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if tags:
        metadata["langfuse_tags"] = tags
    if metadata:
        config["metadata"] = metadata
    return config


# ---------- chat ----------

def chat(rag_chain, rewrite_chain, handler):
    session_id = str(uuid.uuid4())
    history = []

    print("\nChat with your PDFs (empty line to quit).")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            break

        standalone = question
        if history:
            standalone = rewrite_chain.invoke(
                {"chat_history": history, "question": question},
                config=run_config(handler, "rewrite-question", session_id),
            ).strip()
            if standalone != question:
                print(f"(searching for: {standalone})")

        result = rag_chain.invoke(standalone, config=run_config(handler, "rag-answer", session_id))

        print(f"\nBot: {result['answer']}")
        print("\nSources:")
        for source in sources_of(result["docs"]):
            print(f"  - {source}")

        history += [HumanMessage(question), AIMessage(result["answer"])]
        history = history[-10:]  # keep the last 5 turns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain PDF chat with Langfuse tracing")
    parser.add_argument("--chunk-size", type=int, default=500, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="words of overlap")
    parser.add_argument("--k", type=int, default=4, help="chunks to retrieve")
    parser.add_argument("--rebuild", action="store_true", help="re-index the PDFs")
    parser.add_argument("--question", "-q", help="ask a single question and exit")
    args = parser.parse_args()

    rag_chain, rewrite_chain = build_chains(args.chunk_size, args.overlap, args.rebuild, args.k)
    handler = langfuse_handler()

    try:
        if args.question:
            result = rag_chain.invoke(args.question, config=run_config(handler, "rag-answer"))
            print(f"\nAnswer:\n{result['answer']}\n\nSources:")
            for source in sources_of(result["docs"]):
                print(f"  - {source}")
        else:
            chat(rag_chain, rewrite_chain, handler)
    finally:
        flush_langfuse()
