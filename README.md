# Chat with your PDFs (basic RAG)

Ask questions about a folder of PDFs and get answers grounded in the documents, with the file name and page number of every source. The same pipeline is built twice, first by hand and then with LangChain. Every LangChain call is traced in Langfuse.

```
load → chunk → embed → store → retrieve → generate
pypdf   own splitter / RecursiveCharacterTextSplitter
        all-MiniLM-L6-v2   NumPy / Chroma   cosine top-4   Groq
```

## Project layout

| File | What it does |
|---|---|
| `src/basic_rag.py` | From scratch: pypdf, a hand-written word splitter, MiniLM embeddings in a NumPy array, cosine top-4, Groq |
| `src/embeddings.py` | Local `all-MiniLM-L6-v2` embedding helper (normalized vectors) |
| `src/answer_generator.py` | Groq call with the "answer only from the context" prompt |
| `src/langchain_rag.py` | LangChain version: PyPDFLoader, RecursiveCharacterTextSplitter, Chroma on disk, LCEL chain, Langfuse tracing, chat mode |
| `src/evaluate.py` | Chunk-size experiment (200 / 500 / 1000 words) over `questions.json` |
| `questions.json` | 15 test questions with known answers |
| `pdfs/` | The source PDFs |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env          # then fill in the keys
```

`.env` needs:

- `GROQ_API_KEY` from https://console.groq.com
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`. Create a free project at https://cloud.langfuse.com, then go to Project settings → API keys.

## Data

| PDF | Type | Pages |
|---|---|---|
| `document1.pdf` | Lab manual: installing VirtualBox and Ubuntu | 26 |
| _add 2–4 more_ | government policy / product manual / research paper | 20–100 |

## Usage

### 1. From scratch

```bash
python src/basic_rag.py                    # 500-word chunks, 50 overlap, top 4
python src/basic_rag.py --chunk-size 200 --overlap 20
```

The splitter is a sliding window over `(word, page)` pairs, so a chunk can cross a page boundary and still knows which pages it covers (for example `page 7-8`). Embeddings are normalized, so a dot product with the question vector gives cosine similarity.

### 2. LangChain + Langfuse + chat mode

```bash
python src/langchain_rag.py                 # interactive chat
python src/langchain_rag.py -q "your question"
python src/langchain_rag.py --rebuild       # re-index after adding PDFs
```

- Chunks are stored in a Chroma collection under `chroma_db/`, one collection per chunk size.
- The retriever is `vectorstore.as_retriever(search_kwargs={"k": 4})`.
- The chain has the shape `{context: retriever | format_docs, question: passthrough} | prompt | llm | parser`. The retrieved documents are also returned so sources can be printed under every answer.
- **Chat mode:** after the first turn, each follow-up (for example "what about the second one?") is rewritten into a standalone question using the last 5 turns of history, and that question is used for retrieval. The rewritten question is printed as `(searching for: ...)`.
- **Langfuse:** a `langfuse.langchain.CallbackHandler` is passed in each chain's `config`. A chat is grouped into one Langfuse session. Each trace shows the retriever span with the retrieved chunks, the final prompt, latency and token counts.

Example:

```
You: What software is being installed in this guide?
Bot: The guide shows how to install VirtualBox and then use it to install Ubuntu as a virtual machine.
Sources:
  - document1.pdf, page 1-7
  - document1.pdf, page 7-16

You: what is the first step for it?
(searching for: What is the first step to install VirtualBox and set up Ubuntu as a virtual machine?)
Bot: The first step is to download the VirtualBox installer from the official website ...

You: Who won the 2018 World Cup?
Bot: I don't know.
```

## Chunk-size experiment

1. Fill in the 15 questions and expected answers in `questions.json`.
2. Run `python src/evaluate.py run`. This answers every question at 200, 500 and 1000 words, using 10% overlap, and writes `results/chunk_size_eval.csv`. In Langfuse these traces are tagged `chunk-200w`, `chunk-500w` and `chunk-1000w`.
3. Open the CSV and score each row by hand, putting `y` or `n` in the `correct` column.
4. Run `python src/evaluate.py summarize`. This fills in the table below.

<!-- chunk-table:start -->
| Chunk size (words) | Overlap (words) | Correct | Accuracy |
|---|---|---|---|
| 200 | 20 | _ / 15 | _ |
| 500 | 50 | _ / 15 | _ |
| 1000 | 100 | _ / 15 | _ |
<!-- chunk-table:end -->

**Observations:** _(fill in after scoring)_ Small chunks give precise matches but can cut an answer in half. Large chunks keep the context together but dilute the embedding and fill the prompt with irrelevant text.
