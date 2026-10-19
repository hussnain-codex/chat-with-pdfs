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

| PDF | Type | Pages | Words |
|---|---|---|---|
| `10_Education.pdf` | Government policy: Pakistan Economic Survey 2025, ch. 10 | 22 | ~8,900 |
| `11_Health_and_Nutrition.pdf` | Government policy: Pakistan Economic Survey 2025, ch. 11 | 22 | ~8,800 |
| `17_Climate_Change.pdf` | Government policy: Pakistan Economic Survey 2025, ch. 17 | 24 | ~11,800 |
| `virtualbox_ubuntu_lab_manual.pdf` | Manual: installing VirtualBox and Ubuntu (screenshot-heavy) | 26 | ~1,400 |
| `rag_survey.pdf` | Research paper: *Retrieval-Augmented Generation for Large Language Models: A Survey* (Gao et al., arXiv 2312.10997) | 21 | ~15,900 |

The 15 test questions are in `questions.json`. There are 3 for the manual, 8 for the Economic Survey chapters, 3 for the RAG survey, and 1 that isn't answered in any PDF, which checks that the bot says "I don't know".

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
| 200 | 20 | 13 / 15 | 87% |
| 500 | 50 | 11 / 15 | 73% |
| 1000 | 100 | 11 / 15 | 73% |
<!-- chunk-table:end -->

Setup: all-MiniLM-L6-v2 embeddings, top 4 chunks, Groq `openai/gpt-oss-20b` at temperature 0. Every answer is in `results/chunk_size_eval.csv`.

**Observations**

- **200 words scored best (13/15).** Short chunks embed one idea each, so specific facts like a date, a rank or a temperature land in a chunk that closely matches the question.
- **Larger chunks lost specific facts.** At 500 and 1,000 words, "HDI rank 168" (Q5) and "Green Taxonomy notified on 15 September 2025" (Q11) are buried in long chunks about other topics. Their embeddings no longer match the question, so the model correctly said "I don't know".
- **1,000 words gave the only made-up-looking answer.** Q9 answered "Rs 19.375 billion", which is the health PSDP allocation, not total health spending. With 4,000 words of context full of numbers, the model picked the wrong figure.
- **Small chunks can miss context.** At 200 words, Q1 (minimum RAM) failed because the lab manual has almost no text per page and "Minimum size is 1024MB" sits in a short fragment without the words "memory" or "virtual machine". It only succeeded at 1,000 words, where the whole procedure is in one chunk.
- **Q8 (registered doctors) failed at every size.** The number sits in a dense table paragraph. At 500 and 1,000 words the model returned an empty answer: `gpt-oss` is a reasoning model and sometimes uses up its output on reasoning.
- **The "I don't know" test passed at every size.** Q15 (GPT-4 Turbo context length) is not in the PDFs, and every run refused to guess.
- **Groq free-tier limit:** four 1,000-word chunks come to about 8,000 tokens, which hits the free plan's 8,000 tokens-per-minute cap. Answers are capped with `max_tokens=1024` so those requests fit.

**Conclusion:** 200–500-word chunks suit fact lookups in reports. Large chunks help only with short, low-text documents like the lab manual.
