# Citation-Grounded RAG for Scientific Literature

A Retrieval-Augmented Generation (RAG) system for asking natural-language questions across a library of research papers and getting answers grounded in — and traceable back to — the exact source paper and page. Built to explore the Agentic AI literature, but works with any collection of PDF research papers.

## Overview

Large language models are fluent, but they aren't reliable narrators of their own sources. This project addresses that by separating *generation* from *citation*: instead of trusting the LLM to accurately recall or cite what it read, the retrieval step is tracked programmatically, so every answer comes with a verifiable list of the papers and pages it was actually built from.

The system ingests multiple PDFs, chunks and embeds their content, stores them in a vector database, and retrieves the most relevant passages for each question. Those passages are numbered and fed to the LLM, which generates an answer referencing them — and a source list built directly from retrieval metadata is printed alongside every answer, independent of what the model claims to have cited.

## Features

- **Multi-paper ingestion** — drop any number of PDFs into `papers/` and the notebook ingests all of them in one pass
- **Citation grounding** — every answer is paired with a deduplicated, accurate list of source papers and page numbers, generated from retrieval metadata rather than the LLM's own claims
- **Semantic retrieval** — chunked, embedded, and indexed for similarity search rather than keyword matching
- **Swappable LLM backend** — runs on any Hugging Face Inference-served model; not locked to a single provider

## Tools & Stack

| Purpose | Tool |
|---|---|
| Orchestration / chaining | [LangChain](https://www.langchain.com/) (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`) |
| LLM inference | [Hugging Face Inference API](https://huggingface.co/docs/api-inference/) via `langchain-huggingface` |
| Embeddings | `sentence-transformers` (`all-mpnet-base-v2`) |
| Vector store | [Chroma](https://www.trychroma.com/) (`langchain-chroma`) |
| PDF parsing | `pypdf` |
| Core ML | `transformers`, `torch` |

## Project Structure

```
.
├── papers/                                          # add your own research PDFs here (empty by default)
├── RAG-Powered_Research_Literature_Assistant.ipynb   # main notebook: ingestion, retrieval, RAG chain
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add papers**
   Place any PDF research papers you want to query into the `papers/` folder.

4. **Set your Hugging Face token**
   Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a standard "Read" token is simplest — fine-grained tokens need the *"Make calls to Inference Providers"* permission enabled).

   - **In Colab:** add it via the 🔑 secrets manager as `HF_TOKEN`, then:
     ```python
     from google.colab import userdata
     hf_token = userdata.get("HF_TOKEN")
     ```
   - **Locally:** set it as an environment variable before launching:
     ```bash
     export HF_TOKEN=your_token_here
     ```

5. **Run the notebook**
   Run `RAG-Powered_Research_Literature_Assistant.ipynb` top to bottom.

## Usage

Once set up, ask questions using the `ask()` helper defined in the notebook:

```python
ask("What agentic reasoning strategies are proposed across these papers?")
```

Each call prints the generated answer followed by the papers and page numbers it was drawn from.

## Limitations & Future Work

- Retrieval quality depends on chunking strategy — large tables or multi-column layouts in some PDFs may not extract cleanly
- Currently uses a single retriever configuration (top-k similarity search); a reranking step would likely improve relevance on larger paper collections
- No evaluation harness yet — adding faithfulness/relevancy metrics (e.g. RAGAS) would strengthen confidence in answer quality
- No persistent storage — the vector store is rebuilt each run; adding persistence would speed up repeated use

## License

MIT
