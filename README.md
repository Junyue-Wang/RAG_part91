Part 91 RAG

This is a small RAG built from one real PDF:

```text
../source/14 CFR Part 91 (up to date as of 9-03-2026).pdf
```


## Structure

```text
Part 91 PDF
  -> PyMuPDF extracts text
  -> pages are split into overlapping chunks
  -> Chroma creates an embedding for every chunk
  -> Chroma saves vectors, text, filename, and page number
  -> the question is embedded and matched against the vectors
  -> the top chunks are added to an OpenAI prompt
  -> the model writes an answer with citations
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Build the vector database

```bash
python build_index.py
```

The first run downloads Chroma's local MiniLM embedding model into `.model_cache/`.
The resulting vector database is saved in `chroma_db/`. Both folders are ignored
by Git because they can be rebuilt.

```text
14 CFR Part 91*.pdf
```

## Generate an answer

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_CHAT_MODEL="gpt-5-mini"

python rag_main.py "What preflight information is required by 91.103?" --openai
```

The default `--top-k` value is 4. Page numbers are physical PDF pages and can be
different from page numbers printed inside the document.

