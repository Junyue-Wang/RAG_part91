Part 91 RAG

This is a small RAG built from one real PDF:

```text
../source/14 CFR Part 91.pdf
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

```text
14 CFR Part 91*.pdf
```

## Generate an answer

```bash

python rag_main.py "What preflight information is required by part 91?" --openai
```
