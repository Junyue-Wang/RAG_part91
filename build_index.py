import argparse
import re
from pathlib import Path

PROJECT_FOLDER = Path(__file__).parent
SOURCE_FOLDER = PROJECT_FOLDER.parent / "source"
DEFAULT_DB_FOLDER = PROJECT_FOLDER / "chroma_db"
MODEL_CACHE_FOLDER = PROJECT_FOLDER / ".model_cache" / "all-MiniLM-L6-v2"
COLLECTION_NAME = "cfr_part_91"


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(text, chunk_words=160, overlap_words=30):
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def find_part_91_pdf(source_folder):
    matches = sorted(source_folder.glob("14 CFR Part 91*.pdf"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one '14 CFR Part 91*.pdf' in {source_folder}, found {len(matches)}"
        )
    return matches[0]


def read_pdf_chunks(pdf_file):
    import pymupdf

    chunks = []
    with pymupdf.open(pdf_file) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = clean_text(page.get_text("text"))
            for chunk_number, chunk_text in enumerate(split_into_chunks(page_text), start=1):
                chunks.append(
                    {
                        "id": f"page-{page_number}-chunk-{chunk_number}",
                        "text": chunk_text,
                        "source": pdf_file.name,
                        "page": page_number,
                    }
                )
    return chunks


def save_to_chroma(chunks, db_folder):
    """Embed the chunks and save them in a local Chroma database."""
    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    embedding_function = ONNXMiniLM_L6_V2()
    embedding_function.DOWNLOAD_PATH = MODEL_CACHE_FOLDER

    client = chromadb.PersistentClient(path=str(db_folder))
    collection_names = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
    )

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[
                {"source": chunk["source"], "page": chunk["page"]} for chunk in batch
            ],
        )
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    return collection.count()


def build_database(source_folder=SOURCE_FOLDER, db_folder=DEFAULT_DB_FOLDER):
    pdf_file = find_part_91_pdf(source_folder)
    print(f"Reading {pdf_file.name}")
    chunks = read_pdf_chunks(pdf_file)
    print(f"Created {len(chunks)} chunks")
    count = save_to_chroma(chunks, db_folder)
    print(f"Saved {count} vectors to {db_folder}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_FOLDER)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_FOLDER)
    args = parser.parse_args()
    build_database(args.source, args.db)


if __name__ == "__main__":
    main()
