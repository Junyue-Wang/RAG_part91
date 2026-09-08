import argparse
import os
from pathlib import Path

PROJECT_FOLDER = Path(__file__).parent
DEFAULT_DB_FOLDER = PROJECT_FOLDER / "chroma_db"
MODEL_CACHE_FOLDER = PROJECT_FOLDER / ".model_cache" / "all-MiniLM-L6-v2"
COLLECTION_NAME = "cfr_part_91"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def search_chunks(question, db_folder=DEFAULT_DB_FOLDER, top_k=4):
    import chromadb
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    embedding_function = ONNXMiniLM_L6_V2()
    embedding_function.DOWNLOAD_PATH = MODEL_CACHE_FOLDER

    client = chromadb.PersistentClient(path=str(db_folder))
    collection_names = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME not in collection_names:
        raise FileNotFoundError("Chroma database not found. Run: python build_index.py")

    collection = client.get_collection(
        COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    query_result = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for text, metadata, distance in zip(
        query_result["documents"][0],
        query_result["metadatas"][0],
        query_result["distances"][0],
        strict=True,
    ):
        results.append(
            {
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": distance,
            }
        )
    return results


def make_prompt(question, results):
    context_parts = []
    for number, result in enumerate(results, start=1):
        label = f"[Source {number}: {result['source']}, Page {result['page']}]"
        context_parts.append(f"{label}\n{result['text']}")

    context = "\n\n".join(context_parts)
    return f"""You are a private pilot who is trying to answer a question about the Federal Aviation Regulations (FARs). 
    Answer the question using only the sources below.
    If the sources do not contain the answer, say that you could not find it.
    Cite factual statements with the source label exactly as shown to comply with the FARs.

Question: {question}

Sources:
{context}
"""


def generate_answer(question, results):
    """Send the retrieved chunks to an OpenAI model."""
    from openai import OpenAI

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
    response = OpenAI().responses.create(
        model=model,
        input=make_prompt(question, results),
    )
    return response.output_text


def print_sources(results, show_context=False):
    print("\nRetrieved sources:")
    for number, result in enumerate(results, start=1):
        print(
            f"{number}. {result['source']}, Page {result['page']} "
            f"(distance={result['distance']:.3f})"
        )
        if show_context:
            print(f"   {result['text']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_FOLDER)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        results = search_chunks(args.question, args.db, args.top_k)
        if args.openai:
            print(generate_answer(args.question, results))
        else:
            print("Retrieval complete. Add --openai to generate an answer.")
        print_sources(results, args.show_context)
    except (FileNotFoundError, ImportError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
