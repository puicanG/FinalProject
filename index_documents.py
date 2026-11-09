# This program reads all the pdf docs and txt files from a folder
# and saves the text in ChromaDB.
# command in terminal : uv run chroma run --path chroma_data --port 8000

import chromadb
import pymupdf4llm
from pathlib import Path
from sentence_transformers import SentenceTransformer

DOCUMENTS_FOLDER = Path("documents")

# the name of the collection inside ChromaDB
COLLECTION_NAME = "finalproject_docs"

# the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# chunk settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# this function splits a text into small pieces (chunks)
def split_into_chunks(text):
    chunks = []
    start = 0

    while start < len(text):
        # take a piece of text
        chunk = text[start:start + CHUNK_SIZE].strip()

        # we only keep it if it is not empty
        if chunk:
            chunks.append(chunk)

        # move forward, but keep some overlap with the old chunk
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# this function returns a list of (page number, page text)
def get_pages(file_path):
    # for pdf files we use pymupdf4llm
    # page_chunks=True gives us one markdown text for every page
    if file_path.suffix.lower() == ".pdf":
        pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

        result = []
        for page in pages:
            page_number = page["metadata"]["page_number"]
            page_text = page["text"]
            result.append((page_number, page_text))

        return result

    # for txt files we read the file as normal text
    # our converted files use the \f character between pages
    text = file_path.read_text(encoding="utf-8")
    pages = text.split("\f")

    result = []
    for page_number, page_text in enumerate(pages, start=1):
        # skip empty pages
        if page_text.strip():
            result.append((page_number, page_text))

    return result


def main():
    # load the embedding model
    print("Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # connect to the ChromaDB server
    # the server must be started in another terminal
    client = chromadb.HttpClient(host="localhost", port=8000)

    # get the collection, or create it if it does not exist
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # collect all pdf and txt files from the folder
    files = []
    for file_path in sorted(DOCUMENTS_FOLDER.iterdir()):
        if file_path.is_file() and (
            file_path.suffix.lower() == ".pdf" or file_path.suffix.lower() == ".txt"
        ):
            files.append(file_path)

    # remove chunks of files that are no longer in the folder
    current_sources = set()
    for file_path in files:
        current_sources.add(file_path.name)

    old_items = collection.get(include=["metadatas"])
    old_sources = set()
    for metadata in old_items["metadatas"]:
        old_sources.add(metadata["source"])

    for source in old_sources:
        if source not in current_sources:
            collection.delete(where={"source": source})
            print("Removed old chunks for", source)

    # go through every file
    for file_path in files:
        print("Indexing", file_path.name)

        # here we keep the data for one file
        ids = []
        documents = []
        metadatas = []

        # go through every page of the file
        for page_number, page_text in get_pages(file_path):
            # split the page text into chunks
            chunks = split_into_chunks(page_text)

            # we start counting chunks from 1 on every page
            chunk_number = 1

            # go through every chunk of this page
            for chunk in chunks:
                # make a unique id for this chunk
                ids.append(f"{file_path.name}:p{page_number}:c{chunk_number}")

                # save the text of the chunk
                documents.append(chunk)

                # save the metadata of the chunk
                metadatas.append(
                    {
                        "source": file_path.name,
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )

                chunk_number += 1

        # if we index the same file again, first remove its old chunks
        collection.delete(where={"source": file_path.name})

        # only insert if the file had some text
        if len(documents) > 0:
            # make an embedding for every chunk
            embeddings = model.encode(documents, normalize_embeddings=True).tolist()

            # save everything in ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    # show the total number of chunks in the collection
    print("Done. The collection has", collection.count(), "chunks.")


# start the program
if __name__ == "__main__":
    main()
