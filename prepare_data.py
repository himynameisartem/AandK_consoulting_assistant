from app.loaders import (
    clean_docs,
    load_recursive_sitemap,
    load_sitemap_docs,
    save_chunks,
    split_docs,
)
from app.vetorstore import build_vectorstore


def main():
    print("Loading sitemap documents...")
    sitemap_docs = load_sitemap_docs()
    print(f"Sitemap docs: {len(sitemap_docs)}")

    print("Loading recursive documents...")
    recursive_docs = load_recursive_sitemap()
    print(f"Recursive docs: {len(recursive_docs)}")

    docs = sitemap_docs + recursive_docs

    print("Cleaning documents...")
    docs = clean_docs(docs)
    print(f"Clean docs: {len(docs)}")

    print("Splitting into chunks...")
    chunks = split_docs(docs)
    print(f"Chunks: {len(chunks)}")

    print("Saving chunks...")
    save_chunks(chunks)

    print("Building vector store...")
    build_vectorstore(chunks)

    print("Done.")


if __name__ == "__main__":
    main()
