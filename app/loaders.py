import os
import nest_asyncio
import pickle

from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNKS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ROOT_URL,
    SITEMAP_URL,
    USER_AGENT
)

def load_sitemap_docs():
    os.environ["USER_AGENT"] = USER_AGENT
    nest_asyncio.apply()

    loader = SitemapLoader(
        web_path=SITEMAP_URL,
        filter_urls=[ROOT_URL]
    )
    return loader.load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

def save_chunks(chunks):
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)

