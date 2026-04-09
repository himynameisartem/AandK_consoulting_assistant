import os
import pickle
import re

import nest_asyncio
from bs4 import BeautifulSoup
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNKS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ROOT_URL,
    SITEMAP_URL,
    USER_AGENT,
    JUNK_PHRASES,
    JUNK_SELECTORS,
    BOILERPLATE_PATTERNS
)

def load_sitemap_docs():
    os.environ["USER_AGENT"] = USER_AGENT
    nest_asyncio.apply()

    loader = SitemapLoader(
        web_path=SITEMAP_URL,
        filter_urls=[ROOT_URL]
    )
    return loader.load()

def load_recursive_sitemap():
    recursive_loader = RecursiveUrlLoader(
        url=ROOT_URL,
        max_depth=3,
        prevent_outside=True
    )
    return recursive_loader.load()


def is_serialized_garbage(text: str) -> bool:
    markers = [
        r'\\u[0-9a-fA-F]{4}',
        r'https:\\/\\/',
        r'"li_gallery"',
        r'"img"',
        r'"alt"',
        r'"li_name"',
        r'"li_radcb"',
        r'tildacdn',
        r'button=',
        r'\bu[0-9a-fA-F]{4}\b',
        r'"li_type"',
        r'"li_ph"',
        r'"li_req"',
        r'"li_nm"',
        r'"lid":"',
        r'\[{"lid":',
    ]
    return sum(bool(re.search(p, text)) for p in markers) >= 2


# def is_boilerplate_line(line: str) -> bool:
#     line = line.strip()
#     return any(p.search(line) for p in BOILERPLATE_PATTERNS)


def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")

    for selector in JUNK_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    cleaned = "\\n".join(soup.stripped_strings)

    # for tag in soup.find_all(attrs={"data-tilda-forms": True}):
    #     tag.decompose()
    #
    # lines = list(soup.stripped_strings)

    # lines = [l for l in lines if not is_boilerplate_line(l)]

    # lines = [l for l in lines if not re.search(r'\bu[0-9a-fA-F]{4}\b', l)]
    # lines = [l for l in lines if not re.search(r'"li_\w+"', l)]
    #
    # cleaned = "\n".join(lines)

    for phrase in JUNK_PHRASES:
        cleaned = cleaned.replace(phrase, "\\n")

    cleaned = re.sub(r"[ \\t]+", " ", cleaned)
    cleaned = re.sub(r"\\n{2,}", "\\n\\n", cleaned).strip()
    return cleaned



def clean_docs(docs):
    cleaned_docs = []

    for doc in docs:

        if is_serialized_garbage(doc.page_content):
            continue

        cleaned_text = clean_html(doc.page_content)
        if len(cleaned_text) < 200:
            continue

        cleaned_docs.append(
            Document(
                page_content=cleaned_text,
                metadata=doc.metadata,
            )
        )

    return cleaned_docs


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

