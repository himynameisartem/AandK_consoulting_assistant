from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from huggingface_hub import login

login()

from app.config import EMBEDDING_MODEL, OLLAMA_BASE_URL, RoSBRTa_EMBEDDING_MODEL

class PrefixedEmbeddings(Embeddings):
    def __init__(self, base, query_prefix="", doc_prefix=""):
        self.base = base
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts):
        texts_prefixed = [self.doc_prefix + t for t in texts]
        return self.base.embed_documents(texts_prefixed)

    def embed_query(self, text):
        return self.base.embed_query(self.query_prefix + text)


def get_rosberta_embeddings():
    base_embeddings = HuggingFaceEmbeddings(
        model_name=RoSBRTa_EMBEDDING_MODEL,
        encode_kwargs={"batch_size": 8}
    )
    return PrefixedEmbeddings(
        base_embeddings,
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
    )


def get_embedding():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )