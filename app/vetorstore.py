from langchain_community.vectorstores import Chroma
from app.config import CHROMA_DIR, RETRIEVER_K
from app.embeddings import get_embedding

def build_vectorstore(documents):
    embeddings = get_embedding()
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


def load_vectorstore():
    embeddings = get_embedding()
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

def get_retriever():
    return load_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K})