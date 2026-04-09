from langchain_chroma import Chroma
from app.config import CHROMA_DIR, RETRIEVER_K
from app.embeddings import get_rosberta_embeddings

def build_vectorstore(documents):
    embeddings = get_rosberta_embeddings()
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


def load_vectorstore():
    embeddings = get_rosberta_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

def get_retriever():
    return load_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K})

def get_mmr_retriever():
    mmr_retriever = load_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
        },
    )
    return mmr_retriever