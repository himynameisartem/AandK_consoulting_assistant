from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.llm import get_llm
from app.vetorstore import get_retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(prompt):
    retriever = get_retriever()
    llm = get_llm()

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )