from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.llm import get_llm
from app.vetorstore import get_retriever, get_mmr_retriever

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

def format_docs(docs, max_chars: int = 8000):
    formatted = []
    total_len = 0

    for doc in docs:
        source = doc.metadata.get("source", "unknown_source")
        page = doc.metadata.get("page", None)

        header = f"Source: {source}"
        if page is not None:
            header += f" | Page: {page}"

        text = doc.page_content.strip()
        block = f"{header}\n{text}"

        # если следующий блок слишком раздует контекст – останавливаемся
        if total_len + len(block) > max_chars:
            break

        formatted.append(block)
        total_len += len(block)

    return "\n\n---\n\n".join(formatted)

def ensure_context(input_dict: dict) -> dict:
    context = input_dict.get("context", "").strip()
    if not context:
        input_dict["context"] = (
            "Контекст пуст: ретривер не нашёл ни одного подходящего фрагмента. "
            "Если ответ важен, лучше явно сказать пользователю об этом."
        )
    return input_dict

def build_rag_chain(prompt):
    retriever = get_mmr_retriever()
    llm = get_llm()

    return (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "history": lambda _: [],  # пока истории нет – передаём пустой список
    }
    | RunnableLambda(ensure_context)   # защита от пустого контекста
    | prompt
    | llm
    | StrOutputParser()
).with_config(run_name="rag_chain")