from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.llm import get_llm
from app.vetorstore import get_mmr_retriever

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

def build_query_rewriter(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты помощник, который улучшает поисковые запросы для RAG-системы. "
         "Переформулируй вопрос пользователя так, чтобы он лучше совпадал с текстами на сайте "
         "консалтинговой компании A&K (языковые курсы, визы, программы в США). "
         "Убери разговорные обороты, сделай запрос конкретным и информативным. "
         "Верни ТОЛЬКО переформулированный запрос, без пояснений."),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

def build_rag_chain(prompt):
    retriever = get_mmr_retriever()
    llm = get_llm()
    query_rewriter = build_query_rewriter(llm)

    def retrieve_with_rewrite(question: str):
        rewritten = query_rewriter.invoke({"question": question})
        print(f"[Rewritten query]: {rewritten}")
        docs = retriever.invoke(rewritten)
        return format_docs(docs)

    return (
        {
            "context": lambda q: retrieve_with_rewrite(q),
            "question": RunnablePassthrough(),
            "history": lambda _: [],
        }
    | RunnableLambda(ensure_context)
    | prompt
    | llm
    | StrOutputParser()
).with_config(run_name="rag_chain")