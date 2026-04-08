from langchain_openai import ChatOpenAI
from app.config import LLM_MODEL, OLLAMA_LLM_BASE_URL

def get_llm():
    return ChatOpenAI(
        api_key="None",
        base_url=OLLAMA_LLM_BASE_URL,
        model=LLM_MODEL,

        temperature=0.2,
        max_tokens=512,
        top_p=0.9,
    )