from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHUNKS_PATH = DATA_DIR / "chunks.pickle"

ROOT_URL = "https://us-ak.com/"
SITEMAP_URL = f"{ROOT_URL}sitemap.xml"

OLLAMA_BASE_URL =  "http://127.0.0.1:11434"
OLLAMA_LLM_BASE_URL = f"{OLLAMA_BASE_URL}/v1"

EMBEDDING_MODEL = "hf.co/Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf:Q4_K_M"
LLM_MODEL = "gemma3:4b"

USER_AGENT = "Mozilla/5.0"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVER_K = 5
