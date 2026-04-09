from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHUNKS_PATH = DATA_DIR / "chunks.pickle"

ROOT_URL = "https://us-ak.com/"
SITEMAP_URL = f"{ROOT_URL}sitemap.xml"

OLLAMA_BASE_URL =  "http://127.0.0.1:11434"
OLLAMA_LLM_BASE_URL = f"{OLLAMA_BASE_URL}/v1"

EMBEDDING_MODEL = "hf.co/Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf:Q4_K_M"
RoSBRTa_EMBEDDING_MODEL = "ai-forever/ru-en-RoSBERTa"
LLM_MODEL = "gemma3:4b"

USER_AGENT = "Mozilla/5.0"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 5

JUNK_PHRASES = (
    "Введите корректный e-mail",
    "Введите корректное имя",
    "Введите корректный номер телефона",
    "Значение слишком короткое",
    "Оставить заявку",
    "Получить консультацию",
    "Получить бесплатный гайд",
    "Мною прочитаны и приняты условия политики конфиденциальности",
    "Оценка шансов на получение визы",
    "Студенческая виза в США",
    "Туристическая виза в США",
    "Шенгенская виза",
    "Университетские программы",
    "Языковые курсы",
    "Политика конфиденциальности",
    "Публичная оферта",
    "Мы используем файлы cookie",
    "Ок, принимаю",
    "Настроить Cookie",
    "Основные (необходимые) файлы cookie",
    "Аналитические файлы cookie",
    "Рекламные файлы cookie",
    "Все права защищены",
)

JUNK_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "button",
    "input",
    "textarea",
    "select",
    "label",
    "nav",
    "footer",
    ".t-footer",
    ".t-header",
    ".t-nav",
    ".t-cover__carrier",
    ".t-cookie",
    ".t396__elem",
    "[data-tilda-forms]"
)

BOILERPLATE_PATTERNS = (
    re.compile(r'файлы\s+cookie', re.IGNORECASE),
    re.compile(r'Copyright\s*©', re.IGNORECASE),
    re.compile(r'Все права защищены'),
    re.compile(r'office@'),
    re.compile(r'WhatsApp|Telegram', re.IGNORECASE),
    re.compile(r'Главная\s*→'),
    re.compile(r'^(English|Русский)$'),
    re.compile(r'^(НАВИГАЦИЯ|ДОКУМЕНТЫ|КОНТАКТЫ)$'),
    re.compile(r'Подтвердить|Disabled|Всегда включены'),
)
