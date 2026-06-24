import re

from app.config import LOCAL_SAFETY_PATTERNS


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def classify_text(text: str, kind: str = "input") -> str:
    normalized = _normalize(text)

    for pattern in LOCAL_SAFETY_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return "unsafe"

    return "safe"


def is_safe(text: str, kind: str = "input") -> bool:
    return classify_text(text, kind=kind) == "safe"
