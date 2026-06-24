import re
from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import LLAMA_GUARD_MODEL


INPUT_SAFETY_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a safety classifier. Decide whether the user message is safe or unsafe.
Reply with exactly one word: safe or unsafe.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Message:
{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

OUTPUT_SAFETY_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a safety classifier. Decide whether the assistant answer is safe or unsafe.
Reply with exactly one word: safe or unsafe.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Answer:
{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


@lru_cache(maxsize=1)
def _load_guard():
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_GUARD_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLAMA_GUARD_MODEL)
    return tokenizer, model


def _normalize_decision(text: str) -> str:
    match = re.search(r"\b(safe|unsafe)\b", text.lower())
    return match.group(1) if match else "safe"


def classify_text(text: str, kind: str = "input") -> str:
    template = INPUT_SAFETY_TEMPLATE if kind == "input" else OUTPUT_SAFETY_TEMPLATE
    prompt = template.format(text=text)

    try:
        tokenizer, model = _load_guard()
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return _normalize_decision(generated)
    except Exception as exc:
        print(f"[Safety] Llama Guard unavailable, allowing request by fallback: {exc}")
        return "safe"


def is_safe(text: str, kind: str = "input") -> bool:
    return classify_text(text, kind=kind) == "safe"

