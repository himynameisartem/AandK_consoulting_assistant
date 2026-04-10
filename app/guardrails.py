import re
from typing import Optional
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action

from app.prompts import build_prompt
from app.rag import build_rag_chain

BLOCKED_PATTERNS = [
    r"взрыв|взрывчат",
    r"оружи[ея]",
    r"наркотик",
    r"взлом|хакер|фишинг",
    r"вредоносн",
    r"убий|убить|убивать",
    r"суицид|самоубийств",
]


def make_actions(rag_chain):

    @action(is_system_action=True)
    async def ask_rag(context: Optional[dict] = None, **kwargs):
        user_message = context.get("last_user_message", "") if context else ""
        if not user_message:
            return "Не удалось определить вопрос пользователя."
        return rag_chain.invoke(user_message)

    @action(is_system_action=True)
    async def check_output(context: Optional[dict] = None, **kwargs):
        bot_response = context.get("last_bot_message", "") if context else ""
        text_lower = bot_response.lower()
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, text_lower):
                return False
        return True

    return ask_rag, check_output


def build_rails(rag_chain=None) -> LLMRails:
    if rag_chain is None:
        prompt = build_prompt()
        rag_chain = build_rag_chain(prompt)

    config_path = Path(__file__).resolve().parent.parent / "config"
    config = RailsConfig.from_path(str(config_path))

    rails = LLMRails(config)
    ask_rag, check_output = make_actions(rag_chain)
    rails.register_action(ask_rag, name="ask_rag")
    rails.register_action(check_output, name="check_output")

    return rails


async def ask(rails: LLMRails, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    response = await rails.generate_async(messages=messages)
    return response["content"]