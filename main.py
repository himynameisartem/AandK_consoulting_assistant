import argparse

from app.prompts import build_prompt
from app.rag import build_rag_chain


def parse_args():
    parser = argparse.ArgumentParser(description="A&K Consulting Assistant")
    parser.add_argument(
        "--phoenix",
        action="store_true",
        help="Launch Phoenix tracing UI for the current session.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.phoenix:
        from app.observability import launch_phoenix

        session = launch_phoenix()
        print(f"Phoenix: {session.url}")

    print("A&K Consulting Assistant")
    print("Введите вопрос или 'exit' для выхода.\n")

    rag_chain = None

    while True:
        question = input("Вопрос: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Выход.")
            break

        if not question:
            continue

        if rag_chain is None:
            prompt = build_prompt()
            rag_chain = build_rag_chain(prompt)

        try:
            answer = rag_chain.invoke(question)
            print(f"\nОтвет:\n{answer}\n")
        except Exception as e:
            print(f"\nОшибка: {e}\n")


if __name__ == "__main__":
    main()
