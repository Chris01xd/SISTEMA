"""Command line: `python -m hospital_assistant "question"` or no argument for a chat."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hospital_assistant.agent import ask, build_agent, tool_calls_made
from hospital_assistant.config import Settings
from hospital_assistant.data import DatasetError, load_dataset


def _uses_anthropic(model: str) -> bool:
    return ":" not in model or model.startswith("anthropic:")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hospital_assistant", description="Ask questions about the hospital admissions dataset.")
    parser.add_argument("question", nargs="?", help="Single question; omit for an interactive chat.")
    parser.add_argument("--dataset", type=Path, help="Canonical Parquet file (default: backend's ingested MIMIC-IV demo).")
    parser.add_argument("--model", help="Model name, or 'provider:model'. Default: $LLM_MODEL or claude-sonnet-5.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show which tools were called and with what arguments.")
    args = parser.parse_args(argv)

    base = Settings.from_env()
    settings = Settings(dataset_path=args.dataset or base.dataset_path, model=args.model or base.model, temperature=base.temperature)

    if _uses_anthropic(settings.model) and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export it, or choose another provider with --model 'provider:model'.", file=sys.stderr)
        return 2
    try:
        df = load_dataset(settings.dataset_path)
    except DatasetError as exc:
        print(exc, file=sys.stderr)
        return 2

    agent = build_agent(df, settings=settings)
    history = None
    questions = iter([args.question]) if args.question else iter(lambda: input("> ").strip(), "exit")
    for question in questions:
        if not question:
            continue
        answer, history = ask(agent, question, history)
        if args.verbose:
            for call in tool_calls_made(history):
                print(f"[tool] {call['name']}({json.dumps(call['args'], ensure_ascii=False)})", file=sys.stderr)
        print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
