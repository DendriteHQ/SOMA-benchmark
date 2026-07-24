from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "available_prompts"


def list_prompt_ids() -> list[str]:
    return sorted(path.stem for path in PROMPTS_DIR.glob("*.txt") if path.is_file())


def load_prompt_template(prompt_id: str) -> str:
    normalized = prompt_id.strip()
    if not normalized:
        raise ValueError("prompt_id is required")
    path = PROMPTS_DIR / f"{normalized}.txt"
    if not path.is_file():
        raise ValueError(f"unknown prompt_id {prompt_id!r}")
    return path.read_text(encoding="utf-8")
