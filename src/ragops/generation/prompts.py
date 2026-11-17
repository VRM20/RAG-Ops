"""Versioned prompt templates (config/prompts/*.yaml).

Prompts are code: they live in git, have a name + version in the filename, and any
change to them triggers the evaluation gate in CI. The content hash is recorded on
every eval run so a score can always be traced back to the exact prompt text.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import yaml
from pydantic import BaseModel

from ragops.config import PROMPTS_DIR


class PromptTemplate(BaseModel):
    name: str
    description: str = ""
    system: str
    user_template: str

    @property
    def sha(self) -> str:
        return hashlib.sha256((self.system + "\x00" + self.user_template).encode()).hexdigest()[:12]

    def render_user(self, **kwargs: str) -> str:
        return self.user_template.format(**kwargs)


@lru_cache
def load_prompt(name: str) -> PromptTemplate:
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt '{name}' not found at {path}")
    return PromptTemplate.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def list_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.yaml"))
