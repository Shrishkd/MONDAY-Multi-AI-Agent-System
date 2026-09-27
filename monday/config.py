import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    ollama_host: str = "http://localhost:11434"
    agents: dict[str, list[str]] = field(default_factory=dict)
    experience_bank: Path = PROJECT_ROOT / "data" / "experience_bank.yaml"
    database: Path = PROJECT_ROOT / "data" / "monday.db"

    def models_for(self, agent: str) -> list[str]:
        models = self.agents.get(agent)
        if not models:
            raise KeyError(f"No models configured for agent '{agent}' in config.yaml")
        return models


def _resolve(path: str | None, default: Path) -> Path:
    if not path:
        return default
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path or os.environ.get("MONDAY_CONFIG") or DEFAULT_CONFIG)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paths = raw.get("paths", {})
    return Config(
        ollama_host=raw.get("ollama_host", Config.ollama_host),
        agents={name: list(models) for name, models in (raw.get("agents") or {}).items()},
        experience_bank=_resolve(paths.get("experience_bank"), Config.experience_bank),
        database=_resolve(paths.get("database"), Config.database),
    )
