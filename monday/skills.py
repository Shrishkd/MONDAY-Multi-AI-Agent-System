"""Deterministic skill normalization and matching.

Kept out of the LLM on purpose: a score you can't reproduce or explain is useless for
deciding where to spend your time.
"""

import re

# Whole-token aliases applied after normalization. Both sides of a comparison go through
# the same mapping, so "Postgres" and "PostgreSQL" meet in the middle.
ALIASES = {
    "postgres": "postgresql",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "gcp": "google cloud platform",
    "golang": "go",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "llm": "large language models",
    "llms": "large language models",
    "genai": "generative ai",
    "restful": "rest",
    "apis": "api",
    "cicd": "ci cd",
    "tf": "tensorflow",
    "sklearn": "scikit learn",
    "scikitlearn": "scikit learn",
    "reactjs": "react",
    "nodejs": "node",
    "expressjs": "express",
    "nextjs": "next",
    "vuejs": "vue",
}


# Words JDs wrap around a skill that carry no meaning for matching:
# "fine-tuning models" should match "Fine-Tuning (LoRA/QLoRA)".
FILLER = {"models", "model", "experience", "skills", "skill", "knowledge", "familiarity",
          "understanding", "hands", "on", "strong", "solid", "proficiency", "in", "with",
          "working", "basic", "good", "of", "the", "tools", "frameworks", "framework"}


def normalize(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\.js\b", "js", text)                # node.js -> nodejs (then aliased)
    text = re.sub(r"[^a-z0-9+#]+", " ", text)           # keep c++, c#
    tokens = [ALIASES.get(t, t) for t in text.split()]
    return " ".join(" ".join(tokens).split())


def _contains(haystack: str, needle: str) -> bool:
    """Whole-word containment on normalized text: 'rag' matches 'rag hybrid retrieval', 'go' doesn't match 'google'."""
    if not needle:
        return False
    return re.search(rf"(?<![a-z0-9+#]){re.escape(needle)}(?![a-z0-9+#])", haystack) is not None


def _core(normalized: str) -> str:
    """Drop filler words, unless that would leave nothing."""
    words = [w for w in normalized.split() if w not in FILLER]
    return " ".join(words) if words else normalized


def skill_matches(required: str, candidate: str) -> bool:
    r, c = _core(normalize(required)), _core(normalize(candidate))
    return r == c or _contains(c, r) or (len(c) >= 3 and _contains(r, c))


def appears_in(skill: str, text: str) -> bool:
    """Is this skill actually mentioned in the text? Used to catch skills a model invented."""
    s, t = normalize(skill), normalize(text)
    if _contains(t, s):
        return True
    words = [w for w in s.split() if w not in {"and", "or", "of", "the", "with"}]
    return bool(words) and all(_contains(t, w) for w in words)
