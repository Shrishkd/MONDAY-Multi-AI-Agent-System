"""Compile LaTeX to PDF with pdflatex (MiKTeX or TeX Live)."""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_LOCATIONS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/MiKTeX/miktex/bin/x64/pdflatex.exe",
    Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "MiKTeX/miktex/bin/x64/pdflatex.exe",
]

_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}",
    # Unicode that models like to emit and pdflatex can't typeset.
    "‐": "-", "‑": "-", "‒": "-", "−": "-", "–": "--", "—": "---",
    " ": " ", " ": " ", " ": " ", "​": "",
    "‘": "'", "’": "'", "“": "``", "”": "''", "…": "...",
    "→": r"$\rightarrow$", "×": r"$\times$", "≤": r"$\leq$", "≥": r"$\geq$",
    "≈": r"$\approx$",
}


class LatexError(RuntimeError):
    pass


@dataclass
class CompileResult:
    pdf: Path
    pages: int


def escape(text: str) -> str:
    # Latin-1 is fine under T1 + utf8; anything else unmapped (emoji etc.) would break the build.
    return "".join(_SPECIAL.get(ch, ch if ord(ch) < 0x250 else "") for ch in text)


def find_pdflatex(configured: str | None = None) -> str:
    if configured:
        if Path(configured).is_file():
            return configured
        raise LatexError(f"pdflatex not found at configured path: {configured}")
    found = shutil.which("pdflatex", path=_clean_path())
    if found:
        return found
    for candidate in _DEFAULT_LOCATIONS:
        if candidate.is_file():
            return str(candidate)
    raise LatexError("pdflatex not found. Install MiKTeX or set `pdflatex:` in config.yaml.")


def _clean_path() -> str:
    # MiKTeX aborts if any PATH entry is not a directory (e.g. "...\nodejs\node.exe").
    return os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if os.path.isdir(p))


def compile_tex(tex_file: Path, pdflatex: str | None = None, timeout: int = 180) -> CompileResult:
    exe = find_pdflatex(pdflatex)
    env = dict(os.environ, PATH=_clean_path())
    try:
        proc = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
            cwd=tex_file.parent, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise LatexError(f"pdflatex timed out after {timeout}s") from exc

    written = re.search(r"Output written on .*?\((\d+) pages?", proc.stdout)
    if proc.returncode != 0 or not written:
        lines = proc.stdout.splitlines()
        errors = [ln for i, ln in enumerate(lines) if ln.startswith("!") or ln.startswith("l.")]
        detail = "\n".join(errors[:6]) or "\n".join(lines[-8:])
        raise LatexError(f"pdflatex failed on {tex_file.name}:\n{detail}")
    return CompileResult(pdf=tex_file.with_suffix(".pdf"), pages=int(written.group(1)))
