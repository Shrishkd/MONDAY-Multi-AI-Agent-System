"""Turn a TailorResult into a compiled, checked resume on disk."""

import re
from dataclasses import dataclass
from pathlib import Path

from monday.agents.job_matcher import MatchReport
from monday.agents.resume_tailor import TailorResult
from monday.experience import ExperienceBank, Project, Role
from monday.latex import CompileResult, compile_tex
from monday.resume_tex import (RenderedItem, Template, experience_tex, projects_tex,
                               skills_tex, summary_tex)
from monday.skills import appears_in


@dataclass
class Coverage:
    found: list[str]
    missing: list[str]

    @property
    def percent(self) -> float:
        total = len(self.found) + len(self.missing)
        return 100 * len(self.found) / total if total else 0.0


def keyword_coverage(report: MatchReport, text: str) -> Coverage:
    """Which of the JD's requirements a keyword scan of the resume would find."""
    wanted = [m.skill for m in report.must_have + report.nice_to_have]
    found = [k for k in wanted if appears_in(k, text)]
    return Coverage(found=found, missing=[k for k in wanted if k not in found])


def _plain(summary: str, items: list[tuple[Role | Project, list[str]]], skills: dict[str, list[str]]) -> str:
    parts = [summary]
    for source, bullets in items:
        parts += bullets + list(getattr(source, "tech_stack", []))
    parts += [s for names in skills.values() for s in names]
    return "\n".join(parts).replace("**", "")


def baseline_text(bank: ExperienceBank) -> str:
    items = [(s, [f.text for f in s.facts]) for s in [*bank.roles, *bank.projects]]
    return _plain(bank.profile.summary or "", items, bank.skills)


def tailored_text(result: TailorResult) -> str:
    items = [(s.source, [b.text for b in bullets]) for s, bullets in result.sections]
    return _plain(result.summary, items, result.skills)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


@dataclass
class BuiltResume:
    tex: Path
    pdf: Path | None
    pages: int | None
    compile_error: str | None


def build(result: TailorResult, bank: ExperienceBank, template: Path, out_dir: Path,
          company: str, pdflatex: str | None = None) -> BuiltResume:
    tpl = Template.parse(template.read_text(encoding="utf-8"))
    roles = [RenderedItem(s.source, [b.text for b in bl]) for s, bl in result.sections if isinstance(s.source, Role)]
    projects = [RenderedItem(s.source, [b.text for b in bl]) for s, bl in result.sections if isinstance(s.source, Project)]
    tex = tpl.render({
        "Summary": summary_tex(result.summary),
        "Experience": experience_tex(roles),
        "Projects": projects_tex(projects),
        "Skills": skills_tex(result.skills),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{_slug(bank.profile.name)}_{_slug(company)}_Resume.tex"
    tex_path.write_text(tex, encoding="utf-8")
    try:
        compiled: CompileResult = compile_tex(tex_path, pdflatex)
        return BuiltResume(tex_path, compiled.pdf, compiled.pages, None)
    except Exception as exc:  # the .tex is still useful (e.g. for Overleaf) if the build fails
        return BuiltResume(tex_path, None, None, str(exc))
