"""Import a new resume version: LaTeX template + a proposed Experience Bank.

Nothing is applied until you review it. The model's extraction is checked against the
uploaded text (every number in a fact must appear in the resume), and the current
template and bank are backed up before anything is replaced.
"""

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf
import yaml

from monday.agents.resume_tailor import numbers
from monday.config import Config
from monday.experience import ExperienceBank
from monday.llm import LLM
from monday.resume_tex import TAILORED_SECTIONS, Template
from monday.skills import normalize

AGENT = "resume_import"

# Macros the Resume Tailor renders with (Jake's-resume style templates).
REQUIRED_MACROS = ("resumeSubheading", "resumeItem", "resumeProjectHeading", "resumeSubHeadingListStart",
                   "resumeSubHeadingListEnd", "resumeItemListStart", "resumeItemListEnd",
                   "resumeProjectListStart", "resumeProjectListEnd")

SYSTEM = """You convert a resume into a structured Experience Bank.
Rules:
- Copy wording faithfully. Do not improve, summarise away or invent anything.
- One fact per resume bullet. Keep the bullet's text; turn \\textbf{x} into **x**; drop other LaTeX markup.
- Fact ids: short, unique, lowercase-with-hyphens, prefixed by the role/project (e.g. "acme-latency").
- metrics: the real numbers from that bullet, as short phrases. skills: technologies the bullet names.
- Projects: name, dates, links as {label: url} (e.g. {"GitHub": "..."}), tech_stack from its "Tech Stack" line.
- skills: the resume's skill categories exactly as written, category -> list of skills.
- Keep the resume's summary as profile.summary and its headline/title as profile.headline.
- Put every certification and achievement in; leave stories empty."""


@dataclass
class TemplateCheck:
    tex: str                                   # possibly fixed LaTeX
    fixes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def check_template(tex: str) -> TemplateCheck:
    """Apply known fixes and report what would stop the Tailor from using this template."""
    check = TemplateCheck(tex=tex)
    # fontawesome5 has no \faFileAlt-style commands; "-alt" icons are the starred form.
    for name in sorted(set(re.findall(r"\\fa([A-Z][A-Za-z]*?)Alt(?![A-Za-z])", tex))):
        check.tex = re.sub(rf"\\fa{name}Alt(?![A-Za-z])", rf"\\fa{name}*", check.tex)
        check.fixes.append(rf"\fa{name}Alt -> \fa{name}* (fontawesome5 has no \fa{name}Alt)")

    missing = [m for m in REQUIRED_MACROS
               if not re.search(rf"\\(?:re)?newcommand\{{?\\{m}(?![A-Za-z])", check.tex)]
    if missing:
        check.problems.append("Template doesn't define macros the Tailor uses: " + ", ".join(missing))
    try:
        titles = [t for t, _ in Template.parse(check.tex).sections]
        absent = [s for s in TAILORED_SECTIONS if s not in titles]
        if absent:
            check.problems.append(f"No \\section{{...}} named: {', '.join(absent)} - those parts can't be tailored")
    except ValueError as exc:
        check.problems.append(str(exc))
    return check


def pdf_text(data: bytes) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _unique_ids(raw: dict) -> dict:
    seen: set[str] = set()
    for group in ("roles", "projects"):
        for item in raw.get(group, []):
            for fact in item.get("facts", []):
                base, n = fact.get("id") or "fact", 2
                while fact.get("id") in seen or not fact.get("id"):
                    fact["id"] = f"{base}-{n}"
                    n += 1
                seen.add(fact["id"])
    return raw


def grounding_warnings(bank: ExperienceBank, source: str) -> list[str]:
    """Numbers the model put in the bank that the uploaded resume doesn't contain."""
    known = numbers(source)
    warnings = []
    for fact in bank.facts():
        extra = numbers(f"{fact.text} {' '.join(fact.metrics)}") - known
        if extra:
            warnings.append(f"{fact.id}: numbers not in your resume: {sorted(extra)}")
    extra = numbers(bank.profile.summary or "") - known
    if extra:
        warnings.append(f"summary: numbers not in your resume: {sorted(extra)}")
    if bank.profile.email and bank.profile.email.lower() not in source.lower():
        warnings.append(f"email {bank.profile.email} is not in your resume")
    return warnings


def keep_stories(new: ExperienceBank, old: ExperienceBank | None) -> ExperienceBank:
    """Interview stories are written by hand; keep every one whose facts still exist."""
    if not old:
        return new
    ids = set(new.fact_index())
    kept = [s for s in old.stories if set(s.fact_ids) <= ids]
    return new.model_copy(update={"stories": kept})


REWORD_SIMILARITY = 0.75


def pair_facts(old: ExperienceBank | None, new: ExperienceBank) -> tuple[dict[str, str], dict[str, str]]:
    """Match new facts to old ones by content, ignoring ids and markup.

    Returns (unchanged, reworded) as {new_id: old_id}.
    """
    if not old:
        return {}, {}
    old_facts, new_facts = old.fact_index(), new.fact_index()
    old_by_norm = {normalize(f.text): i for i, f in old_facts.items()}
    unchanged = {n: old_by_norm[normalize(f.text)] for n, f in new_facts.items() if normalize(f.text) in old_by_norm}
    free_old = [i for i in old_facts if i not in unchanged.values()]
    reworded = {}
    for n, f in new_facts.items():
        if n in unchanged or not free_old:
            continue
        best = max(free_old, key=lambda o: SequenceMatcher(None, normalize(f.text), normalize(old_facts[o].text)).ratio())
        if SequenceMatcher(None, normalize(f.text), normalize(old_facts[best].text)).ratio() >= REWORD_SIMILARITY:
            reworded[n] = best
            free_old.remove(best)
    return unchanged, reworded


def keep_ids(new: ExperienceBank, old: ExperienceBank | None) -> ExperienceBank:
    """Give re-imported facts their old ids, so stories and past drafts keep pointing at them."""
    unchanged, reworded = pair_facts(old, new)
    mapping = {**unchanged, **reworded}
    if not mapping:
        return new
    raw = new.model_dump()
    taken = set(mapping.values())
    for group in ("roles", "projects"):
        for item in raw[group]:
            for fact in item["facts"]:
                if fact["id"] in mapping:
                    fact["id"] = mapping[fact["id"]]
                elif fact["id"] in taken:          # a new fact that happens to reuse an old id
                    fact["id"] = f"{fact['id']}-new"
    return ExperienceBank.model_validate(_unique_ids(raw))


def _en_dashes(raw: dict) -> dict:
    """LaTeX '--' in names and dates -> a real en dash (rendered back to '--')."""
    for p in raw.get("projects", []):
        for key in ("name", "dates"):
            if p.get(key):
                p[key] = p[key].replace("--", "–")
    for r in raw.get("roles", []):
        for key in ("start", "end"):
            if r.get(key):
                r[key] = str(r[key]).replace("--", "–")
    return raw


def extract_bank(llm: LLM, source: str, old: ExperienceBank | None) -> tuple[ExperienceBank, list[str], str]:
    """Returns (proposed bank, warnings, model)."""
    result = llm.structured(AGENT, SYSTEM, f"RESUME:\n\n{source}", ExperienceBank, temperature=0.1)
    bank = ExperienceBank.model_validate(_unique_ids(_en_dashes(result.output.model_dump())))
    bank = keep_stories(keep_ids(bank, old), old)
    return bank, grounding_warnings(bank, source), result.model


def diff(old: ExperienceBank | None, new: ExperienceBank) -> dict[str, list[str]]:
    old_facts, new_facts = (old.fact_index() if old else {}), new.fact_index()
    unchanged, reworded = pair_facts(old, new)
    matched_old = set(unchanged.values()) | set(reworded.values())
    return {
        "added": [f"{i}: {f.text}" for i, f in new_facts.items() if i not in unchanged and i not in reworded],
        "removed": [f"{i}: {f.text}" for i, f in old_facts.items() if i not in matched_old],
        "reworded": [f"{old_id}: {old_facts[old_id].text}  ->  {new_facts[n].text}" for n, old_id in reworded.items()],
    }


def bank_yaml(bank: ExperienceBank) -> str:
    return yaml.safe_dump(bank.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True, width=100)


def apply(config: Config, bank: ExperienceBank | None, template_tex: str | None,
          original_upload: tuple[str, bytes] | None = None) -> Path:
    """Back up the current bank/template, then write the new ones. Returns the backup folder."""
    backup = config.resume_template.parent / "backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup.mkdir(parents=True, exist_ok=True)
    for path in (config.experience_bank, config.resume_template):
        if path.exists():
            shutil.copy2(path, backup / path.name)
    if original_upload:
        name, data = original_upload
        (config.resume_template.parent / name).write_bytes(data)
    if template_tex is not None:
        config.resume_template.write_text(template_tex, encoding="utf-8")
    if bank is not None:
        header = "# MONDAY Experience Bank - imported from your uploaded resume. Only TRUE facts.\n"
        config.experience_bank.write_text(header + bank_yaml(bank), encoding="utf-8")
    return backup
