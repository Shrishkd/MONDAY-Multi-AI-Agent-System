"""Render a tailored resume into the user's own LaTeX template.

The template is split on \\section{...}. Summary, Experience, Projects and Skills are
regenerated with the template's own macros; every other section (header, Education,
Certifications, ...) is copied through untouched.
"""

import re
from dataclasses import dataclass, field

from monday.experience import Project, Role
from monday.latex import escape

TAILORED_SECTIONS = ("Summary", "Experience", "Projects", "Skills")

LINK_ICONS = {"github": r"\faGithub", "report": r"\faFile*", "live demo": r"\faExternalLink*",
              "demo": r"\faExternalLink*", "website": r"\faExternalLink*", "paper": r"\faFile*"}

# Anchored to line starts so commented-out lines ("% \section{...}") don't count.
_SECTION = re.compile(r"^[ \t]*\\section\{([^}]*)\}", re.M)
_END = re.compile(r"^[ \t]*\\end\{document\}", re.M)


@dataclass
class Template:
    head: str                                   # preamble + header, up to the first \section
    sections: list[tuple[str, str]]             # (title, raw text including the \section line)
    tail: str                                   # \end{document} and after

    @classmethod
    def parse(cls, tex: str) -> "Template":
        end = _END.search(tex)
        if not end:
            raise ValueError("Template has no \\end{document}")
        body, tail = tex[:end.start()], tex[end.start():]
        starts = list(_SECTION.finditer(body))
        if not starts:
            raise ValueError("Template has no \\section{...}")
        sections = []
        for i, m in enumerate(starts):
            stop = starts[i + 1].start() if i + 1 < len(starts) else len(body)
            sections.append((m.group(1).strip(), body[m.start():stop]))
        return cls(head=body[:starts[0].start()], sections=sections, tail=tail)

    def render(self, replacements: dict[str, str]) -> str:
        """Replace the bodies of the named sections; keep the rest byte-for-byte."""
        parts = [self.head]
        for title, raw in self.sections:
            parts.append(f"\\section{{{title}}}\n\n{replacements[title]}\n\n" if title in replacements else raw)
        return "".join(parts) + self.tail


def inline(text: str) -> str:
    """Escape for LaTeX, then turn **bold** into \\textbf{...}."""
    return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", escape(text))


def _dashes(text: str) -> str:
    return escape(text).replace(" - ", " -- ")


@dataclass
class RenderedItem:
    """A role or project with the bullets chosen for this application."""
    source: Role | Project
    bullets: list[str] = field(default_factory=list)   # plain text with optional **bold**


def summary_tex(summary: str) -> str:
    return f"\\small\n{inline(summary)}"


def _items(bullets: list[str]) -> str:
    lines = "\n".join(f"    \\resumeItem{{{inline(b)}}}\n" for b in bullets)
    return f"\\resumeItemListStart\n\n{lines}\n\\resumeItemListEnd"


def experience_tex(items: list[RenderedItem]) -> str:
    blocks = []
    for item in items:
        role: Role = item.source
        blocks.append(
            f"\\resumeSubheading\n    {{{escape(role.company)}}}\n    {{{escape(role.location or '')}}}\n"
            f"    {{{escape(role.title)}}}\n    {{{escape(role.start)} -- {escape(role.end)}}}\n\n"
            + _items(item.bullets)
        )
    return "\\resumeSubHeadingListStart\n\n" + "\n\n".join(blocks) + "\n\n\\resumeSubHeadingListEnd"


def _project_heading(p: Project) -> str:
    links = ""
    for label, url in p.links.items():
        icon = LINK_ICONS.get(label.lower(), r"\faLink")
        links += f"\n    \\textbar\\ {icon}\\ \\href{{{url}}}{{{escape(label)}}}"
    return f"\\resumeProjectHeading\n{{\n    {_dashes(p.name)}{links}\n}}\n{{{_dashes(p.dates or '')}}}"


def projects_tex(items: list[RenderedItem]) -> str:
    blocks = []
    for i, item in enumerate(items):
        p: Project = item.source
        block = _project_heading(p) + "\n\n" + _items(item.bullets)
        if p.tech_stack:
            block += ("\n\n\\vspace{-3pt}\n\\noindent\\small\n\\textbf{Tech Stack:}\n"
                      f"\\textit{{\n{escape(', '.join(p.tech_stack))}\n}}")
            if i < len(items) - 1:
                block += "\n\\vspace{5pt}"
        blocks.append(block)
    return "\\resumeProjectListStart\n\n" + "\n\n".join(blocks) + "\n\n\\resumeProjectListEnd"


def skills_tex(skills: dict[str, list[str]]) -> str:
    rows = "\n\n".join(
        f"\\resumeItem{{\n    \\textbf{{{escape(category)}:}}\n    {escape(', '.join(names))}\n}}"
        for category, names in skills.items()
    )
    return f"\\resumeSubHeadingListStart\n\n{rows}\n\n\\resumeSubHeadingListEnd"
