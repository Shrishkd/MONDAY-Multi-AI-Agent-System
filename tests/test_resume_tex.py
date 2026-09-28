import pytest

from monday.experience import Project
from monday.resume_tex import RenderedItem, Template, inline, projects_tex

TEX = r"""\documentclass{article}
\begin{document}
HEADER
\section{Summary}
old summary
\section{Education}
\textbf{VIT} -- keep me exactly
\section{Skills}
old skills
\end{document}
"""


def test_render_replaces_only_named_sections():
    tpl = Template.parse(TEX)
    assert [t for t, _ in tpl.sections] == ["Summary", "Education", "Skills"]
    out = tpl.render({"Summary": "new summary", "Skills": "new skills"})
    assert "HEADER" in out and "new summary" in out and "new skills" in out
    assert "old summary" not in out and "old skills" not in out
    assert "\\section{Education}\n\\textbf{VIT} -- keep me exactly\n" in out
    assert out.endswith("\\end{document}\n")


def test_render_without_changes_is_identity():
    assert Template.parse(TEX).render({}) == TEX


def test_template_needs_sections():
    with pytest.raises(ValueError):
        Template.parse(r"\begin{document}hi\end{document}")


def test_inline_escapes_then_bolds():
    assert inline("Cut **p95 latency** by 50% & more") == r"Cut \textbf{p95 latency} by 50\% \& more"


def test_inline_normalises_model_unicode():
    assert inline("Qwen3\u20114B \u2013 30\u201390\u202fs") == "Qwen3-4B -- 30--90 s"


def test_project_heading_has_icons_and_links():
    p = Project(name="Bot - v2", dates="Sep 2026", links={"Report": "https://r", "GitHub": "https://g"},
                tech_stack=["Python"], facts=[])
    tex = projects_tex([RenderedItem(p, ["Did a thing"])])
    assert r"\faFile*\ \href{https://r}{Report}" in tex
    assert r"\faGithub\ \href{https://g}{GitHub}" in tex
    assert "Bot -- v2" in tex and r"\resumeItem{Did a thing}" in tex
