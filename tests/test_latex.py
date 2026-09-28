import pytest

from monday.latex import LatexError, compile_tex, escape, find_pdflatex


def test_escape_special_characters():
    assert escape("R&D 100% $5 #1 a_b p < 0.01") == r"R\&D 100\% \$5 \#1 a\_b p \textless{} 0.01"


def _pdflatex():
    try:
        return find_pdflatex()
    except LatexError:
        return None


needs_latex = pytest.mark.skipif(_pdflatex() is None, reason="pdflatex not installed")


@needs_latex
def test_compile_reports_pages(tmp_path):
    tex = tmp_path / "ok.tex"
    tex.write_text(r"\documentclass{article}\begin{document}Hello\end{document}", encoding="utf-8")
    result = compile_tex(tex)
    assert result.pdf.exists() and result.pages == 1


@needs_latex
def test_compile_error_is_readable(tmp_path):
    tex = tmp_path / "bad.tex"
    tex.write_text(r"\documentclass{article}\begin{document}\nosuchcommand\end{document}", encoding="utf-8")
    with pytest.raises(LatexError, match="Undefined control sequence"):
        compile_tex(tex)
