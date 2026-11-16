"""Checks on the manuscript's Quarto source that need no TeX installation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "index.qmd"
DISCLOSURES = ROOT / "paper" / "sections" / "disclosures.tex"


def markdown_lines(text: str) -> Iterator[tuple[int, str]]:
    """Numbered lines of a Quarto document that Pandoc reads as Markdown.

    Leaves out the YAML header and fenced blocks (raw LaTeX among them),
    where a percent sign keeps the meaning its own language gives it.
    """
    in_header = False
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if number == 1 and line.strip() == "---":
            in_header = True
            continue
        if in_header:
            in_header = line.strip() != "---"
            continue
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield number, line


def latex_comment_lines(text: str) -> list[tuple[int, str]]:
    return [
        (number, line)
        for number, line in markdown_lines(text)
        if line.lstrip().startswith("%")
    ]


def test_latex_comment_lines_are_found_only_where_pandoc_prints_them():
    text = "\n".join(
        [
            "---",
            "title: x",
            "% a percent sign inside the header is YAML's business",
            "---",
            "",
            "<!-- an HTML comment is dropped -->",
            "% printed in the PDF",
            "```{=latex}",
            "% a real comment inside raw LaTeX",
            "```",
            "  % also printed",
            r"\input{sections/abstract}",
        ]
    )
    assert latex_comment_lines(text) == [
        (7, "% printed in the PDF"),
        (11, "  % also printed"),
    ]


def test_manuscript_source_has_no_latex_comment_lines():
    # Pandoc reads index.qmd as Markdown, so a line that starts with "%" is
    # text: it is escaped and printed in the PDF. Comments in the Quarto
    # source are HTML comments; LaTeX comments belong in the .tex sections.
    assert latex_comment_lines(MANUSCRIPT.read_text()) == []


def test_availability_statement_quotes_the_registered_input_hashes():
    # The statement gives each input file's SHA-256 by its first eight
    # characters. Each prefix must belong to a hash the loaders register, so
    # the paper cannot name a file the repository does not record.
    import re

    from imputation_paper.data.cps import ASEC_BUNDLE_SHA256
    from imputation_paper.data.scf import KNOWN_EXTRACTS

    registered = [*ASEC_BUNDLE_SHA256.values(), *KNOWN_EXTRACTS]
    quoted = re.findall(r"\\texttt\{([0-9a-f]{8})\}", DISCLOSURES.read_text())
    assert len(quoted) == 4, quoted
    for prefix in quoted:
        assert any(digest.startswith(prefix) for digest in registered), prefix
