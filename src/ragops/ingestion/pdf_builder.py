"""Render the Markdown corpus sources into realistic PDFs (headings, tables, footers).

The PDFs feed the multimodal parsing path, so the demo corpus exercises the same
"PDF -> structured Markdown -> chunks" flow as real enterprise documents. Page footers
are added on purpose: a good parser must drop them.
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_LIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")


def _latin1(text: str) -> str:
    # Core PDF fonts are Latin-1 only; normalise the few typographic characters.
    replacements = {"–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "₹": "INR "}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


class _CorpusPDF(FPDF):
    def __init__(self, footer_text: str):
        super().__init__(format="A4")
        self.footer_text = footer_text
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"{self.footer_text} | Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def markdown_to_pdf(markdown: str, out_path: Path, footer_text: str = "Internal document") -> Path:
    pdf = _CorpusPDF(footer_text)
    pdf.add_page()
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            pdf.ln(2)
            i += 1
            continue

        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append([_latin1(c.replace("**", "")) for c in cells])
                i += 1
            pdf.set_font("Helvetica", "", 9.5)
            with pdf.table(text_align="LEFT", line_height=5.5, first_row_as_headings=True) as table:
                for row in rows:
                    r = table.row()
                    for cell in row:
                        r.cell(cell)
            pdf.ln(3)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            size = {1: 17, 2: 13.5, 3: 11.5}.get(level, 11)
            pdf.ln(3 if level > 1 else 0)
            pdf.set_font("Helvetica", "B", size)
            pdf.multi_cell(0, size * 0.55, _latin1(heading.group(2)), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        elif line.startswith(">"):
            pdf.set_font("Helvetica", "I", 9.5)
            pdf.set_text_color(90, 90, 90)
            pdf.multi_cell(0, 5, _latin1(line.lstrip("> ").strip()), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        elif _LIST_RE.match(line):
            pdf.set_font("Helvetica", "", 10.5)
            item = _LIST_RE.match(line).group(1)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 5.5, _latin1(f"- {item}"), markdown=True, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 10.5)
            pdf.multi_cell(0, 5.5, _latin1(line.strip()), markdown=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return out_path
