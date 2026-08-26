# -*- coding: utf-8 -*-
"""Convert manuscript_v1.md (IEEE TIM revision) to IEEEtran LaTeX source.

Produces a compilable IEEEtran .tex file (IEEEtrans.cls, double column,
US Letter, 10pt) targeting the IEEE TIM Information for Authors format:
  - title, author placeholder block, abstract (200-250 words), index terms
  - numbered sections I..VIII, subsections A.., formulas, tables, figures
  - IEEE bibliography style placeholder (user fills via BibTeX or manual)

Usage:
  python scripts/md2ieeetex.py
Output:
  docs/paper_draft_ieee_tim.tex
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "tim_submission_package" / "manuscript_v1.md"
OUT = ROOT / "docs" / "paper_draft_ieee_tim.tex"

PRE = r"""% =====================================================================
% IEEE TIM submission draft (auto-generated from manuscript_v1.md)
% Compile with: pdflatex (IEEEtran class, TeX Live / MiKTeX)
% =====================================================================
\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
\usepackage{cite}
\usepackage{multirow}
\hyphenation{op-tical net-works semi-conduc-tor}
\begin{document}

\title{Measurement-Cost-Aware Sampling and Hierarchical Weakly Supervised
Detection for Three-Phase Partial Discharge Monitoring}

\author{%~AUTHOR-NAME~,~\IEEEmembership{Student Member,~IEEE,}~
        %~AUTHOR-NAME~,~\IEEEmembership{Student Member,~IEEE,}~
        %~SUPERVISOR-NAME~,~\IEEEmembership{Member,~IEEE}% <- replace with real authors
\thanks{Manuscript received XX, 2026. This work was supported in part by ...}
\thanks{The authors are with ... (e-mail: ...).}}

\markboth{Journal of IEEE Transactions on Instrumentation and Measurement,~Vol.~XX, No.~XX, 2026}%
{Author \MakeLowercase{\textit{et al.}}: Measurement-Cost-Aware Sampling and Hierarchical Weakly Supervised Detection for Three-Phase Partial Discharge Monitoring}

\maketitle

\begin{abstract}
%~ABSTRACT~%
\end{abstract}

\begin{IEEEkeywords}
Partial discharge measurement; coverage-aware sampling; weakly supervised learning; multiple instance learning; three-phase power distribution; leakage-safe evaluation; measurement cost.
\end{IEEEkeywords}

"""

POST = r"""
\end{document}
"""


def esc_tex(s: str) -> str:
    """Escape special LaTeX characters outside math (do NOT escape '#')."""
    s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    return s


def md_to_tex(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []
    table_align = ""
    table_caption = ""
    started = False  # True once we pass the top matter (title/abstract)
    for line in lines:
        stripped = line.strip()

        # --- top matter skipping (until the first numbered section) ---
        if not started:
            if re.match(r"^## [IVX]+\.", stripped):
                started = True
            else:
                continue  # skip title, blockquote, abstract, index terms, ---

        # skip horizontal rules and the References section (bib in POST)
        if stripped == "---":
            continue
        if re.match(r"^## References", stripped):
            break
        if re.match(r"^\[?\d+\] ", stripped):
            continue

        # table caption
        m = re.match(r"\*\*TABLE ([IVX]+)[*_]*\s*(.*)", stripped)
        if m:
            table_caption = f"\\caption{{{esc_tex(m.group(2))}}}"
            continue

        # horizontal rule -> table end
        if re.match(r"^\|[\s\-|]+\|?$", stripped):
            continue
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not in_table:
                in_table = True
                table_rows = []
                # header row may appear as first data row; align default
            table_rows.append(cells)
            continue
        elif in_table:
            # flush table
            if table_rows:
                ncol = max(len(r) for r in table_rows)
                head = table_rows[0] if len(table_rows) > 1 else [""] * ncol
                body = table_rows[1:] if len(table_rows) > 1 else table_rows
                out.append("\\begin{table}[!t]")
                out.append("\\centering")
                out.append("\\caption{" + table_caption + "}")
                out.append("\\begin{tabular}{" + "c" * ncol + "}")
                out.append("\\toprule")
                out.append(" & ".join(esc_tex(c) for c in head) + r" \\")
                out.append("\\midrule")
                for row in body:
                    row = row + [""] * (ncol - len(row))
                    out.append(" & ".join(esc_tex(c) for c in row) + r" \\")
                out.append("\\bottomrule")
                out.append("\\end{tabular}")
                out.append("\\end{table}")
                out.append("")
            in_table = False
            table_rows = []
            table_caption = ""

        # section headers
        m = re.match(r"^## ([IVX]+)\.\s*(.*)", stripped)
        if m:
            out.append(f"\\section{{{esc_tex(m.group(2))}}}")
            continue
        m = re.match(r"^### ([A-Z])\.\s*(.*)", stripped)
        if m:
            out.append(f"\\subsection{{{esc_tex(m.group(2))}}}")
            continue
        m = re.match(r"^#### ([A-Z])\.\d+\s*(.*)", stripped)
        if m:
            out.append(f"\\subsubsection{{{esc_tex(m.group(2))}}}")
            continue

        # figure placeholder
        if stripped.startswith("![") and "](" in stripped:
            alt = stripped[2:].split("](")[0]
            out.append("\\begin{figure}[!t]")
            out.append("\\centering")
            out.append(f"\\includegraphics[width=0.9\\columnwidth]{{figures/{alt}}}")
            out.append(f"\\caption{{{esc_tex(alt)}}}")
            out.append("\\label{fig:" + re.sub(r"\W+", "_", alt) + "}")
            out.append("\\end{figure}")
            continue

        # math display
        if stripped.startswith("$$"):
            out.append("\\begin{equation}")
            continue
        if stripped.endswith("$$"):
            out.append("\\end{equation}")
            continue

        # equations like "$$z(v) = ... (1)$$" are handled by the two above
        # inline math $...$ passes through

        # bold paragraphs / list items
        if stripped.startswith("1. ") or stripped.startswith("2. ") or \
           stripped.startswith("3. ") or stripped.startswith("4. ") or \
           stripped.startswith("5. ") or stripped.startswith("6. ") or \
           stripped.startswith("7. "):
            out.append("\\noindent " + esc_tex(stripped) + r" \\")
            out.append("")
            continue

        # blank
        if not stripped:
            out.append("")
            continue

        # regular paragraph
        out.append(esc_tex(stripped))
        out.append("")

    # flush trailing table
    if in_table and table_rows:
        ncol = max(len(r) for r in table_rows)
        head = table_rows[0]
        body = table_rows[1:]
        out.append("\\begin{table}[!t]")
        out.append("\\centering")
        out.append("\\caption{" + table_caption + "}")
        out.append("\\begin{tabular}{" + "c" * ncol + "}")
        out.append("\\toprule")
        out.append(" & ".join(esc_tex(c) for c in head) + r" \\")
        out.append("\\midrule")
        for row in body:
            row = row + [""] * (ncol - len(row))
            out.append(" & ".join(esc_tex(c) for c in row) + r" \\")
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")

    return "\n".join(out)


def refs_to_bibitem(md: str) -> str:
    """Convert the markdown reference list into thebibliography bibitems."""
    m = re.search(r"## References\n\n(.*)$", md, re.S)
    if not m:
        return ""
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        mm = re.match(r"\[(\d+)\]\s*(.*)", line)
        if not mm:
            continue
        num, body = mm.group(1), mm.group(2)
        # IEEE-ish latex: quotes -> ``'', italicize journal names (*...*)
        body = body.replace('"', "''").replace('"', "``", 1)
        # handle "Title," *Journal*,
        body = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", body)
        body = esc_tex(body)
        out.append(f"\\bibitem{{ref{num}}} {body}")
    return "\n".join(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--title", default="Measurement-Cost-Aware Sampling and Hierarchical Weakly Supervised\nDetection for Three-Phase Partial Discharge Monitoring")
    ap.add_argument("--journal", default="Journal of IEEE Transactions on Instrumentation and Measurement")
    ap.add_argument("--running", default="Measurement-Cost-Aware Sampling and Hierarchical Weakly Supervised Detection for Three-Phase Partial Discharge Monitoring")
    args = ap.parse_args()

    md = Path(args.src).read_text(encoding="utf-8")
    # extract abstract
    am = re.search(r"## Abstract\n\n(.*?)(?=\n\n\*\*Index Terms)", md, re.S)
    abstract = am.group(1).replace("\n", " ") if am else "%~ABSTRACT~%"
    body = md_to_tex(md)
    bib = refs_to_bibitem(md)
    pre = PRE.replace("%~ABSTRACT~%", esc_tex(abstract))
    pre = pre.replace(
        "Measurement-Cost-Aware Sampling and Hierarchical Weakly Supervised\nDetection for Three-Phase Partial Discharge Monitoring",
        args.title.replace("\\n", "\n"))
    pre = pre.replace(
        "Journal of IEEE Transactions on Instrumentation and Measurement,~Vol.~XX, No.~XX, 2026",
        args.journal + ",~Vol.~XX, No.~XX, 2026")
    pre = pre.replace(
        "Measurement-Cost-Aware Sampling and Hierarchical Weakly Supervised Detection for Three-Phase Partial Discharge Monitoring",
        args.running)
    tex = pre
    tex += body
    tex += "\n\\begin{thebibliography}{00}\n" + bib + "\n\\end{thebibliography}\n"
    tex += POST
    Path(args.out).write_text(tex, encoding="utf-8")
    print(f"Wrote {args.out} ({len(tex)} chars)")


if __name__ == "__main__":
    main()
