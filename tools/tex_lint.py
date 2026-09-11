"""Structural lint for paper/main.tex when no LaTeX toolchain is installed.

Catches the errors that would otherwise only surface at submission time on a
machine that can actually run pdflatex: unbalanced environments, dangling
\\ref, uncited \\bibitem, missing figure files. It does NOT check page count,
which is the one ICRA requirement that needs a real build.

  run.cmd tools/tex_lint.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "main.tex")


def main() -> int:
    s = open(TEX, encoding="utf-8").read()
    problems = 0

    envs: dict = {}
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
        envs.setdefault(m.group(2), []).append(m.group(1))
    bad = {k: v for k, v in envs.items() if v.count("begin") != v.count("end")}
    print("unbalanced environments :", bad if bad else "none")
    problems += len(bad)

    t = re.sub(r"\\[{}]", "", s)
    delta = t.count("{") - t.count("}")
    print("brace balance           :", delta)
    problems += int(delta != 0)

    labels = set(re.findall(r"\\label\{([^}]+)\}", s))
    # \eqref and \autoref are references too; counting only \ref reports
    # perfectly good labels as orphans
    refs = set(re.findall(r"\\(?:eq|auto|c)?ref\{([^}]+)\}", s))
    print("refs with no label      :", (refs - labels) or "none")
    print("labels never referenced :", (labels - refs) or "none")
    problems += len(refs - labels)

    groups = re.findall(r"\\cite\{([^}]+)\}", s)
    cites = {c.strip() for g in groups for c in g.split(",")}
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
    print("cites with no bibitem   :", (cites - bib) or "none")
    print("bibitems never cited    :", (bib - cites) or "none")
    problems += len(cites - bib)

    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s):
        p = os.path.join(ROOT, "paper", f)
        ok = os.path.exists(p)
        print("figure %-42s %s" % (f, "OK" if ok else "MISSING"))
        problems += int(not ok)

    body = re.sub(r"\\[a-zA-Z]+", " ", s)
    body = re.sub(r"[{}$&\\%]", " ", body)
    words = len(body.split())
    print("approx words            : %d  (~%.1f IEEE two-column pages of text)"
          % (words, words / 900.0))

    print("")
    print("PAGE COUNT IS NOT CHECKED HERE. ICRA allows 8 pages including")
    print("references; build with pdflatex before submitting.")
    print("problems found: %d" % problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
