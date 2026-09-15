"""Check paper/main.tex against paper/numbers.tex: undefined or missing ('??') macros, citations, figures.

  run.cmd tools/paper_check.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tex = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
nums = open(os.path.join(ROOT, "paper", "numbers.tex"), encoding="utf-8").read()

defs = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", nums))
std = set("""documentclass IEEEoverridecommandlockouts usepackage input newcommand repo begin end title author
IEEEauthorblockN IEEEauthorblockA textit maketitle emph cite centering includegraphics textwidth caption label
section subsection ref textbf item alpha epsilon approx times Pr big theta in quad le leq footnotesize setlength
tabcolsep toprule midrule bottomrule multicolumn texttt bibitem url delta pi quad left right log sim mathrm
IEEEkeywords columnwidth""".split())
used = set(re.findall(r"\\([A-Za-z]+)", tex))
undefined = sorted(u for u in used if u not in defs and u not in std)
missing = sorted(u for u in used if defs.get(u, "").strip() == "??")
cites = {k.strip() for c in re.findall(r"\\cite\{([^}]*)\}", tex) for k in c.split(",")}
bib = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", tex)
nofig = [f for f in figs if not os.path.exists(os.path.join(ROOT, "paper", f)) and not os.path.exists(os.path.join(ROOT, f))]
words = len(re.sub(r"\\[A-Za-z]+|[{}$\\]", " ", tex.split("\\begin{document}")[1]).split())
print("undefined (non-standard) commands:", undefined)
print("macros still '??':", missing)
print("cited without bibitem:", sorted(cites - bib), "| bibitems never cited:", sorted(bib - cites))
print("missing figure files:", nofig)
print("approx words in body:", words)
sys.exit(1 if (missing or cites - bib or nofig) else 0)
