# Paper build

`main.tex` is the single source: *Raw Physical Parameters Are the Wrong
Coordinates for Sim-to-Real Uncertainty.* It builds as an IEEE conference
paper (ICRA) by default and switches to ICLR with a preamble change.

## Build (ICRA / IEEE)

```
cd paper
pdflatex main && pdflatex main
```

Needs `IEEEtran.cls` (ships with TeX Live and MiKTeX). References are a
`thebibliography` block, so no bibliography tool is needed.

**Page count has never been checked** -- there is no LaTeX toolchain on the
machine this was written on. ICRA allows 8 pages including references.

## Before submitting

```
run.cmd tools/tex_lint.py            # environments, refs, cites, figure files
run.cmd tools/check_paper_numbers.py # every number re-read from results/*.json
run.cmd tools/dimensional_check.py   # the "nothing beyond Buckingham-Pi" claim
```

## What the paper claims, and what it deliberately does not

**Claims.** Raw-parameter distance misorders transfer risk for a learned
controller; the redundant directions are exactly dimensional similarity
(verified, not assumed); similarity coordinates correctly decomposed a real,
accidental CAD defect before it was measured.

**Does not claim.**
- *A new symmetry.* The solved group is Buckingham-Pi plus one parameter that
  never enters the equations. An earlier draft said otherwise; that was wrong.
- *Dimensionless control or similarity transfer.* Prior work owns both
  (Girard 2024; Pascoa, Lalonde & Girard 2025; Charvet, Stein & Murray-Smith
  2025; Kir Hromatko et al. 2025). The open question is using the similarity
  quotient as the space in which uncertainty is randomised and identified.
- *That random search proves there is no other null direction.* It is evidence.
- *Hardware results.* The machine is not assembled. Section IX is the protocol,
  registered in advance.
- *The randomisation result.* It was voided by a fall-through bug and is
  withheld until the 5-seed re-run completes (`results/T5_VOID_fallthrough/`).

## References

All thirteen were checked against the publications on 2026-09-12. The four
2025 dimensionless-control entries are arXiv preprints and should be updated
if they acquire a venue.

## Switching to ICLR

Single column, 9 pages, **double blind**. Replace the preamble and author block
with the ICLR style, redirect `\repo` (defined once at the top) to an
anonymised mirror, and change `\columnwidth` to `\textwidth` in figures.
