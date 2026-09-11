# Pre-registration — H9: does the geometry beat fidelity as a predictor?

**Committed before `results/h9_score.json` exists.**

## The claim

Every fidelity measure tested so far scores a simulator by how far its
*trajectories* diverge. The geometry says that is the wrong object: what
matters is where the parameter error points relative to the closed loop's
sensitivity structure. If that is right, then ranking conditions by

$$d_G(\xi) \;=\; \theta(\xi)^\top G\, \theta(\xi)$$

— the parameter displacement measured in the metric $G$ estimated by
`dynamics/geometry.py` — should predict transfer better than trajectory RMSE,
which is the incumbent and the strongest baseline this project has found.

## Material

The 38-condition suite in `results/h2_test.json`, measured months before the
geometry existed, each with success under the frozen policy.

**23 of the 38 are expressible** as a displacement $\theta$ in the
12-parameter space (`m1..m3, I1..I3, cart_mass, gravity, tau, kv, b_joint,
fc_joint`). The other 15 are not, and the reason matters more than the count:

| blocked by | conditions |
|---|---|
| second-order actuator (`order`, `omega_n`, `zeta`) | 5 |
| transport delay (`delay_s`) | 5 |
| deadband | 2 |
| force clamp (`f_clamp`) | 2 |

These are **model-form** changes, not parameter changes. A metric tensor over
a parameter vector is blind to them by construction. That is a structural
limitation of the whole approach and it is stated here, in advance, rather
than discovered later: the geometry cannot even be evaluated on the class of
error that produced this project's sharpest negative result, the fidelity
twins, whose leading pair is a second-order actuator against joint damping.

## Predictions

**P1 (primary).** Over the 23 expressible conditions,

$$|\rho(d_G,\ \text{success})| \;\ge\; |\rho(\text{traj\_rmse},\ \text{success})| + 0.10$$

with both correlations computed **on the same 23 conditions**. The published
$-0.506$ for trajectory RMSE was over all 38 and must not be used as the
benchmark here; recomputing it on the subset is part of the test.

**P2.** $\rho(d_G, \text{success}) < 0$, i.e. larger geometry-weighted
displacement means lower success. A positive correlation would mean the metric
is inverted and P1 would be meaningless even if the magnitude cleared.

**P3.** $d_G$ also beats raw parameter distance $\|\theta\|$ by $\ge 0.10$ in
$|\rho|$ on the same subset. Otherwise the weighting by $G$ is doing nothing
and the result is about using a parameter vector at all, not about the
geometry.

## Kill conditions

* **P1 fails** if the margin over trajectory RMSE is under 0.10. Then
  conventional trajectory fidelity remains the best available predictor and
  the geometry is a description, not a tool.
* **P3 fails** if $G$ adds nothing over an unweighted norm.
* With $n = 23$ a permutation $p$ is reported for every correlation, and no
  claim is made on a correlation with $p \ge 0.05$.

## Known weaknesses, stated in advance

$G$ was estimated on `dynamics/closed_loop.py`, the CPU reproduction, which
`RESULTS.md` records as discovery-grade. **H6 found it unstable across
policies**: the exact uniform-inertial symmetry was recovered at alignment
0.936, 0.539 and 0.138 on three seeds, which invalidated H6's own primary
prediction. The same instability may well sink H9. It is being run anyway
because $G$ and the suite share the *same* frozen policy, which is the one
condition under which H6's objection does not directly apply.

$\theta$ is a linearisation around nominal and several suite conditions are
far from nominal (`neg_mass_250` is $\times 2.5$). A quadratic form fitted
near the origin is not expected to hold there, and if H9 fails, that is the
first thing to check before the geometry is blamed.
