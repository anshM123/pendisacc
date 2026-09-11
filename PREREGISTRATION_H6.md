# Pre-registration — H6: is the reality-gap geometry a property of the *policy*?

**Committed before `results/geometry_policy.json` exists.**

## Why this is the load-bearing experiment now

H4 tested whether the anisotropy belongs to the action interface. It does not
(`results/h4_score.json`: gap $+4.3$ against a threshold of $40$). That was the
paper's bridge from "one pendulum" to a general principle, and it broke.

H6 tests a different and stronger claim. If the geometry is a property of the
*plant*, then simulator adequacy is a fact about the robot and could in
principle be derived once, offline, from the mechanism. If it is a property of
the *learned policy*, then it cannot — two controllers that are
indistinguishable in the simulator can face different reality gaps, and the
geometry has to be measured per policy.

`results/geometry_discover.json` already points this way: the most-null
direction found was `cart_mass` $\leftrightarrow$ `kv`, which couples a plant
parameter to a controller gain. No analysis of the mechanism alone can produce
that direction. H6 asks whether the coupling goes further, to the learned
weights themselves.

## Material

The three nominal velocity policies `C5_S_nominal_s1..s3`. They were trained
before this hypothesis existed, under an identical simulator, reward,
architecture, iteration count and protocol. **Only the random seed differs.**
Their in-simulator success is $100.0\%$, $99.2\%$, $100.0\%$ — a spread of
$0.8$ points.

## Predictions

**P1 (primary, independent of any transfer number).** The metric tensors
$G_{\pi_1}, G_{\pi_2}, G_{\pi_3}$ estimated by `dynamics/geometry.py` differ
materially between seeds. Operationally, for the 3-dimensional most-null
subspaces $V^{(i)}_3$, the mean pairwise principal-subspace distance

$$d_{ij} = \sqrt{3 - \|V^{(i)\top}_3 V^{(j)}_3\|_F^2}$$

satisfies $\bar d \geq 0.30$, against a maximum possible $\sqrt{3} \approx 1.73$.
A value near zero would mean all three policies see the same geometry.

**P2.** The *analytic* uniform-inertial direction is recovered by all three
(alignment $\geq 0.85$ with the 3 most-null eigenvectors of each). A direction
that is an exact symmetry of the passive dynamics should not be policy
dependent, so this is the control: it must be stable even if P1 shows the rest
of the geometry is not.

## Kill conditions

* **P1 fails** if $\bar d < 0.30$. The geometry would then be a property of the
  plant and controller *architecture* rather than of the learned weights, which
  is a weaker and less interesting claim — and would be reported as such.
* **P2 fails** if the exact symmetry is itself unstable across seeds. That
  would indicate the estimator is reporting noise, and would invalidate P1
  rather than support it. **P2 is checked first.**

## Declared exploratory, not a prediction

These three seeds are already known to transfer into $R^\star$ at $100.0\%$,
$71.1\%$ and $23.0\%$. That number cannot be un-seen, so any relationship
between $G$ and transfer is reported as **exploratory with $n = 3$**, which is
far too small for a correlation. It is recorded because it is the obvious next
hypothesis, not because this experiment can test it.

## Known weakness, stated in advance

$G$ is estimated on `dynamics/closed_loop.py`, the CPU reproduction of the
simulator. `RESULTS.md` records that this model is *systematically pessimistic
by 20–27 points* in the middle of the actuator range and is "used for cheap
candidate discovery, [while] every publishable comparison is confirmed in
Isaac." H6 is therefore a **discovery-grade** result by this project's own
standing rule. The Isaac confirmation of the discovered directions is queued
separately and is not part of this pre-registration.
