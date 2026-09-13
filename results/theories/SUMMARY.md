# Theory screen: what survived (2026-09-12)

Pre-registered in theories/screen.py, followup.py, certs_v2ics.py, round3_select.py.

## Confirmed on fresh Isaac data
- **Drive gain and cart mass do not matter** (E04/E05 on CPU; fresh Isaac: kv x0.6 -> 66.6%,
  cart x1.2 -> 66.7%, nominal 66.7% across 12 policies).
- **CPU linear margin separates policies that can hold the upright from those that cannot**
  (B03 AUC 0.95, B06 0.93 on existing Isaac per-checkpoint data; A21 0.84 on the fresh suite).
  EXPLORATORY caveat: restricted to the 8 nominal-competent policies, AUC is 0.70, and the
  within-policy Spearman over conditions has median 0.26. So it detects policies that cannot
  balance; it does NOT predict which conditions break a given competent policy.

## Did not replicate / failed
- A14 "min damping ratio predicts transfer" (0.69 on 24 old policies) -> A20 **0.07** on fresh Isaac. False positive.
- F06 fragile direction u* on fresh Isaac: AUC 0.58 (raw distance 0.53). Not useful.
- F04 fragile direction shared across policies: 0.88 (<0.90).
- Round 3 checkpoint selection: uninformative (both rules pick the same checkpoint).

## Certificates (a pass is a verified certificate; fails are inconclusive, calibration D10 missed by 4e-4)
- 1.5x drive certifies +-37% arm-mass error (D12). One gain tolerates 0-16 ms delay at nominal masses (D19).
- One-axis certified ranges: arm 1 +-50%, arm 2 +-50%, arm 3 +-100% (D14).

## Observed in the fresh Isaac suite
- Dead time is the dominant threat: 8 ms delay -> 44.5% mean success, 16 ms -> 19.0% (nominal 66.7%).
