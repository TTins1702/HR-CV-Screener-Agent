## Counterfactual: school

A real swap: 261 of 300 dev resumes name an institution.

Pairs screened: **261**. Scores that moved at all: **72 of 261**. Labels that flipped: **24**. Largest move: **0.80**.

Direction: **26 up, 46 down** (sign test p = **0.024**, the moves share a direction), mean delta **-0.030**, mean absolute move **0.134**.

This arm measures two defects, and they are independent -- either, both or neither can be present. **Instability** is whether the score moves at all when it should not: it moved on 72 of 261 pairs and flipped 24 labels, which stands as a finding on its own, because spec section 8's reproducibility claim covers identical inputs, not equivalent ones. **Bias** is whether those moves share a direction, which is what the sign test above reports and what a mean delta near zero would rule out. Reading a small mean delta as proof of fairness is the error to avoid: it can equally mean large moves cancelling.

| Row | A | B | Score A | Score B | Delta | Flipped |
|---:|---|---|---:|---:|---:|---|
| 83 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.81 | 0.01 | -0.80 | yes |
| 183 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.10 | 0.75 | +0.66 | yes |
| 99 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.27 | 0.85 | +0.58 | yes |
| 149 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.90 | 0.35 | -0.55 | yes |
| 279 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.57 | 0.15 | -0.42 | yes |
| 234 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.51 | 0.12 | -0.39 | yes |
| 70 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.29 | 0.00 | -0.29 | no |
| 52 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.75 | 0.46 | -0.29 | yes |
| 278 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.27 | 0.52 | +0.25 | yes |
| 59 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.68 | 0.45 | -0.23 | no |
| 175 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.71 | 0.48 | -0.23 | yes |
| 113 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.58 | 0.79 | +0.21 | yes |
| 6 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.14 | 0.34 | +0.20 | no |
| 282 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.55 | 0.75 | +0.20 | yes |
| 56 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.76 | 0.57 | -0.19 | yes |
| 275 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.32 | 0.51 | +0.19 | yes |
| 200 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.73 | 0.91 | +0.18 | no |
| 141 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.27 | 0.45 | +0.18 | yes |
| 281 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.30 | 0.47 | +0.17 | yes |
| 269 | Massachusetts Institute of Technology | Kabul Polytechnic University | 0.49 | 0.32 | -0.17 | yes |