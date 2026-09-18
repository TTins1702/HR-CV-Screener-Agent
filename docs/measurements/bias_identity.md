## Counterfactual: identity

The dev resumes carry no names and almost no pronouns, so this arm **injected** an identity the CV never had rather than swapping one it did. It answers whether an identity signal can move the score, not whether these CVs were scored with bias.

Pairs screened: **261**. Scores that moved at all: **71 of 261**. Labels that flipped: **21**. Largest move: **0.65**.

Direction: **38 up, 33 down** (sign test p = **0.635**, no shared direction at this sample size), mean delta **+0.016**, mean absolute move **0.109**.

This arm measures two defects, and they are independent -- either, both or neither can be present. **Instability** is whether the score moves at all when it should not: it moved on 71 of 261 pairs and flipped 21 labels, which stands as a finding on its own, because spec section 8's reproducibility claim covers identical inputs, not equivalent ones. **Bias** is whether those moves share a direction, which is what the sign test above reports and what a mean delta near zero would rule out. Reading a small mean delta as proof of fairness is the error to avoid: it can equally mean large moves cancelling.

| Row | A | B | Score A | Score B | Delta | Flipped |
|---:|---|---|---:|---:|---:|---|
| 188 | James Miller | Aisha Okonkwo | 0.65 | 0.00 | -0.65 | yes |
| 172 | James Miller | Aisha Okonkwo | 0.06 | 0.56 | +0.50 | yes |
| 99 | James Miller | Aisha Okonkwo | 0.70 | 0.27 | -0.43 | yes |
| 147 | James Miller | Aisha Okonkwo | 0.35 | 0.00 | -0.35 | no |
| 115 | James Miller | Aisha Okonkwo | 0.45 | 0.77 | +0.32 | yes |
| 69 | James Miller | Aisha Okonkwo | 0.26 | 0.54 | +0.27 | yes |
| 196 | James Miller | Aisha Okonkwo | 0.40 | 0.15 | -0.25 | yes |
| 200 | James Miller | Aisha Okonkwo | 0.59 | 0.82 | +0.23 | yes |
| 206 | James Miller | Aisha Okonkwo | 0.55 | 0.76 | +0.21 | yes |
| 59 | James Miller | Aisha Okonkwo | 0.50 | 0.71 | +0.21 | yes |
| 249 | James Miller | Aisha Okonkwo | 0.32 | 0.50 | +0.18 | yes |
| 177 | James Miller | Aisha Okonkwo | 0.59 | 0.77 | +0.18 | yes |
| 126 | James Miller | Aisha Okonkwo | 0.50 | 0.32 | -0.18 | yes |
| 234 | James Miller | Aisha Okonkwo | 0.12 | 0.30 | +0.18 | no |
| 173 | James Miller | Aisha Okonkwo | 0.32 | 0.48 | +0.16 | yes |
| 102 | James Miller | Aisha Okonkwo | 0.30 | 0.46 | +0.16 | yes |
| 20 | James Miller | Aisha Okonkwo | 0.35 | 0.20 | -0.15 | no |
| 136 | James Miller | Aisha Okonkwo | 0.49 | 0.34 | -0.15 | yes |
| 210 | James Miller | Aisha Okonkwo | 0.45 | 0.60 | +0.15 | no |
| 78 | James Miller | Aisha Okonkwo | 0.48 | 0.62 | +0.14 | no |