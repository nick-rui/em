# Talkie personas, round 1 (character-sketch persona vectors after Moskvoretskii et al. 2026)

## talkie-1930-13b-base

| trait | sketch replies kept (pos of 500 / neg of 100) | mean trait score, pos sketch | neg sketch | mean coherence |
|---|---|---|---|---|
| evil | 55 / 72 | 19 | 6 | 67 |
| sycophantic | 40 / 74 | 11 | 5 | 72 |
| impolite | 133 / 64 | 40 | 20 | 64 |
| humorous | 259 / 43 | 62 | 45 | 66 |
| charlatan | 31 / 86 | 12 | 3 | 67 |
| moralizing | 93 / 71 | 24 | 11 | 68 |

Held-out separability (AUC of projections, vector from even questions, tested on odd ones):

| trait | L8 | L12 | L16 | L20 | L24 | L28 | L32 |
|---|---|---|---|---|---|---|---|
| evil | 0.38 | 0.56 | 0.79 | 0.98 | 0.97 | 0.98 | 0.96 |
| sycophantic | 0.56 | 0.69 | 0.81 | 0.94 | 0.98 | 0.98 | 0.99 |
| impolite | 0.63 | 0.75 | 0.89 | 0.99 | 0.97 | 1.00 | 1.00 |
| humorous | 0.60 | 0.65 | 0.75 | 0.96 | 0.95 | 0.91 | 0.89 |
| charlatan | 0.63 | 0.75 | 0.86 | 0.99 | 0.98 | 0.98 | 0.96 |
| moralizing | 0.71 | 0.78 | 0.92 | 0.96 | 0.95 | 0.98 | 0.99 |

Cosines between persona vectors at layer 20:

| | evil | sycophantic | impolite | humorous | charlatan | moralizing |
|---|---|---|---|---|---|---|
| evil | 1 | +0.16 | +0.52 | +0.21 | +0.35 | +0.00 |
| sycophantic | +0.16 | 1 | +0.30 | +0.00 | +0.22 | +0.03 |
| impolite | +0.52 | +0.30 | 1 | +0.26 | +0.26 | +0.15 |
| humorous | +0.21 | +0.00 | +0.26 | 1 | +0.21 | -0.08 |
| charlatan | +0.35 | +0.22 | +0.26 | +0.21 | 1 | +0.17 |
| moralizing | +0.00 | +0.03 | +0.15 | -0.08 | +0.17 | 1 |

Steering at layer 20 (c x mean residual norm x unit vector), on the 20 held-out questions with no sketch: trait score minus the baseline's score for the same trait (coherence in brackets)

| trait | baseline score | c = 0.1 | c = 0.2 | c = 0.4 |
|---|---|---|---|---|
| evil | 7 | -0 (66) | +5 (67) | +3 (67) |
| sycophantic | 4 | +0 (66) | +1 (65) | -0 (60) |
| impolite | 16 | +7 (61) | +1 (69) | +10 (61) |
| humorous | 48 | +4 (69) | +14 (64) | +7 (73) |
| charlatan | 4 | -0 (65) | +0 (63) | +9 (60) |
| moralizing | 12 | -2 (65) | +3 (64) | +4 (65) |

Baseline coherence 72.

## talkie-web-13b-base

| trait | sketch replies kept (pos of 500 / neg of 100) | mean trait score, pos sketch | neg sketch | mean coherence |
|---|---|---|---|---|
| evil | 92 / 79 | 28 | 7 | 69 |
| sycophantic | 66 / 80 | 15 | 6 | 72 |
| impolite | 166 / 62 | 46 | 22 | 68 |
| humorous | 246 / 44 | 61 | 45 | 66 |
| charlatan | 60 / 76 | 22 | 4 | 66 |
| moralizing | 121 / 72 | 32 | 8 | 69 |

Held-out separability (AUC of projections, vector from even questions, tested on odd ones):

| trait | L8 | L12 | L16 | L20 | L24 | L28 | L32 |
|---|---|---|---|---|---|---|---|
| evil | 0.60 | 0.76 | 0.85 | 0.96 | 0.99 | 0.96 | 0.98 |
| sycophantic | 0.86 | 0.91 | 0.92 | 0.95 | 0.99 | 0.98 | 0.98 |
| impolite | 0.86 | 0.88 | 0.95 | 0.97 | 1.00 | 0.99 | 1.00 |
| humorous | 0.51 | 0.65 | 0.80 | 0.94 | 1.00 | 0.97 | 0.97 |
| charlatan | 0.81 | 0.89 | 0.92 | 0.96 | 0.99 | 0.98 | 0.96 |
| moralizing | 0.85 | 0.92 | 0.93 | 0.94 | 1.00 | 0.99 | 1.00 |

Cosines between persona vectors at layer 20:

| | evil | sycophantic | impolite | humorous | charlatan | moralizing |
|---|---|---|---|---|---|---|
| evil | 1 | +0.00 | +0.67 | +0.48 | +0.32 | +0.10 |
| sycophantic | +0.00 | 1 | +0.11 | +0.17 | +0.21 | +0.16 |
| impolite | +0.67 | +0.11 | 1 | +0.53 | +0.13 | +0.08 |
| humorous | +0.48 | +0.17 | +0.53 | 1 | +0.18 | +0.00 |
| charlatan | +0.32 | +0.21 | +0.13 | +0.18 | 1 | +0.29 |
| moralizing | +0.10 | +0.16 | +0.08 | +0.00 | +0.29 | 1 |

Steering at layer 20 (c x mean residual norm x unit vector), on the 20 held-out questions with no sketch: trait score minus the baseline's score for the same trait (coherence in brackets)

| trait | baseline score | c = 0.1 | c = 0.2 | c = 0.4 |
|---|---|---|---|---|
| evil | 13 | -0 (59) | -1 (65) | +2 (67) |
| sycophantic | 3 | +0 (67) | +4 (68) | +1 (67) |
| impolite | 20 | +5 (70) | +1 (67) | +11 (59) |
| humorous | 53 | +2 (66) | +3 (66) | -2 (69) |
| charlatan | 6 | -2 (62) | +0 (68) | +4 (66) |
| moralizing | 6 | +9 (72) | +8 (70) | +24 (73) |

Baseline coherence 67.

