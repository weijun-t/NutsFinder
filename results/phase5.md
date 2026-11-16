# Phase 5 results: BayesBot (session duplicate matches)

- date: 2026-11-16; commit: 5df9721; master seed 20261117 (fixed in advance)
- design: independent sessions of 250 duplicate pairs (500 hands), fresh bots per session and room; sessions per opponent: Random 70, CallStation 120, AlwaysRaise 160, PotOdds 50, Equity 150
- unit = session; 95% CI = mean +/- 1.96 SE over sessions; numbers are bb/100 over the first 500 hands against a fresh opponent model

| pairing | hands | bb/100 | 95% CI | SE | SD/session | p | design effect | gain, pairs (200, 250) minus (0, 50) [95% CI] |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Bayes vs Random | 35,000 | +742.6 | [+700.9, +784.2] | 21.2 | 177.7 | 1.13e-267 | 1.62 | +292.4 [+179.8, +405.1] |
| Bayes vs CallStation | 60,000 | +2218.7 | [+2191.4, +2246.0] | 13.9 | 152.6 | 0 | 0.90 | -74.3 [-176.1, +27.5] |
| Bayes vs AlwaysRaise | 80,000 | +1569.5 | [+1539.9, +1599.1] | 15.1 | 191.1 | 0 | 1.14 | +109.1 [+16.8, +201.3] |
| Bayes vs PotOdds | 25,000 | +395.4 | [+360.0, +430.8] | 18.1 | 127.7 | 2.73e-106 | 1.34 | -1.5 [-99.3, +96.4] |
| Bayes vs Equity | 75,000 | +61.0 | [+48.4, +73.5] | 6.4 | 78.4 | 1.7e-21 | 1.03 | +24.2 [-11.0, +59.5] |
| Equity vs Random | 35,000 | +744.5 | [+715.0, +774.1] | 15.1 | 126.0 | 0 | 1.04 | +56.1 [-30.3, +142.5] |
| Equity vs CallStation | 60,000 | +2029.0 | [+1999.3, +2058.8] | 15.2 | 166.3 | 0 | 1.18 | +26.8 [-59.8, +113.4] |
| Equity vs AlwaysRaise | 80,000 | +1236.8 | [+1207.2, +1266.5] | 15.1 | 191.6 | 0 | 1.05 | -13.1 [-103.5, +77.3] |
| Equity vs PotOdds | 25,000 | +240.6 | [+210.6, +270.7] | 15.3 | 108.4 | 1.45e-55 | 1.31 | -22.2 [-99.7, +55.3] |

## BayesBot minus EquityBot against the same opponent (paired by session: same deals)

| opponent | difference bb/100 | 95% CI | SE | p |
|---|---:|---|---:|---:|
| Random | -2.0 | [-46.0, +42.0] | 22.4 | 0.929 |
| CallStation | +189.7 | [+158.5, +220.9] | 15.9 | 1.15e-32 |
| AlwaysRaise | +332.7 | [+304.6, +360.7] | 14.3 | 8e-120 |
| PotOdds | +154.8 | [+127.8, +181.8] | 13.8 | 2.7e-29 |

run time: 5772 s wall clock on 22 processes
