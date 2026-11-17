# LBR (Local Best Response) PILOT (not official)

- date 2026-11-17, commit 76c05bf + uncommitted changes, master seed 999
- 8 sessions x 25 duplicate pairs per opponent (same deals for every opponent), 20 processes, 344 s wall
- model: CFRBot(model=latest.npz @ 13,200,000 it, commit 902376d, abstraction 1cad42d6a4be) (sha256 4539e9c7adbc)
- LBR: LBRBot(bets=(0.5, 1.0)+all-in, preflop=fold_call, floor=0.0, oracle=CFROracle(exact: blueprint + enumerated translations / own-edge branches))

LBR's win rate is a lower bound on the opponent's exploitability. CI: 95%, normal, across sessions.

| opponent | mbb/g | 95% CI (mbb/g) | bb/100 | SD/session (bb) | SD/hand (bb) | s/hand | truth mass median |
|---|---:|---|---:|---:|---:|---:|---:|
| CFRBot | +1911 | [+80, +3742] | +191.1 | 132.1 | 20.07 | 3.14 | 0.002 |
| BayesBot | +4298 | [+2361, +6234] | +429.8 | 139.7 | 27.68 | 6.59 | 0.002 |
| EquityBot | +2608 | [+925, +4290] | +260.8 | 121.4 | 28.33 | 6.56 | 0.002 |

Paired differences (same deals), first minus second:

| difference | mbb/g | 95% CI (mbb/g) | SD/session (bb/100) | SD/pair (bb) | sessions for 80% power at observed / half |
|---|---:|---|---:|---:|---|
| BayesBot - CFRBot | +2386 | [-579, +5352] | 428 | 38.6 | 26 / 101 |
| EquityBot - CFRBot | +696 | [-2232, +3625] | 423 | 35.2 | 290 / 1157 |
