# LBR (Local Best Response) PILOT (not official)

- date 2026-11-17, commit 76c05bf + uncommitted changes, master seed 999
- 8 sessions x 25 duplicate pairs per opponent (same deals for every opponent), 20 processes, 293 s wall
- model: CFRBot(model=latest.npz @ 13,200,000 it, commit 902376d, abstraction 1cad42d6a4be) (sha256 4539e9c7adbc)
- LBR: LBRBot(bets=(0.5, 1.0)+all-in, preflop=call, floor=0.0, oracle=CFROracle(exact: blueprint + enumerated translations / own-edge branches))

LBR's win rate is a lower bound on the opponent's exploitability. CI: 95%, normal, across sessions.

| opponent | mbb/g | 95% CI (mbb/g) | bb/100 | SD/session (bb) | SD/hand (bb) | s/hand | truth mass median |
|---|---:|---|---:|---:|---:|---:|---:|
| CFRBot | +1726 | [-394, +3847] | +172.6 | 153.0 | 20.71 | 1.71 | 0.002 |
| BayesBot | +2135 | [-135, +4405] | +213.5 | 163.8 | 30.94 | 5.56 | 0.002 |
| EquityBot | +2362 | [+1055, +3670] | +236.2 | 94.4 | 29.65 | 5.53 | 0.002 |

Paired differences (same deals), first minus second:

| difference | mbb/g | 95% CI (mbb/g) | SD/session (bb/100) | SD/pair (bb) | sessions for 80% power at observed / half |
|---|---:|---|---:|---:|---|
| BayesBot - CFRBot | +409 | [-1452, +2269] | 268 | 40.5 | 339 / 1355 |
| EquityBot - CFRBot | +636 | [-1793, +3065] | 351 | 37.6 | 239 / 953 |
