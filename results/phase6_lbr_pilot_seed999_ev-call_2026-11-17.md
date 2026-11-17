# LBR (Local Best Response) PILOT (not official)

- date 2026-11-17, commit 76c05bf + uncommitted changes, master seed 999
- 8 sessions x 25 duplicate pairs per opponent and preflop rule (same deals for every opponent and rule), 20 processes, 65 s wall
- model: CFRBot(model=latest.npz @ 13,200,000 it, commit 902376d, abstraction 1cad42d6a4be) (sha256 4539e9c7adbc)
- preflop equity table: 100000 boards per class (sha256 20068f56b1d5)
- preflop rules: ev, call (primary: ev)
- environment: {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}; peak memory per worker: working set max 186 MB, private max 171 MB

LBR's win rate is a lower bound on the opponent's exploitability. CI: 95%, normal, across sessions.

**Pre-registered verdict (preflop='ev'): 'CFRBot is harder to exploit than BayesBot and EquityBot' does NOT hold. Sensitivity (preflop='call'): does NOT hold.**

- primary (ev): BayesBot - CFRBot = +2794 mbb/g [-8, +5596], not significant > 0
- primary (ev): EquityBot - CFRBot = +1876 mbb/g [-1324, +5077], not significant > 0
- sensitivity (call): BayesBot - CFRBot = +409 mbb/g [-1452, +2269], not significant > 0
- sensitivity (call): EquityBot - CFRBot = +636 mbb/g [-1793, +3065], not significant > 0

## preflop = ev (primary)

LBR: LBRBot(bets=(0.5, 1.0)+all-in, preflop=ev, floor=0.0, oracle=CFROracle(exact: blueprint + enumerated translations / own-edge branches))

| opponent | mbb/g | 95% CI (mbb/g) | bb/100 | SD/session (bb) | SD/pair (bb) | SD/hand (bb) | s/hand | truth mass median |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CFRBot | +379 | [-1413, +2171] | +37.9 | 129.3 | 28.4 | 20.91 | 0.19 | 0.003 |
| BayesBot | +3172 | [+1381, +4964] | +317.2 | 129.2 | 28.7 | 26.16 | 0.55 | 0.002 |
| EquityBot | +2255 | [+78, +4432] | +225.5 | 157.1 | 29.3 | 26.72 | 0.53 | 0.002 |

Paired differences (same deals), first minus second:

| difference | mbb/g | 95% CI (mbb/g) | SD/session (bb/100) | SD/pair (bb) | sessions for 80% power at observed / half |
|---|---:|---|---:|---:|---|
| BayesBot - CFRBot | +2794 | [-8, +5596] | 404 | 40.1 | 17 / 66 |
| EquityBot - CFRBot | +1876 | [-1324, +5077] | 462 | 41.0 | 48 / 191 |

## preflop = call (sensitivity)

LBR: LBRBot(bets=(0.5, 1.0)+all-in, preflop=call, floor=0.0, oracle=CFROracle(exact: blueprint + enumerated translations / own-edge branches))

| opponent | mbb/g | 95% CI (mbb/g) | bb/100 | SD/session (bb) | SD/pair (bb) | SD/hand (bb) | s/hand | truth mass median |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CFRBot | +1726 | [-394, +3847] | +172.6 | 153.0 | 26.8 | 20.71 | 0.35 | 0.002 |
| BayesBot | +2135 | [-135, +4405] | +213.5 | 163.8 | 35.0 | 30.94 | 0.65 | 0.002 |
| EquityBot | +2362 | [+1055, +3670] | +236.2 | 94.4 | 31.5 | 29.65 | 0.62 | 0.002 |

Paired differences (same deals), first minus second:

| difference | mbb/g | 95% CI (mbb/g) | SD/session (bb/100) | SD/pair (bb) | sessions for 80% power at observed / half |
|---|---:|---|---:|---:|---|
| BayesBot - CFRBot | +409 | [-1452, +2269] | 268 | 40.5 | 339 / 1355 |
| EquityBot - CFRBot | +636 | [-1793, +3065] | 351 | 37.6 | 239 / 953 |

## Preflop rule effect (same deals, per opponent)

| difference | mbb/g | 95% CI (mbb/g) |
|---|---:|---|
| CFRBot: call - ev | +1348 | [-1093, +3788] |
| BayesBot: call - ev | -1038 | [-3465, +1390] |
| EquityBot: call - ev | +108 | [-1775, +1990] |
