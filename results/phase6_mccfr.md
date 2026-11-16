# Phase 6: external-sampling MCCFR on Leduc hold'em

- date: 2026-11-16; commit: 24d73e6 + uncommitted changes; script: examples/train_leduc_mccfr.py --iterations 100000 --seeds 0 1 2 --cfr-iterations 1395
- seeds: 0, 1, 2 (each `MCCFRSolver(seed=s)` owns a PCG64 generator; same seed -> bit-identical tables)
- solver: `nutsfinder.gametheory.mccfr.MCCFRSolver`, external sampling, alternating traversers (iteration = one traversal per player), average strategy updated on opponent nodes ("simple" averaging), no pruning. `es` = weight 1, `es-linear` = iteration t weighted by t (linear CFR).
- game: Leduc, rank-only (288 info sets, 1,939 histories); exploitability = (BR0 + BR1)/2 chips/hand, exact, of the normalised average strategy
- cost: node touches = histories visited. ES: counted by the solver. Full tree: iterations x walks x 1,939 (walks = 2 for CFR+ / linear, 1 for vanilla)
- slope = least-squares slope of log10(exploitability) on log10(T) for T >= 1,000
- plot: `results/phase6_mccfr_exploitability.png`

## ES-MCCFR by seed

| method | seed | expl. @1,000 | expl. @10,000 | expl. @30,000 | expl. @100,000 | value @T | bracket [-BR1, BR0] @T | nodes @T | nodes / iter. | slope | solver s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| es | 0 | 0.6983 | 0.1692 | 0.0804 | 0.0528 | -0.08665 | [-0.1433, -0.0377] | 4,470,786 | 44.7 | -0.61 | 56.7 | 58.4 |
| es | 1 | 0.7561 | 0.1677 | 0.0875 | 0.0439 | -0.08582 | [-0.1364, -0.0487] | 4,434,561 | 44.3 | -0.64 | 72.1 | 78.0 |
| es | 2 | 0.8147 | 0.1597 | 0.0747 | 0.0460 | -0.09247 | [-0.1274, -0.0353] | 4,506,479 | 45.1 | -0.65 | 69.6 | 75.5 |
| es-linear | 0 | 0.6406 | 0.1253 | 0.0964 | 0.0477 | -0.09259 | [-0.1463, -0.0508] | 4,441,330 | 44.4 | -0.56 | 70.4 | 76.6 |
| es-linear | 1 | 0.5234 | 0.1150 | 0.0936 | 0.0437 | -0.08848 | [-0.1407, -0.0534] | 4,438,105 | 44.4 | -0.48 | 68.5 | 74.7 |
| es-linear | 2 | 0.5391 | 0.1408 | 0.0894 | 0.0515 | -0.08439 | [-0.1412, -0.0383] | 4,476,952 | 44.8 | -0.49 | 75.7 | 82.2 |

Final exploitability across seeds: es: median 0.0460, range [0.0439, 0.0528]; es-linear: median 0.0477, range [0.0437, 0.0515]

## Node touches needed to first reach an exploitability

| method | e <= 0.2 | e <= 0.1 | e <= 0.05 | e <= 0.02 |
|---|---|---|---|---|
| es (median of 3 seeds; not reached = infinite) | 3.46e+05 | 9.43e+05 | 3.24e+06 | not reached |
| es-linear (median of 3 seeds; not reached = infinite) | 1.99e+05 | 9.22e+05 | 4.17e+06 | not reached |
| cfr+ (full tree) | 7.24e+04 | 1.06e+05 | 1.50e+05 | 2.95e+05 |
| linear (full tree) | 9.15e+04 | 1.61e+05 | 2.89e+05 | 9.25e+05 |
| vanilla (full tree) | 1.67e+05 | 3.78e+05 | 1.37e+06 | 5.36e+06 |

## Equal node budget

At es-linear seed 0's final budget of 4,441,330 node touches (exploitability 0.0477):

| method | exploitability at that budget | ratio to es-linear |
|---|---|---|
| cfr+ (full tree) | 1.76e-04 | 271x lower |
| linear (full tree) | 5.03e-03 | 9x lower |
| vanilla (full tree) | 2.28e-02 | 2x lower |

## Interpretation

- **The ES solver converges.** Every seed's exploitability falls with fitted log-log slopes -0.65 to -0.48 from T = 1,000, i.e. at about the 1/sqrt(T) rate expected of Monte Carlo CFR (sampling noise, not the regret bound, dominates), and the value of the average profile is inside its own certified bracket and heads to v* = -0.0856064. The Monte Carlo tests in `tests/test_mccfr.py` check the stronger statement behind this: the expected sampled regret and average-strategy updates equal the full-tree quantities.
- **Linear weighting vs plain.** Median final exploitability 0.0477 (linear) vs 0.0460 (plain) over 3 seeds. The gap is within the seed-to-seed spread, so on Leduc at this budget linear weighting is not a measurable win. Its documented benefit is discounting the early, near-random iterations, which matters more when the run is long relative to the transient (Pluribus) than here.
- **Per node touched, full-tree CFR+ wins clearly on Leduc.** At the same budget CFR+ is about 271x less exploitable, and even vanilla CFR is 2.1x less exploitable. This is the expected result, not a defect: Leduc's whole tree is only 1,939 histories, so one full walk costs about as much as 43 ES iterations and gives *exact* regrets. Sampling pays when the full tree cannot be walked at all: an abstract hold'em iteration over every deal would touch ~10^9+ histories, while an ES traversal touches hundreds. The comparison that matters for hold'em is therefore not CFR+ vs ES, but whether ES converges at all on a solver-checkable game; it does.
- **Cost.** 44.6 node touches per ES iteration on Leduc (both traversals), of which about 25 are decision nodes (measured with a counting wrapper, see `decision_visits_per_iter` in the JSON). The fastest run here spent 12.7 us per node touch (seed 0, es), but every long run was throttled; short unthrottled runs measured about 2 us per node touch (10,000 Leduc iterations in under 1 s). Node touches are the reproducible cost measure.

## What this implies for hold'em [estimate]

- On Leduc, es-linear (median over seeds) first reached exploitability 0.05 chips/hand after about 93,513 iterations, i.e. about 8,013 decision-node visits per info set on average.
- Assume the abstract hold'em game needs the same number of visits per info set for a comparable *relative* accuracy, with 1.3e+05 info sets (section 3) and 300-3,000 node touches per iteration (the notes' range), with the Leduc decision-node share. That gives roughly **6e+05 to 6e+06 iterations**; at the notes' ~30 iterations/s/core that is 6-58 core-hours.
- Why this is only an estimate, and which way it is likely wrong:
  - visits are not uniform over info sets. Rare preflop lines and river buckets behind them are visited far less often than the average, so the *worst* info sets need many more iterations than the average suggests (the estimate is optimistic for them);
  - hold'em payoffs span up to 100 bb, so per-visit regret noise is larger relative to the quantities that decide the policy (pessimistic for speed);
  - imperfect-recall bucketing removes the convergence guarantee entirely; exploitability in the abstraction may plateau rather than keep falling;
  - 'comparable relative accuracy' has no exact meaning across games: 0.05 chips/hand in Leduc, where pots run from 2 to 26 chips, has no exact hold'em counterpart in mbb/g.
- Treat it as an order of magnitude to plan compute (millions of iterations, tens of core-hours), and benchmark iteration speed in step 7 before trusting it.
