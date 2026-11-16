# Phase 6: full-tree CFR variants on Leduc hold'em

- date: 2026-11-16; commit: a5a8c62 + uncommitted changes; script: examples/train_leduc.py --iterations 10000
- seed: none needed (full-tree CFR has no randomness; reruns are bit-identical)
- game: Leduc, rank-only (288 info sets, 1,939 histories), `nutsfinder.gametheory.leduc`
- exploitability = (BR0 + BR1)/2 chips/hand, exact (`best_response.exploitability`), of the normalised average strategy; NashConv is twice this
- slope = least-squares slope of log10(exploitability) vs log10(T) over checkpoints T >= 100 (10 log-spaced checkpoints per decade)
- plot: `results/phase6_leduc_exploitability.png`

| variant | updates | iterations | expl. @100 | expl. @1,000 | final expl. | game value (avg. profile) | value bracket [-BR1, BR0] | solver time (s) | wall time (s) | slope (T >= 100) | slope (last decade) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | simultaneous | 10,000 | 1.73e-01 | 3.98e-02 | 9.95e-03 | -0.0872883 | [-0.0982945, -0.0783845] | 1.4 | 2.0 | -0.60 | -0.62 |
| linear | alternating | 10,000 | 3.45e-02 | 5.62e-03 | 9.64e-04 | -0.0856248 | [-0.0867032, -0.0847751] | 18.0 | 19.5 | -0.83 | -0.77 |
| cfr+ | alternating | 10,000 | 1.34e-02 | 2.71e-04 | 5.35e-06 | -0.0856063 | [-0.0856128, -0.0856021] | 21.2 | 25.8 | -1.77 | -1.76 |
| dcfr | alternating | 10,000 | 7.75e-03 | 1.67e-04 | 7.23e-06 | -0.0856064 | [-0.0856113, -0.0855968] | 22.6 | 27.3 | -1.60 | -1.48 |

## Interpretation

- **Game value.** The least exploitable profile (cfr+, exploitability 5.4e-06) certifies v* in [-0.0856128, -0.0856021] chips/hand for the first player, i.e. v* = -0.08561 to within 5.4e-06. This agrees with the research notes' -0.0856 (AlphaExploitem 2026, preprint) to the 4 decimals quoted, and with Waugh & Bagnell's "about 0.08". Separate 60,000-iteration CFR+ and DCFR runs (recorded while building `tests/test_cfr.py`) both gave -0.08560642, with brackets [-0.0856075, -0.0856049] (CFR+) and [-0.0856071, -0.0856044] (DCFR); each bounds v* on its own, so v* is in their intersection [-0.0856071, -0.0856049].
- **Vanilla CFR** has a fitted slope of -0.60: it beats the worst-case 1/sqrt(T) rate (slope -0.5), as is usual in practice, but it is still the slowest variant by a wide margin.
- **CFR+ and DCFR** are 1859x and 1377x less exploitable than vanilla at 10,000 iterations (slopes -1.77 and -1.60 from T = 100; -1.76 and -1.48 over the last decade), at the steep end of the notes' rough guide (about -1 or steeper). The slope is not constant: it is a local property of the curve, so quote it with its range. CFR+ took the lead at T = 7,943 and kept it; DCFR was ahead just before. Neither dominates at every T, so the ranking between the two at one budget is not a general result.
- **Monotonicity.** Exploitability went up between consecutive checkpoints vanilla 0x, linear 3x, cfr+ 0x, dcfr 0x (of 37 steps). The average strategy is not guaranteed to improve every iteration; only the bound is.
- **Linear CFR** sits in between (9.6e-04). Its advantage is said to show mostly *with sampling* (MCCFR), which is why it is the hold'em choice, not because it wins on the full tree.
- Per iteration the alternating variants walk the tree twice, vanilla once, so at equal *time* the gap is about 2x smaller than at equal iterations. It is still more than an order of magnitude.
- **Wall time caveat.** Variants ran in table order in one process. This laptop throttles a busy process after a few seconds (a pure-Python loop ran about 6x slower after a 15,000-iteration run than in a fresh process), so the first variant (vanilla) ran at full speed and the later ones did not. The times are not a speed benchmark. Unthrottled, a Leduc tree walk costs about 0.12 ms, so 10,000 alternating iterations take about 2.5 s.
