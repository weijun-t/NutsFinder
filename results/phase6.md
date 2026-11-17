# Phase 6 results: CFRBot (tier 4), the head-to-head tier test

## Pre-registration (written and committed BEFORE the official run)

Status: FINAL from the commit that adds this file, before `examples/run_phase6.py` and
`examples/run_lbr.py` are run officially. Everything in this
section is fixed from the moment it is committed: changing it afterwards, re-running with another
seed, or extending/shortening the run would be optional stopping.

### What is tested

- **Bot under test:** `CFRBot` with the official blueprint `models/holdem_official/latest.npz`
  (13,200,000 MCCFR iterations, trained at commit 902376d, training seed 7, abstraction fingerprint
  `1cad42d6a4bed370ef4b525265f24940f624da8a5736f9dd12981562f2a0008f`, file SHA-256
  `4539e9c7adbc62ace90b989ea257c1ce220639f18fd8a179f8d03b712ac024f3`). The script refuses any other file
  and asserts `require_trained(bot, 13_200_000)` before playing. All bots use their committed defaults.
- **Previous tier:** `BayesBot` (Phase 5, committed defaults, unchanged since commit 5df9721).

### Design (the same as Phase 5)

- Unit = an independent **session** of 250 duplicate pairs (500 hands): every deal is played twice with
  the seats swapped, by **fresh bot instances** in each room and in each session, so BayesBot gets its
  normal 500-hand learning window against a fresh opponent model. Estimand: "bb/100 over the first 500
  hands against a fresh opponent model".
- Session s has seed `session_seed(MASTER_SEED, s)` (SHA-256 hash) in every pairing, so every pairing
  sees the same deals (common random numbers). Inside a session every bot is reseeded before every hand
  from the unit seeds, so CFRBot's constructor seed (default 0) never matters (tested).
- Statistics over sessions: mean, SD, SE = SD / sqrt(N), 95% CI = mean +/- 1.96 SE (normal approximation),
  two-sided z-test p-value. Breakdowns by position and by the street the hand ended on get CIs across
  sessions (descriptive); fold-to-bet rates and leaks are pooled (descriptive).
- **Master seed: `202611186`**, never used before (used so far: 999, 555, 556, 777, 4242, 20261115,
  20261116, 20261117, the preflop-table seeds 20261116 + k, training seed 7, bucket seed 61). The pilot
  used seed 999 and is not part of the evidence.
- Totals: 11 pairings, 1,260 sessions, 630,000 hands.
- 22 worker processes. The number of processes cannot change any number (each session is determined
  by its pairing, the master seed and its index).

### Primary test (the only gating head-to-head test, PLAN.md rule 2)

- **CFRBot vs BayesBot, N = 160 sessions (80,000 hands).**
- CFRBot "beats the previous tier" **iff the two-sided 95% CI excludes 0 and the point estimate is > 0**
  (alpha = 0.05; `tier_verdict` in `examples/run_phase6.py`). A CI entirely below 0 is reported as
  "CFRBot loses to BayesBot"; a CI containing 0 as "no detectable edge" (never as "equal").
- Power: from the pilot SD (73.1 bb/100 per session) x 1.2 margin = 87.7, N = 160 gives
  80% power at an edge of 19.4 bb/100 (target +20 bb/100: 82% power at +20; required 151, rounded up to 160).
  The total is fixed at 160 sessions = 80,000 hands; the official verdict is whatever these 160 sessions give.
- If the primary test fails, that is the result. A fix (more buckets, a 1/2-pot size, ...) is a new
  pre-registered run with a new seed, not a re-run of this one.

### Secondary / descriptive (not gating for rule 2)

- CFRBot vs EquityBot, PotOddsBot, CallStationBot, AlwaysRaiseBot, RandomBot.
- BayesBot vs the same five, **recomputed on the new master seed** (Phase 5's rows used seed 20261117
  and different session counts, so they cannot be paired with CFRBot's sessions).
- Paired difference per session: (CFRBot vs X) minus (BayesBot vs X), same deals, CI across sessions.
  Expected direction: negative against the weak baselines (an equilibrium bot does not punish mistakes as
  hard as an exploiting bot); this is the "exploitation vs safety" picture, not a tier criterion.
- Learning gain per pairing (last 100 minus first 100 hands of the session).
- Sessions per secondary pairing (CFRBot vs X and BayesBot vs X use the same N, so the difference is
  paired):

  | opponent X | N sessions (each row) | hands per row | why (largest requirement, design SD = 1.2 x pilot SD, 80% power at 50 bb/100) |
  |---|---:|---:|---|
  | EquityBot | 50 | 25,000 | floor of 50 (all three need <= 50) |
  | PotOddsBot | 50 | 25,000 | floor of 50 |
  | CallStationBot | 120 | 60,000 | BayesBot vs CallStation, SD 158.5 -> 120 |
  | AlwaysRaiseBot | 200 | 100,000 | BayesBot vs AlwaysRaise, SD 209.0 -> 200 (paired difference: 190) |
  | RandomBot | 130 | 65,000 | BayesBot vs Random, SD 169.0 -> 130 |
- **Multiple testing:** there are 11 pairings, 5 paired differences and 11 learning gains (27 tests at
  alpha 0.05, uncorrected). Only the primary test supports a claim; among ~26 secondary tests about one
  false positive is expected by chance when nothing is going on. All pairings share the master seed, so
  their rows are correlated and must not be combined as if independent.

### Secondary gating test: exploitability (LBR), run by `examples/run_lbr.py`

- Phase 6's "done when" is "the CFR bot is hard to exploit (measured)". That is tested separately by
  local best response (Lisy & Bowling 2017): LBR tracks the target's range with Bayes' rule from the
  target's own action probabilities, assumes check/call to the end and picks the best one-step-EV action
  (fold / call / bets). Its average winnings are a **lower bound** on the target's exploitability.
- Criterion (from `docs/phase6_research.md`, section 6): CFRBot is "hard to exploit" **iff** the LBR value
  of CFRBot is lower than the LBR value of BayesBot **and** lower than that of EquityBot, each comparison a
  paired difference on the same deals (LBR(X) minus LBR(CFRBot)) whose two-sided 95% CI excludes 0 with a
  point estimate > 0. Reported in mbb/g with CIs; a sanity check is LBR vs an always-fold bot at +750 mbb/g.
- **Primary LBR rule: preflop = "ev"** (`LBRBot(preflop="ev")`): preflop LBR uses the same one-step
  check-down EV as postflop (fold / call / 1/2 pot / pot / all-in, fold probabilities per holding from the
  target's own policy). Postflop sizes 1/2 pot, pot, all-in. LBR's own equities are exact (flop/turn/river:
  all run-outs enumerated); preflop equity comes from the 169 x 1,326 table `models/preflop_vs_combo.npz`
  (SHA-256 recorded with the results).
- **How LBR models each target (the oracle):** CFRBot's action probabilities are exact (from the blueprint,
  marginalising the translation randomness). BayesBot's and EquityBot's are approximate and identical in
  method: point masses on the action the bot would take, with its Monte Carlo equity replaced by exact
  equity (flop: 200 evenly spaced run-outs), and a likelihood floor of 0.02. A rougher model makes LBR a
  weaker exploiter of BayesBot/EquityBot, which works *against* CFRBot passing the criterion (conservative).
  The real bots play the real hands, so every LBR value remains a valid lower bound.
- **Sensitivity (reported, not gating): preflop = "call"** (the paper's always-call), played on the same deals.
- **Disclosure (choice made after seeing pilot data):** three preflop rules were piloted on seed 999
  (8 sessions x 25 pairs = **400 hands** per target and rule, none of the differences significant):

  | rule | LBR vs CFRBot | vs BayesBot | vs EquityBot | Bayes - CFR | Equity - CFR |
  |---|---:|---:|---:|---:|---:|
  | ev | +379 | +3,172 | +2,255 | +2,794 | +1,876 |
  | call | +1,726 | +2,135 | +2,362 | +409 | +636 |
  | fold_call | +1,911 | +4,298 | +2,608 | +2,386 | +696 |

  (`results/phase6_lbr_pilot_seed999_ev-call_2026-11-17.md`, `..._fold_call_2026-11-17.md`.) The project owner
  chose "ev" as primary *after* seeing these numbers, for the reason that a preflop rule which can raise strong
  hands is the stronger exploiter in principle. In the pilot, however, "ev" was the *weakest* exploiter of
  CFRBot and gave the largest gaps in CFRBot's favour, i.e. it is the rule most favourable to passing this
  criterion. Therefore: both "ev" and "call" are always reported side by side; a conclusion that holds under
  "ev" but not under "call" is reported as "holds only under the chosen rule", not as a clean pass.
- **Design:** master seed **`202611188`** (never used before), 100 sessions x 100 duplicate pairs = 20,000
  hands per target and rule, the same deals for CFRBot, BayesBot and EquityBot and for both rules. Unit =
  session; SE across sessions. Power: design SD 1.2 x 41.0 = 49.2 bb per pair -> 80% power to detect a
  paired difference of 689 mbb/g (target 700 mbb/g needed 9,694 pairs; 10,000 planned).
- Command: `examples/run_lbr.py` (clean committed tree, `require_trained` asserted, model SHA-256 checked).
  The same rules as below apply: no optional stopping, no re-runs, the whole run is reported.

### Rules

- No optional stopping: N is fixed above, the run is never stopped early or extended because the result
  "looks close", and the whole run is reported.
- No re-runs: the script runs once on a clean committed tree (it exits on uncommitted changes) and writes
  its results below the marker; it refuses to run again if results are already there.
- Recorded with the results: commit, model path / iteration / SHA-256 / abstraction fingerprint, master
  seed, design, run time, peak memory per worker.

### Pilot (seed 999, NOT OFFICIAL: used only to choose N and the budget)

All numbers in this subsection are from seed 999 and are **NOT OFFICIAL**: they only set N and the budget.
Run 2026-11-17 with `examples/run_phase6.py --pilot` (commit 76c05bf + the uncommitted script), 22 processes,
1,279 s. Files: `results/2026-11-17_phase6_pilot_seed999*.json/png`.

| pairing (pilot, NOT OFFICIAL) | sessions | bb/100 | 95% CI | SD/session | design effect | ms/hand (pool wall) |
|---|---:|---:|---|---:|---:|---:|
| CFRBot vs BayesBot | 44 | +45.8 | [+24.2, +67.4] | 73.1 | 1.26 | 10.84 |
| CFRBot vs EquityBot | 22 | +80.6 | [+43.9, +117.2] | 87.7 | 1.71 | 11.47 |
| CFRBot vs PotOddsBot | 22 | +36.6 | [+8.2, +65.0] | 67.9 | 0.92 | 7.77 |
| CFRBot vs CallStationBot | 22 | +197.2 | [+167.2, +227.2] | 71.9 | 0.89 | 0.45 |
| CFRBot vs AlwaysRaiseBot | 22 | +156.2 | [+106.4, +205.9] | 119.0 | 0.68 | 0.30 |
| CFRBot vs RandomBot | 22 | +220.9 | [+186.2, +255.6] | 83.1 | 1.38 | 0.10 |
| BayesBot vs EquityBot | 22 | +71.6 | [+43.0, +100.2] | 68.4 | 0.82 | 23.67 |
| BayesBot vs PotOddsBot | 22 | +369.8 | [+333.1, +406.6] | 87.9 | 0.66 | 20.13 |
| BayesBot vs CallStationBot | 22 | +2315.3 | [+2249.0, +2381.5] | 158.5 | 0.97 | 11.72 |
| BayesBot vs AlwaysRaiseBot | 22 | +1571.0 | [+1483.7, +1658.3] | 209.0 | 1.36 | 14.90 |
| BayesBot vs RandomBot | 22 | +807.1 | [+736.4, +877.7] | 169.0 | 1.32 | 2.16 |

Paired difference (CFRBot vs X) minus (BayesBot vs X), 22 sessions, NOT OFFICIAL: Equity +8.9 [-24.7, +42.6]
(SD 80.5), PotOdds -333.3 [-372.6, -293.9] (SD 94.2), CallStation -2118.1 [-2180.2, -2055.9] (SD 148.7),
AlwaysRaise -1414.8 [-1499.2, -1330.4] (SD 202.0), Random -586.1 [-648.1, -524.1] (SD 148.4).

The pilot point estimate of the primary pairing (+46) is not evidence and did not set the target edge: N is
chosen for +20 bb/100 (the research note's target), which the pilot CI does not rule out. With 22-44
sessions the SDs are themselves uncertain by about 11-15%, hence the 1.2 margin. Design effects with 22
sessions are noisy (CFRBot vs EquityBot, two bots without memory, shows 1.71 by chance or by session-level
heterogeneity; it does not enter the analysis, which is per session anyway).

### Budget

Hands x pilot ms/hand: about 81 min at the pilot rate on 22 processes (primary 14.5 min; BayesBot vs
AlwaysRaise 24.8 min is the largest row); with Phase 5's 1.6x margin for sustained throttling, plan for
about **130 min** (worst case 2.5 h). Memory (pilot, measured with `GetProcessMemoryInfo`): peak worker
working set 144 MB, peak worker private bytes 129 MB; parent 228 MB working set / 245 MB private (it holds
the hand logs of one pairing at a time; the official 200-session pairing should take it to about 0.5 GB).
Total about 3.5 GB for 22 workers. Earlier jobs committed ~800 MB per process at `import numpy` (OpenBLAS
buffers for 24 threads); the script sets `OPENBLAS_NUM_THREADS=1` (same numbers, checked).

## Results (official run)

(empty: nothing below the marker until the official run)

<!-- OFFICIAL RESULTS BELOW: written by examples/run_phase6.py, do not edit by hand -->

- date 2026-11-17; commit d8f494f; master seed 202611186; 22 processes; run time 863 s
- model: CFRBot(model=latest.npz @ 13,200,000 it, commit 902376d, abstraction 1cad42d6a4be); sha256 4539e9c7adbc62ace90b989ea257c1ce220639f18fd8a179f8d03b712ac024f3

**Primary test: CFRBot vs BayesBot = +68.2 bb/100, 95% CI [+58.0, +78.3], SE 5.2, SD/session 65.6, p = 2.02e-39, 160 sessions = 80,000 hands. Pre-registered verdict: beats.**

| pairing | sessions | hands | bb/100 | 95% CI | SE | SD/session | p | design effect | learning gain (200, 250) - (0, 50) [95% CI] |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| CFRBot vs BayesBot | 160 | 80,000 | +68.2 | [+58.0, +78.3] | 5.2 | 65.6 | 2.02e-39 | 0.99 | +23.7 [-5.0, +52.5] |
| CFRBot vs EquityBot | 50 | 25,000 | +57.0 | [+39.8, +74.3] | 8.8 | 62.2 | 8.98e-11 | 0.85 | -3.0 [-63.0, +57.0] |
| CFRBot vs PotOddsBot | 50 | 25,000 | +60.3 | [+41.9, +78.8] | 9.4 | 66.6 | 1.5e-10 | 0.82 | +26.4 [-44.9, +97.7] |
| CFRBot vs CallStationBot | 120 | 60,000 | +223.4 | [+210.7, +236.1] | 6.5 | 71.2 | 3.59e-259 | 0.89 | +3.6 [-41.4, +48.6] |
| CFRBot vs AlwaysRaiseBot | 200 | 100,000 | +175.6 | [+155.2, +196.0] | 10.4 | 147.3 | 8.46e-64 | 1.03 | -20.2 [-84.9, +44.4] |
| CFRBot vs RandomBot | 130 | 65,000 | +217.5 | [+206.0, +228.9] | 5.8 | 66.7 | 1.29e-302 | 0.88 | -7.4 [-47.5, +32.7] |
| BayesBot vs EquityBot | 50 | 25,000 | +57.4 | [+36.1, +78.6] | 10.8 | 76.7 | 1.21e-07 | 0.97 | -7.5 [-73.8, +58.8] |
| BayesBot vs PotOddsBot | 50 | 25,000 | +392.5 | [+368.0, +417.1] | 12.5 | 88.4 | 3.47e-216 | 0.65 | -3.9 [-115.3, +107.5] |
| BayesBot vs CallStationBot | 120 | 60,000 | +2224.1 | [+2196.2, +2252.1] | 14.3 | 156.1 | 0 | 0.95 | +1.0 [-89.2, +91.1] |
| BayesBot vs AlwaysRaiseBot | 200 | 100,000 | +1540.2 | [+1516.2, +1564.3] | 12.3 | 173.7 | 0 | 0.94 | +35.3 [-47.4, +118.1] |
| BayesBot vs RandomBot | 130 | 65,000 | +803.4 | [+770.7, +836.1] | 16.7 | 190.1 | 0 | 1.69 | +341.7 [+255.1, +428.4] |

### CFRBot minus BayesBot against the same opponent (paired by session: same deals)

| opponent | difference bb/100 | 95% CI | SE | SD/session | p |
|---|---:|---|---:|---:|---:|
| Equity | -0.4 | [-28.3, +27.6] | 14.3 | 100.8 | 0.979 |
| PotOdds | -332.2 | [-363.6, -300.9] | 16.0 | 113.1 | 8.14e-96 |
| CallStation | -2000.7 | [-2030.8, -1970.6] | 15.4 | 168.2 | 0 |
| AlwaysRaise | -1364.6 | [-1390.7, -1338.5] | 13.3 | 188.4 | 0 |
| Random | -586.0 | [-618.7, -553.2] | 16.7 | 190.7 | 6.91e-269 |

### Breakdown of every pairing (position: CI across sessions; street: contribution to A's bb/100 by the street the hand ended on, CI across sessions, descriptive)

- **CFRBot vs BayesBot**: button +76.7 [+60.4, +93.0], big blind +59.6 [+42.5, +76.8]; preflop 22% of hands -4.1 [-5.1, -3.2], flop 30% of hands -26.0 [-27.4, -24.5], turn 16% of hands -9.4 [-11.0, -7.9], river 12% of hands -4.6 [-6.1, -3.1], showdown 20% of hands +112.3 [+102.1, +122.6]
  - A fold-to-bet: preflop 16% of 58,244, flop 65% of 24,599, turn 62% of 9,342, river 68% of 5,407
  - leak: CFRBot (A) folds 65% to flop bets (24,599 times faced): very exploitable by betting
  - leak: CFRBot (A) folds 62% to turn bets (9,342 times faced): very exploitable by betting
  - leak: CFRBot (A) folds 68% to river bets (5,407 times faced): very exploitable by betting
  - leak: BayesBot (B) folds 71% to flop bets (11,328 times faced): very exploitable by betting
  - leak: BayesBot (B) folds 81% to turn bets (8,821 times faced): very exploitable by betting
  - leak: BayesBot (B) folds 81% to river bets (7,230 times faced): very exploitable by betting
- **CFRBot vs EquityBot**: button +54.6 [+22.1, +87.1], big blind +59.4 [+31.5, +87.3]; preflop 16% of hands -3.4 [-4.2, -2.6], flop 28% of hands -26.6 [-28.8, -24.4], turn 18% of hands -23.4 [-26.3, -20.5], river 13% of hands -16.2 [-20.1, -12.2], showdown 24% of hands +126.6 [+110.8, +142.5]
  - A fold-to-bet: preflop 12% of 17,149, flop 58% of 7,716, turn 62% of 3,658, river 67% of 2,172
  - leak: CFRBot (A) folds 58% to flop bets (7,716 times faced): very exploitable by betting
  - leak: CFRBot (A) folds 62% to turn bets (3,658 times faced): very exploitable by betting
  - leak: CFRBot (A) folds 67% to river bets (2,172 times faced): very exploitable by betting
  - leak: EquityBot (B) folds 62% to flop bets (4,061 times faced): very exploitable by betting
  - leak: EquityBot (B) folds 72% to turn bets (3,220 times faced): very exploitable by betting
  - leak: EquityBot (B) folds 73% to river bets (2,610 times faced): very exploitable by betting
- **CFRBot vs PotOddsBot**: button +25.5 [-9.0, +60.0], big blind +95.2 [+65.1, +125.2]; preflop 5% of hands -3.9 [-4.6, -3.3], flop 19% of hands -24.1 [-26.1, -22.1], turn 16% of hands -40.2 [-44.4, -36.0], river 16% of hands -54.5 [-61.1, -48.0], showdown 44% of hands +183.0 [+165.5, +200.6]
  - A fold-to-bet: preflop 7% of 13,900, flop 45% of 7,164, turn 44% of 5,152, river 48% of 4,314
- **CFRBot vs CallStationBot**: button +266.9 [+242.8, +291.1], big blind +179.9 [+163.4, +196.4]; preflop 2% of hands -0.9 [-1.0, -0.8], flop 0% of hands +0.0 [+0.0, +0.0], turn 0% of hands +0.0 [+0.0, +0.0], river 0% of hands +0.0 [+0.0, +0.0], showdown 98% of hands +224.3 [+211.6, +237.0]
  - A fold-to-bet: preflop 4% of 30,000
  - leak: CallStationBot (B) never folds to preflop bets (55,046 times faced): bluffing it is pointless, value-bet it
  - leak: CallStationBot (B) never folds to flop bets (14,819 times faced): bluffing it is pointless, value-bet it
  - leak: CallStationBot (B) never folds to turn bets (18,324 times faced): bluffing it is pointless, value-bet it
  - leak: CallStationBot (B) never folds to river bets (20,877 times faced): bluffing it is pointless, value-bet it
- **CFRBot vs AlwaysRaiseBot**: button +161.9 [+129.1, +194.7], big blind +189.3 [+161.7, +216.9]; preflop 49% of hands -130.2 [-132.2, -128.2], flop 25% of hands -172.7 [-176.1, -169.2], turn 10% of hands -153.2 [-156.8, -149.5], river 3% of hands -93.9 [-97.1, -90.7], showdown 12% of hands +725.5 [+707.0, +744.0]
  - A fold-to-bet: preflop 31% of 160,614, flop 52% of 48,985, turn 50% of 20,161, river 47% of 7,141
  - leak: CFRBot (A) folds 52% to flop bets (48,985 times faced): very exploitable by betting
  - leak: CFRBot (A) folds 50% to turn bets (20,161 times faced): very exploitable by betting
  - leak: AlwaysRaiseBot (B) never folds to preflop bets (88,510 times faced): bluffing it is pointless, value-bet it
  - leak: AlwaysRaiseBot (B) never folds to flop bets (3,706 times faced): bluffing it is pointless, value-bet it
  - leak: AlwaysRaiseBot (B) never folds to turn bets (2,815 times faced): bluffing it is pointless, value-bet it
  - leak: AlwaysRaiseBot (B) never folds to river bets (1,168 times faced): bluffing it is pointless, value-bet it
- **CFRBot vs RandomBot**: button +238.6 [+220.3, +257.0], big blind +196.3 [+178.3, +214.3]; preflop 67% of hands -18.7 [-20.5, -16.8], flop 21% of hands +53.4 [+48.7, +58.1], turn 7% of hands +58.5 [+53.8, +63.2], river 3% of hands +52.2 [+47.1, +57.3], showdown 3% of hands +72.0 [+63.7, +80.3]
  - A fold-to-bet: preflop 37% of 55,386, flop 77% of 7,572, turn 65% of 2,828, river 55% of 1,115
  - leak: CFRBot (A) folds 77% to flop bets (7,572 times faced): very exploitable by betting
  - leak: CFRBot (A) folds 65% to turn bets (2,828 times faced): very exploitable by betting
  - leak: CFRBot (A) folds 55% to river bets (1,115 times faced): very exploitable by betting
  - leak: RandomBot (B) folded 15,733 times when checking was free (pure loss: it gives up its equity in the pot for nothing)
- **BayesBot vs EquityBot**: button +43.5 [-0.2, +87.3], big blind +71.2 [+20.8, +121.7]; preflop 9% of hands -2.6 [-3.0, -2.3], flop 36% of hands -2.8 [-3.9, -1.6], turn 15% of hands -25.2 [-28.9, -21.5], river 9% of hands -14.6 [-17.5, -11.6], showdown 32% of hands +102.5 [+81.3, +123.7]
  - A fold-to-bet: preflop 8% of 17,933, flop 55% of 8,431, turn 67% of 3,145, river 75% of 1,630
  - leak: BayesBot (A) folds 55% to flop bets (8,431 times faced): very exploitable by betting
  - leak: BayesBot (A) folds 67% to turn bets (3,145 times faced): very exploitable by betting
  - leak: BayesBot (A) folds 75% to river bets (1,630 times faced): very exploitable by betting
  - leak: EquityBot (B) folds 53% to flop bets (8,026 times faced): very exploitable by betting
  - leak: EquityBot (B) folds 60% to turn bets (2,715 times faced): very exploitable by betting
  - leak: EquityBot (B) folds 68% to river bets (1,481 times faced): very exploitable by betting
- **BayesBot vs PotOddsBot**: button +408.2 [+360.3, +456.1], big blind +376.9 [+317.5, +436.3]; preflop 5% of hands -3.8 [-4.2, -3.5], flop 25% of hands +0.3 [-1.3, +2.0], turn 15% of hands -24.8 [-28.7, -20.9], river 12% of hands -31.7 [-37.3, -26.0], showdown 43% of hands +452.5 [+427.4, +477.7]
  - A fold-to-bet: preflop 7% of 14,186, flop 52% of 6,510, turn 56% of 3,746, river 66% of 2,543
  - leak: BayesBot (A) folds 52% to flop bets (6,510 times faced): very exploitable by betting
  - leak: BayesBot (A) folds 56% to turn bets (3,746 times faced): very exploitable by betting
  - leak: BayesBot (A) folds 66% to river bets (2,543 times faced): very exploitable by betting
- **BayesBot vs CallStationBot**: button +2212.1 [+2137.0, +2287.2], big blind +2236.2 [+2162.2, +2310.1]; preflop 0% of hands +0.0 [+0.0, +0.0], flop 0% of hands +0.0 [+0.0, +0.0], turn 0% of hands +0.0 [+0.0, +0.0], river 0% of hands +0.0 [+0.0, +0.0], showdown 100% of hands +2224.1 [+2196.2, +2252.1]
  - A fold-to-bet: preflop 0% of 30,000
  - leak: BayesBot (A) never folds to preflop bets (30,000 times faced): bluffing it is pointless, value-bet it
  - leak: CallStationBot (B) never folds to preflop bets (50,537 times faced): bluffing it is pointless, value-bet it
  - leak: CallStationBot (B) never folds to flop bets (23,410 times faced): bluffing it is pointless, value-bet it
  - leak: CallStationBot (B) never folds to turn bets (6,076 times faced): bluffing it is pointless, value-bet it
  - leak: CallStationBot (B) never folds to river bets (5,240 times faced): bluffing it is pointless, value-bet it
- **BayesBot vs AlwaysRaiseBot**: button +1534.8 [+1476.5, +1593.1], big blind +1545.6 [+1483.4, +1607.9]; preflop 4% of hands -4.1 [-4.2, -3.9], flop 25% of hands -74.7 [-75.5, -73.9], turn 9% of hands -84.5 [-86.2, -82.8], river 7% of hands -179.2 [-183.8, -174.6], showdown 56% of hands +1882.8 [+1858.1, +1907.4]
  - A fold-to-bet: preflop 3% of 156,680, flop 29% of 86,954, turn 24% of 38,381, river 32% of 20,430
  - leak: AlwaysRaiseBot (B) never folds to preflop bets (71,313 times faced): bluffing it is pointless, value-bet it
  - leak: AlwaysRaiseBot (B) never folds to flop bets (33,545 times faced): bluffing it is pointless, value-bet it
  - leak: AlwaysRaiseBot (B) never folds to turn bets (8,064 times faced): bluffing it is pointless, value-bet it
  - leak: AlwaysRaiseBot (B) never folds to river bets (4,296 times faced): bluffing it is pointless, value-bet it
- **BayesBot vs RandomBot**: button +838.6 [+790.1, +887.2], big blind +768.2 [+722.8, +813.6]; preflop 62% of hands +202.6 [+182.0, +223.2], flop 19% of hands +181.7 [+172.1, +191.4], turn 5% of hands +69.5 [+63.7, +75.3], river 2% of hands +23.1 [+19.5, +26.6], showdown 12% of hands +326.5 [+303.8, +349.2]
  - A fold-to-bet: preflop 27% of 54,983, flop 56% of 6,324, turn 62% of 1,836, river 62% of 565
  - leak: BayesBot (A) folds 56% to flop bets (6,324 times faced): very exploitable by betting
  - leak: BayesBot (A) folds 62% to turn bets (1,836 times faced): very exploitable by betting
  - leak: BayesBot (A) folds 62% to river bets (565 times faced): very exploitable by betting
  - leak: RandomBot (B) folded 13,959 times when checking was free (pure loss: it gives up its equity in the pot for nothing)

peak memory: worker working set max 145 MB, worker private max 130 MB; parent 692 MB working set
