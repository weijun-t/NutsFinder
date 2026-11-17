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
