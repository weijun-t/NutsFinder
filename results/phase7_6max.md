# Phase 7: 6-max evaluation (the multiway tier ladder against a fixed mixed field)

## Pre-registration (written and committed BEFORE the official run)

Status: FINAL from the commit that adds this file, before `examples/run_6max.py` is run officially.
Everything in this section is fixed from the moment it is committed. Changing it afterwards, re-running with
another seed, or extending/shortening the run would be optional stopping.

### The question

PLAN.md's ladder rule says a tier only counts if it beats the previous tier. Heads-up, BayesBot beat EquityBot by
+61.0 bb/100 (Phase 5) and CFRBot beat BayesBot by +68.2 (Phase 6). At a 6-max table: **does MultiBayesBot (tier 3)
beat MultiEquityBot (tier 2) when it sits in the same seat, on the same deals, against the same fixed field?**
There is one gating test. Everything else is descriptive.

**The result depends on the field (docs/phase7_multiway_research.md section 3).** A multiway win rate belongs to the
(bot, field) pair. Whatever comes out holds for this field, and for the 5 cyclic seatings of it, only. It is not
pooled with other fields, and no transitive claim follows from it.

### Table and field

- 6 seats, 10,000 chips (100bb) reset every hand, blinds 50/100. This is the committed N-player engine
  (`src/nutsfinder/engine/multiway/`, `play_hand_multi`).
- **Field `mixed` = [MultiEquityBot, MultiPotOddsBot, CallStation, AlwaysRaise, Random]**, all committed bots with
  default parameters. Reasons:
  - It matches what the tier claim is about. BayesBot's tier exists to exploit opponents. A field with two
    competent non-learning bots (tier 2 and tier 1) and three distinct leak types (a passive station, a maniac,
    noise) gives it something to learn. A soft low-stakes table looks like this: a few regulars and several
    recreational players.
  - It reuses the project's baseline set (the heads-up tier tests used the same five opponent types).
  - It is the mixed field from the research note's template (F3), with the cyclic order fixed and stated.
- **Seating order:** the base order above, clockwise from the test seat. Session s uses it rotated by `s % 5`, so
  the 5 cyclic arrangements are used equally often (N is a multiple of 10). The estimand averages over those 5
  arrangements, not over all 5! = 120 orders. This is a deliberate simplification.
- **Not run: the "tough field" of 5 × MultiEquityBot.** It costs about 650 ms per hand on one core against about
  230 ms for the mixed field (timing measured before the pilot, no outcomes looked at), so it would roughly
  triple the run. It was dropped before the pilot for budget reasons. It is listed under "what is not tested".

### Rotation duplicate, sessions, seeds

- **Block** = one deal: a deal seed plus one reseed per seat. The block is played in 6 **rooms**. In room r the test
  bot sits in seat r and the field fills seats r+1..r+5 in the session's order. The hole cards per seat, the board,
  the button (`block % 6`) and the per-seat reseeds are identical in all rooms. So on every deal, every bot,
  test bot and field bots alike, sits in every seat exactly once (asserted per session from the recorded layouts).
- **Reseeding:** every bot is reseeded before every hand from (session, block, seat). Field bots therefore draw the
  same random numbers in both arms, unless the test bot's actions change what they are asked.
- **Session (the unit)** = 250 blocks × 6 rooms = **1,500 test-seat hands**. Every room is a fresh table with fresh
  bot objects, so MultiBayesBot learns its 5 opponents over 250 hands. That is the same experience each heads-up
  BayesBot instance had in Phases 5 and 6 (250 hands per instance in 500-hand sessions). The estimand is
  "bb/100 over the first 250 hands at a fresh table, averaged over the 6 seats".
- **Arms = the test bot**: MultiBayesBot and MultiEquityBot. Both use the same session seeds (common random
  numbers): the same deals, the same field, the same seating, the same reseeds. The pairing is checked through a
  per-room digest of every seed.
- **Seeds:** the field seed is the first 8 bytes of SHA-256 of `"nutsfinder/6max|<master>|<field>"`, and session s
  uses `session_seed(field_seed, s)` (`nutsfinder/eval/sessions.py`, unchanged). **Official master seed
  `202611241`.** The pilot and smoke seed is `75025`. Both were grepped against the whole repo and the list of
  used seeds (999, 555, 556, 777, 4242, 20261115-17, 20261116+k, 202611186, 202611188, 202611231, 7, 61, 27182818,
  16180339, 31415926, 17320508, 14142135, 46368, 20261124 (`run_multiway_sanity.py`), 20251123, 987654321): no
  occurrence.

### Arms that are not run, and why

- **MultiCFRBot.** It uses the heads-up blueprint only when a hand is heads-up preflop between some player X and
  the big blind after everyone else folded (`src/nutsfinder/bots/multiway/cfr_bot.py`). CallStation and AlwaysRaise
  never fold, and both sit in every hand of this field, so that spot cannot occur. In this field MultiCFRBot *is*
  MultiEquityBot with a different seed derivation. The smoke run confirmed it: 0 blueprint decisions out of 269.
  CFR − Equity would be zero by construction, apart from Monte Carlo noise, and would cost a quarter of the run.
  CFR needs a field in which the blueprint can act. That is not this test.
- **MultiPotOddsBot as a test bot (secondary Equity − PotOdds, the tier 1 → 2 step).** It was in the pilot. At
  the pre-pilot N = 150 it added about 1.1 h (5.0 h instead of 3.9 h at the pilot rate); at the final N = 110,
  about 0.8 h. A scope rule fixed before the pilot (in `run_6max.py`) said: if the run exceeds the 3 h budget, drop
  the PotOdds arm; never drop a primary arm; never change the target edge; and if the primary alone does not fit,
  say so and let the owner decide. The PotOdds drop was applied. The pilot's Equity − PotOdds number is disclosed
  below, and its sign was visible when the rule was applied. The rule removes a comparison rather than choosing
  one. The rest of the rule's history is under "Sample size, power, budget".

### Primary test (the only gating test)

- **D = MultiBayesBot's session bb/100 minus MultiEquityBot's**, same session, same seats, same deals, same field.
  The mean is taken across sessions, with a normal 95% CI across sessions (two-sided, alpha 0.05).
- Rule (`tier_verdict`):
  - **"beats"** iff the CI lies entirely above 0. Tier 3 then counts in this field.
  - **"loses"** iff the CI lies entirely below 0.
  - Otherwise **"no detectable edge"**, never "no edge". The output then states how many sessions would be needed
    to detect the observed D, and the CI shows which edges are ruled out.
- Integrity conditions must hold, or the output is "INVALID" and no claim is made:
  - Rotation: every bot sits in every seat once per deal.
  - Pairing: in every arm and room, the seed digest equals the one recomputed from (master seed, s), and the
    dealt cards (seed, button, hole cards, board) and the field seating are identical.
  - Records: every record is filed under its own (field, arm, session, room).

### Secondary / descriptive (not gating)

- Each arm's absolute bb/100 against the field, with SD per session, SD per hand and the per-hand-equivalent SD.
- Breakdowns, with CIs across sessions, per arm and for the paired difference:
  - by position (UTG, HJ, CO, BTN, SB, BB; each has exactly 250 hands per session);
  - by the test bot's **exit street** (the street it folded on, or "showdown" if it never folded).

  The street the *hand* ended on is useless here. CallStation and AlwaysRaise never fold, so every hand reaches a
  showdown. The smoke run showed this, and the breakdown was changed to the exit street before the pilot.
- Pooled fold-to-bet rates per street, VPIP/PFR, the leak list (the rules of `nutsfinder.eval.breakdown.leaks`),
  each field bot's bb/100 (who pays), and the share of hands with the same net in both arms.
- **Multiple testing:** one primary test. The other 45 CIs are uncorrected and descriptive: per arm, 1 overall,
  6 positions, 5 exit streets and 5 field bots (2 arms); for D, 6 positions and 5 exit streets. About 1 in 20
  will exclude its true value by chance.

### Sample size, power, budget

- Design SD = pilot SD × 1.2. 16 pilot sessions give an SD that is uncertain by about ±18%, and the margin covers
  about one standard error. Settings: 80% power, two-sided alpha 0.05, sessions rounded up to a multiple of 10,
  floor 50.
- **Target edge: 60 bb/100**, one heads-up tier step (+61 / +68). **Disclosure: this was changed AFTER the pilot,
  against the pre-pilot rule.**
  - The pre-pilot target was 50 bb/100. Even with the PotOdds arm dropped, the primary alone did not fit: N = 150,
    3.9 h at the pilot rate, 6.3 h with the throttling margin, against the 3 h budget. That triggered the rule's
    last branch: "say so and let the owner decide".
  - The project owner then **overrode the rule's "never change the target edge" clause** and chose 60, for run time.
    The pilot's primary estimate was visible at the time: D = **−8.0 bb/100, 95% CI [−96.5, +80.5]**.
  - A larger target only lowers the power to call "beats", so the change does not favour the expected result
    (tier 3 beats tier 2). The decision rule (CI vs 0) is unchanged; the target only sets N.
  - Even at 60 the design is **still over budget** by the script's own definition (pilot rate × 1.6 = 4.6 h > 3 h).
    The owner accepted that.

  | quantity | value |
  |---|---|
  | pilot SD of D (bb/100 per 1,500-hand session) | 180.7 |
  | design SD | 216.8 |
  | required N at 60 | 103 → **N = 110 sessions per arm (fixed)** (at 50 it would be 148 → 150) |
  | MDE at N = 110 (80% power) | 57.9 (design SD); 48.3 (pilot SD) |
  | power at 60 / at the pre-pilot 50 / at 25 | 83% / **68%** / 23% (design SD); 94% / 83% / 31% (pilot SD) |
  | hands | 110 × 1,500 × 2 arms = 330,000 test-seat hands |

- **What this cannot detect:** a true edge of 50 bb/100 is missed about one time in three, and one of 25 about
  three times in four. A "no detectable edge" result rules out only edges beyond the CI bound.
- **Run time:** 173 min (2.9 h) at the pilot rate, which was measured with all 22 processes busy and the pool
  about 98% utilised. With the Phase 5 throttling margin of 1.6 it is up to 277 min (4.6 h). The budget is defined as
  pilot rate × 1.6 ≤ 3 h, so this is **over budget**; the owner accepted it. The run is crash-safe and resumable
  (`--resume-partial`). The PotOdds arm would add about 0.8 h and stays out.

### Rules

- Fixed hand count and no peeking. The official run is played once, to the end, on seed 202611241. A crash is
  finished with `--resume-partial` (rooms are deterministic; equality with an uninterrupted run is tested), never
  restarted.
- If a crash happens after the verdict is spliced below the marker but before the summary JSON or plots are
  written, the verdict stands. Recover the outputs with
  `.venv\Scripts\python examples\run_6max.py --rebuild-outputs results\phase7_6max_official_partial.jsonl`.
  It re-analyses the complete log, plays nothing and never touches this file (tested).
- The script refuses to run if:
  - the tree is dirty (untracked files included);
  - `--allow-dirty` is passed;
  - official result files exist;
  - a partial log exists without `--resume-partial`;
  - anything is already below the marker.
- Official command (after committing this file and the files listed below):
  `.venv\Scripts\python examples\run_6max.py`

### Pilot (seed 75025, NOT OFFICIAL: used only to choose N and the budget)

Every number in this subsection comes from seed 75025 and is **NOT OFFICIAL**. It is not evidence and must never
be quoted as the result.
- Run: 2026-11-24, `examples/run_6max.py --pilot`, commit 7cb9678 + the uncommitted script, arms Bayes, Equity and
  PotOdds, 16 sessions × 1,500 test-seat hands per arm, 22 processes, 1,953 s.
- Files: the committed PNGs `results/2026-11-24_6max_pilot_seed75025_*.png` and the log
  `results/phase7_6max_pilot_run.log`. The JSON/JSONL files are gitignored (`results/*.json`, `results/*.jsonl`) and
  stay local. Their SHA-256:
  - `2026-11-24_6max_pilot_seed75025_summary.json`
    `bdba30b20959d04ba3974aafcbf6d8d24b3e4fe18310a4d6037d7908de3aecf3`
  - `..._mixed_Bayes_minus_Equity.json` `2b2796664947bc6246908cadc3fff00a224da583eb3655a2b33a13562ec09a8d`
  - `..._mixed_Bayes_minus_PotOdds.json` `577ea8e2c08a034770c607e859cc24bd5af24a826ae384e451207c16b469bd53`
  - `..._mixed_Equity_minus_PotOdds.json` `ef0980a08a0e82bff8b88473a5f1ad8060ad356d5dedfc15e50cf46dfa4c616e`
  - `..._mixed_Bayes_vs_field_sessions.json` `93e7fc17e8d83dd132e6c93ae8c4846350a92844eff17b05364d273fbb854033`
  - `..._mixed_Equity_vs_field_sessions.json` `8eeef778ae3e433fdf7215a90451e7855576f3bf4cf25d3770a666a4f865d97a`
  - `..._mixed_PotOdds_vs_field_sessions.json` `02a1ea032e840ff65aeb74002714ae10d780dce0d9d3db1cc68962ceac2fe855`
  - `phase7_6max_pilot_seed75025_partial.jsonl` `bcbfeba366cd203d64380a357949d00928dce98832b8956cece7f5b5399904df`
- The pilot predates the card-digest integrity check. Its records have no `cards_digest`, so re-analysing the pilot
  log with the current script would report that check as failed. Its seed and layout checks passed.

| pilot, NOT OFFICIAL | bb/100 | 95% CI | SD/session |
|---|---:|---|---:|
| D = Bayes − Equity (primary quantity) | −8.0 | [−96.5, +80.5] | 180.7 |
| Equity − PotOdds | −117.1 | [−205.2, −29.1] | 179.6 |
| Bayes − PotOdds | −125.1 | [−209.3, −40.9] | 171.9 |
| Bayes vs field | +1193.0 | [+1084.8, +1301.3] | 220.9 |
| Equity vs field | +1201.0 | [+1104.4, +1297.6] | 197.1 |
| PotOdds vs field | +1318.2 | [+1223.8, +1412.5] | 192.5 |

- **Variance.** SD per hand is about 104-117 bb. The field has two players who never fold and a random raiser, so
  most hands go to a showdown with big pots.
- **Pairing.** The correlation between the Bayes and Equity session results is about 0.63, and 81% of hands have
  the same net in both arms. The rest, mostly large all-in pots, carry the variance of D.
- **Integrity.** It passed on all 16 sessions.
- **Timing** (pool-wall ms per test-seat hand, 22 processes): Bayes 31.2, Equity 31.7, PotOdds 17.2. Peak worker
  working set was 139 MB.
- The verdict lines the pilot printed are not results.

### Other disclosures

- **Smoke runs.** There were two 2-session × 72-hand smoke runs on the pilot seed, both seen before the pilot:
  - the first had a CFR arm;
  - the second ran without it, with the exit-street breakdown, and its files are in `results/`.

  They led to two pre-pilot changes. The CFR arm was dropped, on the structural argument above, which the smoke's
  0 blueprint decisions only confirmed. The street breakdown became the exit-street breakdown. Their win-rate
  numbers (2 sessions) were visible and are meaningless.
- **The tough field was dropped before the pilot**, from per-hand timing only.
- **Choices fixed before the pilot:** the field, the seating rotation, the session length, the target edge (50, later replaced by 60 after the pilot),
  the 1.2 SD margin, the 50-session floor and the scope rule.
- **Fixed after the pilot:** the target edge 60 (replacing 50; chosen by the owner for run time, see above), N = 110
  (from the pilot SD), and dropping the PotOdds arm (by the pre-pilot rule).
- **Known limitations:**
  - The field is one of many, and the result belongs to it.
  - Field bots' Monte Carlo draws inside a hand can differ between arms once the test bot acts differently,
    which weakens the pairing a little.
  - The 250-hand learning window is short for a 5-opponent model split by heads-up/multiway context.
  - The 5 cyclic orders are not all 120 orders.
  - Positions are balanced by the rotation, but the field's position relative to the test bot is fixed within a
    session.

<!-- OFFICIAL RESULTS BELOW: written by examples/run_6max.py, do not edit by hand -->
