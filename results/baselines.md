# Baseline results

- date: 2026-11-16
- master seed: 20261115 (same deals for every pairing)
- n: 10,000 duplicate pairs = 20,000 hands per pairing, fixed in advance
- duplicate: each deal played twice, bots swapped between seats (cards and button swap with them)
- commit: f49b969
- run time: 62 s

Win rate is for bot A (first named), in bb/100 hands. 95% CI = mean +/- 1.96 SE
(normal approximation, per-pair results treated as i.i.d.). p = two-sided z-test vs 0.

## Duplicate matches

| pairing (A vs B) | bb/100 | 95% CI (+/-) | SE | SD per pair (bb) | p-value | significant |
|---|---:|---:|---:|---:|---:|---|
| Random vs Random | -11.0 | 62.1 | 31.7 | 63.4 | 0.729 | no |
| Random vs CallStation | -1890.1 | 61.6 | 31.5 | 62.9 | < 1e-300 | yes |
| Random vs AlwaysRaise | -1646.0 | 57.4 | 29.3 | 58.5 | < 1e-300 | yes |
| CallStation vs CallStation | +0.0 | 0.0 | 0.0 | 0.0 | n/a (no variation) | exact 0 (no variation) |
| CallStation vs AlwaysRaise | +0.0 | 0.0 | 0.0 | 0.0 | n/a (no variation) | exact 0 (no variation) |
| AlwaysRaise vs AlwaysRaise | +0.0 | 0.0 | 0.0 | 0.0 | n/a (no variation) | exact 0 (no variation) |

## Variance reduction: plain vs duplicate (same number of hands)

Plain: 20,000 independent hands, button alternating. Hands-equivalent = (SE plain / SE duplicate)^2: how many times more plain hands
would be needed for the same precision.

| pairing | plain bb/100 | plain SE | duplicate SE | SE ratio | hands-equivalent |
|---|---:|---:|---:|---:|---:|
| Random vs Random | -3.6 | 30.8 | 31.7 | 0.97 | 0.9x |
| Random vs CallStation | -1906.9 | 32.4 | 31.5 | 1.03 | 1.1x |
| Random vs AlwaysRaise | -1662.1 | 39.6 | 29.3 | 1.35 | 1.8x |
| CallStation vs CallStation | -0.5 | 0.7 | 0.0 | inf (duplicate SE = 0) | inf |
| CallStation vs AlwaysRaise | -43.7 | 56.1 | 0.0 | inf (duplicate SE = 0) | inf |
| AlwaysRaise vs AlwaysRaise | -54.0 | 69.3 | 0.0 | inf (duplicate SE = 0) | inf |

## Verdicts and breakdowns (duplicate matches)

### Random vs Random

- verdict: no detectable edge: the 95% CI rules out edges beyond +/-73.1 bb/100; underpowered for the observed -10.97 bb/100: detecting it with 80% power would need about 1,309,820 hands
- A on button: +191.1 bb/100 (SE 43.6, per hand, no duplicate cancellation)
- A on big_blind: -213.0 bb/100 (SE 43.2, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them): preflop 67% (-8.5), flop 19% (+14.8), turn 5% (-18.7), river 2% (-0.6), showdown 7% (+2.0)
- leak: RandomBot (A) folded 2,783 times when checking was free (pure loss: it gives up its equity in the pot for nothing)
- leak: RandomBot (B) folded 2,813 times when checking was free (pure loss: it gives up its equity in the pot for nothing)

### Random vs CallStation

- verdict: significant edge for CallStationBot (p < 1e-300)
- A on button: -1846.8 bb/100 (SE 45.7, per hand, no duplicate cancellation)
- A on big_blind: -1933.5 bb/100 (SE 45.3, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them): preflop 34% (-25.1), flop 22% (-593.7), turn 15% (-672.5), river 10% (-598.8), showdown 19% (+0.0)
- leak: RandomBot (A) folded 12,755 times when checking was free (pure loss: it gives up its equity in the pot for nothing)
- leak: CallStationBot (B) never folds to preflop bets (16,824 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to flop bets (4,314 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to turn bets (2,984 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to river bets (1,914 times faced): bluffing it is pointless, value-bet it

### Random vs AlwaysRaise

- verdict: significant edge for AlwaysRaiseBot (p < 1e-300)
- A on button: -1530.3 bb/100 (SE 56.3, per hand, no duplicate cancellation)
- A on big_blind: -1761.7 bb/100 (SE 55.3, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them): preflop 58% (-1094.5), flop 15% (-421.1), turn 3% (-110.3), river 0% (-12.6), showdown 24% (-7.5)
- leak: RandomBot (A) folded 1,391 times when checking was free (pure loss: it gives up its equity in the pot for nothing)
- leak: AlwaysRaiseBot (B) never folds to preflop bets (18,663 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to flop bets (2,123 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to turn bets (388 times faced): bluffing it is pointless, value-bet it

### CallStation vs CallStation

- verdict: no variation: every pair gave exactly +0 bb, so there is no CI or p-value. This is not a statistical estimate; it is only meaningful if the setup forces it (e.g. deterministic bots whose duplicate pairs cancel exactly)
- A on button: +0.7 bb/100 (SE 1.0, per hand, no duplicate cancellation)
- A on big_blind: -0.7 bb/100 (SE 1.0, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them): showdown 100% (+0.0)
- leak: CallStationBot (A) never folds to preflop bets (10,000 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to preflop bets (10,000 times faced): bluffing it is pointless, value-bet it

### CallStation vs AlwaysRaise

- verdict: no variation: every pair gave exactly +0 bb, so there is no CI or p-value. This is not a statistical estimate; it is only meaningful if the setup forces it (e.g. deterministic bots whose duplicate pairs cancel exactly)
- A on button: +59.1 bb/100 (SE 79.3, per hand, no duplicate cancellation)
- A on big_blind: -59.1 bb/100 (SE 79.3, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them): showdown 100% (+0.0)
- leak: CallStationBot (A) never folds to preflop bets (30,000 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (A) never folds to flop bets (20,000 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (A) never folds to turn bets (20,000 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (A) never folds to river bets (20,000 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to preflop bets (10,000 times faced): bluffing it is pointless, value-bet it

### AlwaysRaise vs AlwaysRaise

- verdict: no variation: every pair gave exactly +0 bb, so there is no CI or p-value. This is not a statistical estimate; it is only meaningful if the setup forces it (e.g. deterministic bots whose duplicate pairs cancel exactly)
- A on button: +73.0 bb/100 (SE 97.9, per hand, no duplicate cancellation)
- A on big_blind: -73.0 bb/100 (SE 97.9, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them): showdown 100% (+0.0)
- leak: AlwaysRaiseBot (A) never folds to preflop bets (60,000 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to preflop bets (60,000 times faced): bluffing it is pointless, value-bet it
