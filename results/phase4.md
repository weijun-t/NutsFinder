# Phase 4 results: PotOddsBot and EquityBot

- date: 2026-11-16
- master seed: 20261116 (fixed before the run; same deals for every pairing)
- n: 10,000 duplicate pairs = 20,000 hands per pairing, fixed in advance, no early stopping
- duplicate: each deal played twice, bots swapped between seats (cards and button swap with them)
- bot parameters: the committed defaults (PotOddsBot: bet > 60%, raise > 70%, 300 samples; EquityBot: ranges 100/50/30/15%, sizes 1/2 pot, pot, all-in, max_fold 0.1, 400 samples)
- commit: d978824
- run time: 1333 s wall clock (14 processes; 9772 s summed over the 14 jobs). Every job is determined by its pairing and the master seed alone, so parallelism cannot change any number.

## Tuning history (why this seed)

EquityBot's `max_fold` (0.1, chosen from 0, 0.1, 0.2, 0.3, 0.5) and its range widths were tuned on
master seed 777 (1,000 pairs per setting) and checked on seed 4242 as a holdout (1,000 pairs).
Choosing parameters on the data you then report is overfitting, so the official evaluation uses a
fresh seed, 20261116, fixed before this run, with no parameter changes and no re-runs.

Win rate is for bot A (first named), in bb/100 hands. 95% CI = mean +/- 1.96 SE
(normal approximation, per-pair results treated as i.i.d.). p = two-sided z-test vs 0.

## Duplicate matches

| pairing (A vs B) | hands | bb/100 | 95% CI | +/- | SE | SD per pair (bb) | p-value | significant |
|---|---:|---:|---|---:|---:|---:|---:|---|
| PotOdds vs Random | 20,000 | +1235.9 | [+1189.5, +1282.3] | 46.4 | 23.7 | 47.4 | < 1e-300 | yes |
| PotOdds vs CallStation | 20,000 | +256.6 | [+247.8, +265.4] | 8.8 | 4.5 | 9.0 | < 1e-300 | yes |
| PotOdds vs AlwaysRaise | 20,000 | +1670.6 | [+1619.2, +1722.0] | 51.4 | 26.2 | 52.5 | < 1e-300 | yes |
| Equity vs Random | 20,000 | +798.0 | [+758.7, +837.3] | 39.3 | 20.0 | 40.1 | < 1e-300 | yes |
| Equity vs CallStation | 20,000 | +2043.1 | [+1995.9, +2090.3] | 47.2 | 24.1 | 48.2 | < 1e-300 | yes |
| Equity vs AlwaysRaise | 20,000 | +1244.9 | [+1187.1, +1302.7] | 57.8 | 29.5 | 59.0 | < 1e-300 | yes |
| Equity vs PotOdds | 20,000 | +266.8 | [+237.8, +295.8] | 29.0 | 14.8 | 29.6 | 9.63e-73 | yes |

All pairings use the same master seed, so they share deals (common random numbers). Their
estimates are therefore correlated: do not compare two pairings using independent-sample SEs.

## Phase 4 checks

- PotOddsBot beats every baseline significantly: yes
- EquityBot beats every baseline significantly: yes
- tier test (rule 2), EquityBot vs PotOddsBot: +266.8 bb/100, 95% CI [+237.8, +295.8]: EquityBot counts as a new tier

## Variance reduction: plain vs duplicate (same number of hands, same seed)

Plain: 20,000 independent hands, button alternating. Hands-equivalent = (SE plain / SE duplicate)^2: how many times more plain hands
would be needed for the same precision.

| pairing | plain bb/100 | plain 95% CI | plain SE | duplicate SE | SE ratio | hands-equivalent |
|---|---:|---|---:|---:|---:|---:|
| PotOdds vs Random | +1250.8 | [+1200.8, +1300.7] | 25.5 | 23.7 | 1.08 | 1.2x |
| PotOdds vs CallStation | +258.5 | [+246.5, +270.6] | 6.2 | 4.5 | 1.37 | 1.9x |
| PotOdds vs AlwaysRaise | +1658.0 | [+1563.5, +1752.5] | 48.2 | 26.2 | 1.84 | 3.4x |
| Equity vs Random | +782.5 | [+740.8, +824.3] | 21.3 | 20.0 | 1.06 | 1.1x |
| Equity vs CallStation | +2030.2 | [+1957.0, +2103.4] | 37.3 | 24.1 | 1.55 | 2.4x |
| Equity vs AlwaysRaise | +1246.4 | [+1169.6, +1323.1] | 39.2 | 29.5 | 1.33 | 1.8x |
| Equity vs PotOdds | +279.5 | [+225.3, +333.8] | 27.7 | 14.8 | 1.87 | 3.5x |

## Verdicts and breakdowns (duplicate matches)

### PotOdds vs Random

- verdict: significant edge for PotOddsBot (p < 1e-300)
- A on button: +1252.3 bb/100 (SE 34.9, per hand, no duplicate cancellation)
- A on big_blind: +1219.5 bb/100 (SE 36.4, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them; descriptive, no CI): preflop 50% (+38.0), flop 26% (+468.9), turn 11% (+299.9), river 4% (+177.2), showdown 10% (+251.9)

| bot | street | decisions | fold | check | call | bet | raise | all-in | faced bet | fold to bet | free folds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PotOdds | preflop | 20,211 | 3,002 | 2,839 | 13,375 | 0 | 995 | 164 | 16,894 | 18% | 0 |
| PotOdds | flop | 9,999 | 1,169 | 4,569 | 1,514 | 2,065 | 682 | 1,148 | 3,365 | 35% | 0 |
| PotOdds | turn | 4,319 | 433 | 2,060 | 785 | 798 | 243 | 512 | 1,461 | 30% | 0 |
| PotOdds | river | 1,941 | 143 | 923 | 406 | 380 | 89 | 277 | 638 | 22% | 0 |
| Random | preflop | 20,699 | 6,915 | 3,215 | 3,675 | 0 | 6,894 | 89 | 10,995 | 33% | 3,232 |
| Random | flop | 11,215 | 3,951 | 2,852 | 1,047 | 2,809 | 556 | 561 | 2,747 | 42% | 2,807 |
| Random | turn | 4,741 | 1,676 | 1,187 | 417 | 1,244 | 217 | 276 | 1,041 | 39% | 1,269 |
| Random | river | 2,097 | 733 | 532 | 194 | 566 | 72 | 155 | 469 | 43% | 530 |

- PotOdds: river bets/raises 469, of which with < 50% equity vs a random hand (bluff proxy) 0 (0%); folds to a bet 4,747, of which the folded hand would have won or tied on the full board 1,889 (40%); free folds 0
- Random: river bets/raises 638, of which with < 50% equity vs a random hand (bluff proxy) 346 (54%); folds to a bet 5,437, of which the folded hand would have won or tied on the full board 2,208 (41%); free folds 7,838
- leak: RandomBot (B) folded 7,838 times when checking was free (pure loss: it gives up its equity in the pot for nothing)

### PotOdds vs CallStation

- verdict: significant edge for PotOddsBot (p < 1e-300)
- A on button: +239.4 bb/100 (SE 8.3, per hand, no duplicate cancellation)
- A on big_blind: +273.8 bb/100 (SE 9.0, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them; descriptive, no CI): showdown 100% (+256.6)

| bot | street | decisions | fold | check | call | bet | raise | all-in | faced bet | fold to bet | free folds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PotOdds | preflop | 20,000 | 0 | 8,529 | 9,704 | 0 | 1,767 | 0 | 10,000 | 0% | 0 |
| PotOdds | flop | 20,000 | 0 | 13,817 | 0 | 6,183 | 0 | 0 | 0 | - | 0 |
| PotOdds | turn | 20,000 | 0 | 13,159 | 0 | 6,841 | 0 | 0 | 0 | - | 0 |
| PotOdds | river | 20,000 | 0 | 12,099 | 0 | 7,901 | 0 | 0 | 0 | - | 0 |
| CallStation | preflop | 21,471 | 0 | 9,704 | 11,767 | 0 | 0 | 0 | 11,767 | 0% | 0 |
| CallStation | flop | 23,080 | 0 | 16,897 | 6,183 | 0 | 0 | 0 | 6,183 | 0% | 0 |
| CallStation | turn | 23,421 | 0 | 16,580 | 6,841 | 0 | 0 | 0 | 6,841 | 0% | 0 |
| CallStation | river | 23,966 | 0 | 16,065 | 7,901 | 0 | 0 | 0 | 7,901 | 0% | 0 |

- PotOdds: river bets/raises 7,901, of which with < 50% equity vs a random hand (bluff proxy) 0 (0%); folds to a bet 0, of which the folded hand would have won or tied on the full board 0; free folds 0
- CallStation: river bets/raises 0, of which with < 50% equity vs a random hand (bluff proxy) 0; folds to a bet 0, of which the folded hand would have won or tied on the full board 0; free folds 0
- leak: PotOddsBot (A) never folds to preflop bets (10,000 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to preflop bets (11,767 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to flop bets (6,183 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to turn bets (6,841 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to river bets (7,901 times faced): bluffing it is pointless, value-bet it

### PotOdds vs AlwaysRaise

- verdict: significant edge for PotOddsBot (p < 1e-300)
- A on button: +1656.5 bb/100 (SE 66.7, per hand, no duplicate cancellation)
- A on big_blind: +1684.6 bb/100 (SE 69.1, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them; descriptive, no CI): preflop 2% (-1.8), flop 23% (-68.9), turn 10% (-91.8), river 9% (-247.6), showdown 56% (+2080.6)

| bot | street | decisions | fold | check | call | bet | raise | all-in | faced bet | fold to bet | free folds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PotOdds | preflop | 30,828 | 353 | 0 | 29,055 | 0 | 1,420 | 562 | 30,828 | 1% | 0 |
| PotOdds | flop | 31,963 | 4,594 | 6,717 | 12,857 | 2,830 | 4,965 | 3,331 | 22,416 | 20% | 0 |
| PotOdds | turn | 17,453 | 2,040 | 4,239 | 8,683 | 1,352 | 1,139 | 1,804 | 11,862 | 17% | 0 |
| PotOdds | river | 10,458 | 1,822 | 2,649 | 4,587 | 775 | 625 | 1,827 | 7,034 | 26% | 0 |
| AlwaysRaise | preflop | 21,124 | 0 | 0 | 296 | 0 | 20,828 | 562 | 11,420 | 0% | 0 |
| AlwaysRaise | flop | 24,050 | 0 | 0 | 1,634 | 16,255 | 6,161 | 3,331 | 7,795 | 0% | 0 |
| AlwaysRaise | turn | 12,299 | 0 | 0 | 437 | 9,808 | 2,054 | 1,804 | 2,491 | 0% | 0 |
| AlwaysRaise | river | 7,941 | 0 | 0 | 907 | 6,541 | 493 | 1,836 | 1,400 | 0% | 0 |

- PotOdds: river bets/raises 1,400, of which with < 50% equity vs a random hand (bluff proxy) 0 (0%); folds to a bet 8,809, of which the folded hand would have won or tied on the full board 2,550 (29%); free folds 0
- AlwaysRaise: river bets/raises 7,034, of which with < 50% equity vs a random hand (bluff proxy) 3,621 (51%); folds to a bet 0, of which the folded hand would have won or tied on the full board 0; free folds 0
- leak: AlwaysRaiseBot (B) never folds to preflop bets (11,420 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to flop bets (7,795 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to turn bets (2,491 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to river bets (1,400 times faced): bluffing it is pointless, value-bet it

### Equity vs Random

- verdict: significant edge for EquityBot (p < 1e-300)
- A on button: +845.2 bb/100 (SE 29.1, per hand, no duplicate cancellation)
- A on big_blind: +750.9 bb/100 (SE 31.2, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them; descriptive, no CI): preflop 60% (+43.3), flop 22% (+269.6), turn 7% (+131.6), river 3% (+53.6), showdown 8% (+300.0)

| bot | street | decisions | fold | check | call | bet | raise | all-in | faced bet | fold to bet | free folds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equity | preflop | 20,340 | 4,932 | 2,391 | 9,657 | 0 | 3,360 | 357 | 17,023 | 29% | 0 |
| Equity | flop | 7,654 | 1,302 | 3,097 | 633 | 1,972 | 650 | 1,356 | 2,585 | 50% | 0 |
| Equity | turn | 2,543 | 429 | 1,137 | 241 | 562 | 174 | 454 | 844 | 51% | 0 |
| Equity | river | 864 | 143 | 438 | 75 | 170 | 38 | 162 | 256 | 56% | 0 |
| Random | preflop | 21,275 | 7,128 | 2,591 | 4,533 | 0 | 7,023 | 195 | 13,360 | 34% | 2,642 |
| Random | flop | 8,966 | 3,140 | 2,117 | 1,124 | 2,129 | 456 | 708 | 2,622 | 40% | 2,098 |
| Random | turn | 2,907 | 1,048 | 713 | 302 | 732 | 112 | 217 | 736 | 44% | 726 |
| Random | river | 980 | 375 | 265 | 84 | 234 | 22 | 79 | 208 | 49% | 273 |

- Equity: river bets/raises 208, of which with < 50% equity vs a random hand (bluff proxy) 30 (14%); folds to a bet 6,806, of which the folded hand would have won or tied on the full board 3,012 (44%); free folds 0
- Random: river bets/raises 256, of which with < 50% equity vs a random hand (bluff proxy) 137 (54%); folds to a bet 5,952, of which the folded hand would have won or tied on the full board 2,604 (44%); free folds 5,739
- leak: EquityBot (A) folds 50% to flop bets (2,585 times faced): very exploitable by betting
- leak: EquityBot (A) folds 51% to turn bets (844 times faced): very exploitable by betting
- leak: EquityBot (A) folds 56% to river bets (256 times faced): very exploitable by betting
- leak: RandomBot (B) folded 5,739 times when checking was free (pure loss: it gives up its equity in the pot for nothing)

### Equity vs CallStation

- verdict: significant edge for EquityBot (p < 1e-300)
- A on button: +2027.3 bb/100 (SE 52.2, per hand, no duplicate cancellation)
- A on big_blind: +2058.8 bb/100 (SE 53.0, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them; descriptive, no CI): showdown 100% (+2043.1)

| bot | street | decisions | fold | check | call | bet | raise | all-in | faced bet | fold to bet | free folds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equity | preflop | 20,000 | 0 | 7,273 | 7,915 | 0 | 4,812 | 97 | 10,000 | 0% | 0 |
| Equity | flop | 19,903 | 0 | 11,915 | 0 | 7,988 | 0 | 1,976 | 0 | - | 0 |
| Equity | turn | 17,927 | 0 | 11,527 | 0 | 6,400 | 0 | 1,932 | 0 | - | 0 |
| Equity | river | 15,995 | 0 | 10,832 | 0 | 5,163 | 0 | 2,381 | 0 | - | 0 |
| CallStation | preflop | 22,727 | 0 | 7,915 | 14,812 | 0 | 0 | 97 | 14,812 | 0% | 0 |
| CallStation | flop | 23,842 | 0 | 15,854 | 7,988 | 0 | 0 | 1,976 | 7,988 | 0% | 0 |
| CallStation | turn | 21,158 | 0 | 14,758 | 6,400 | 0 | 0 | 1,932 | 6,400 | 0% | 0 |
| CallStation | river | 18,575 | 0 | 13,412 | 5,163 | 0 | 0 | 2,381 | 5,163 | 0% | 0 |

- Equity: river bets/raises 5,163, of which with < 50% equity vs a random hand (bluff proxy) 891 (17%); folds to a bet 0, of which the folded hand would have won or tied on the full board 0; free folds 0
- CallStation: river bets/raises 0, of which with < 50% equity vs a random hand (bluff proxy) 0; folds to a bet 0, of which the folded hand would have won or tied on the full board 0; free folds 0
- leak: EquityBot (A) never folds to preflop bets (10,000 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to preflop bets (14,812 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to flop bets (7,988 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to turn bets (6,400 times faced): bluffing it is pointless, value-bet it
- leak: CallStationBot (B) never folds to river bets (5,163 times faced): bluffing it is pointless, value-bet it

### Equity vs AlwaysRaise

- verdict: significant edge for EquityBot (p < 1e-300)
- A on button: +1255.2 bb/100 (SE 54.4, per hand, no duplicate cancellation)
- A on big_blind: +1234.5 bb/100 (SE 56.0, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them; descriptive, no CI): preflop 18% (-17.6), flop 38% (-120.1), turn 10% (-130.2), river 2% (-53.3), showdown 33% (+1566.0)

| bot | street | decisions | fold | check | call | bet | raise | all-in | faced bet | fold to bet | free folds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equity | preflop | 30,909 | 3,510 | 0 | 23,636 | 0 | 3,763 | 822 | 30,909 | 11% | 0 |
| Equity | flop | 23,836 | 7,596 | 5,210 | 5,658 | 2,763 | 2,609 | 2,983 | 15,863 | 48% | 0 |
| Equity | turn | 7,268 | 1,998 | 1,696 | 2,404 | 740 | 430 | 1,524 | 4,832 | 41% | 0 |
| Equity | river | 2,018 | 311 | 440 | 912 | 184 | 171 | 886 | 1,394 | 22% | 0 |
| AlwaysRaise | preflop | 21,678 | 0 | 0 | 769 | 0 | 20,909 | 822 | 13,763 | 0% | 0 |
| AlwaysRaise | flop | 18,277 | 0 | 0 | 2,414 | 12,905 | 2,958 | 2,984 | 5,372 | 0% | 0 |
| AlwaysRaise | turn | 5,519 | 0 | 0 | 687 | 4,349 | 483 | 1,546 | 1,170 | 0% | 0 |
| AlwaysRaise | river | 1,738 | 0 | 0 | 344 | 1,383 | 11 | 964 | 355 | 0% | 0 |

- Equity: river bets/raises 355, of which with < 50% equity vs a random hand (bluff proxy) 74 (21%); folds to a bet 13,415, of which the folded hand would have won or tied on the full board 5,448 (41%); free folds 0
- AlwaysRaise: river bets/raises 1,394, of which with < 50% equity vs a random hand (bluff proxy) 725 (52%); folds to a bet 0, of which the folded hand would have won or tied on the full board 0; free folds 0
- leak: AlwaysRaiseBot (B) never folds to preflop bets (13,763 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to flop bets (5,372 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to turn bets (1,170 times faced): bluffing it is pointless, value-bet it
- leak: AlwaysRaiseBot (B) never folds to river bets (355 times faced): bluffing it is pointless, value-bet it

### Equity vs PotOdds

- verdict: significant edge for EquityBot (p = 9.6e-73)
- A on button: +360.3 bb/100 (SE 38.0, per hand, no duplicate cancellation)
- A on big_blind: +173.3 bb/100 (SE 40.5, per hand, no duplicate cancellation)
- hands ended on (share, A's bb/100 from them; descriptive, no CI): preflop 1% (+0.2), flop 23% (+3.8), turn 14% (+0.5), river 12% (-12.5), showdown 50% (+274.9)

| bot | street | decisions | fold | check | call | bet | raise | all-in | faced bet | fold to bet | free folds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equity | preflop | 21,231 | 54 | 7,054 | 9,364 | 0 | 4,759 | 108 | 11,527 | 0% | 0 |
| Equity | flop | 22,693 | 2,343 | 9,966 | 2,320 | 6,777 | 1,287 | 2,314 | 5,950 | 39% | 0 |
| Equity | turn | 15,576 | 1,426 | 8,137 | 1,536 | 3,966 | 511 | 1,620 | 3,473 | 41% | 0 |
| Equity | river | 11,323 | 1,265 | 6,341 | 899 | 2,458 | 360 | 1,539 | 2,524 | 50% | 0 |
| PotOdds | preflop | 22,674 | 90 | 6,742 | 14,315 | 0 | 1,527 | 58 | 14,759 | 1% | 0 |
| PotOdds | flop | 23,870 | 2,264 | 10,891 | 4,765 | 4,915 | 1,035 | 1,227 | 8,064 | 28% | 0 |
| PotOdds | turn | 16,121 | 1,422 | 8,599 | 2,627 | 3,045 | 428 | 898 | 4,477 | 32% | 0 |
| PotOdds | river | 11,604 | 1,185 | 6,495 | 1,400 | 2,291 | 233 | 915 | 2,818 | 42% | 0 |

- Equity: river bets/raises 2,818, of which with < 50% equity vs a random hand (bluff proxy) 451 (16%); folds to a bet 5,088, of which the folded hand would have won or tied on the full board 394 (8%); free folds 0
- PotOdds: river bets/raises 2,524, of which with < 50% equity vs a random hand (bluff proxy) 0 (0%); folds to a bet 4,961, of which the folded hand would have won or tied on the full board 445 (9%); free folds 0
- leak: EquityBot (A) folds 50% to river bets (2,524 times faced): very exploitable by betting

## Files

- results/2026-11-16_PotOdds_vs_Random.json, results/2026-11-16_PotOdds_vs_Random.png, results/2026-11-16_PotOdds_vs_Random_plain.json, results/2026-11-16_PotOdds_vs_Random_plain.png
- results/2026-11-16_PotOdds_vs_CallStation.json, results/2026-11-16_PotOdds_vs_CallStation.png, results/2026-11-16_PotOdds_vs_CallStation_plain.json, results/2026-11-16_PotOdds_vs_CallStation_plain.png
- results/2026-11-16_PotOdds_vs_AlwaysRaise.json, results/2026-11-16_PotOdds_vs_AlwaysRaise.png, results/2026-11-16_PotOdds_vs_AlwaysRaise_plain.json, results/2026-11-16_PotOdds_vs_AlwaysRaise_plain.png
- results/2026-11-16_Equity_vs_Random.json, results/2026-11-16_Equity_vs_Random.png, results/2026-11-16_Equity_vs_Random_plain.json, results/2026-11-16_Equity_vs_Random_plain.png
- results/2026-11-16_Equity_vs_CallStation.json, results/2026-11-16_Equity_vs_CallStation.png, results/2026-11-16_Equity_vs_CallStation_plain.json, results/2026-11-16_Equity_vs_CallStation_plain.png
- results/2026-11-16_Equity_vs_AlwaysRaise.json, results/2026-11-16_Equity_vs_AlwaysRaise.png, results/2026-11-16_Equity_vs_AlwaysRaise_plain.json, results/2026-11-16_Equity_vs_AlwaysRaise_plain.png
- results/2026-11-16_Equity_vs_PotOdds.json, results/2026-11-16_Equity_vs_PotOdds.png, results/2026-11-16_Equity_vs_PotOdds_plain.json, results/2026-11-16_Equity_vs_PotOdds_plain.png
