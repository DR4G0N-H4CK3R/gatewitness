# Where each number in the paper comes from

All numbers are at a declared budget of B = 1 benign alarm per home per day
(Bernoulli-exact calibration) unless stated otherwise. Intervals are exact
(Clopper–Pearson for detection rates, Garwood for alarm rates).

## Channel and commissioning (`results/channel.json`)

| Quantity | Home A | Home B |
|---|---|---|
| Paired span | 21.6 min | 20.3 min |
| Commissioning coupling D (permutation p, M = 499) | 0.466 (0.002) | 0.379 (0.002) |
| No-report rate ε, median (min–max over devices) | 0.66 (0.54–0.85) | 0.71 (0.55–0.80) |
| 1 − γ, median | 0.69 | 0.82 |
| Per-event signal s = (1−ε)(1−γ), median | 0.23 | 0.24 |

## Seven attack classes (`out/episodes_main.csv`, `out/episodes_long.csv` → `results/summary.json["classes"]`, `["long"]`)

| Class | Detected | 95% CI |
|---|---|---|
| Forge, host-timed ACK | 59/60 (0.98) | 0.91–1.00 |
| Leak to a new host (placebo 0.17) | 0.80 | |
| Delay 30 s | 26/60 (0.43) | 0.31–0.57 |
| Change +60 B | 22/60 (0.37) | 0.25–0.50 |
| Hide f = 0.8, 3–4.5 min | 16/60 (0.27) | 0.16–0.40 |
| Forge, firmware-exact ACK | 14/60 (0.23) | 0.13–0.36 |
| Invent (placebo 0.07) | 0.10 | |
| Swap | 1/60 (0.02) | |
| Hide f = 0.8 / f = 1, long horizon (11–14 min) | 55/60 (0.92) / 60/60 | |

Floor check (`results/summary.json["floor_check"]`): of the short episodes
the theory predicted would be missed, 35 of 39 were missed.

## Alarm budget (`out/benign_main.csv`, `out/benign_long.csv` → `results/summary.json["benign"]`)

| Calibration | Benign alarms/day at B = 1 |
|---|---|
| Bernoulli-exact | 234 [86, 510] (6 alarms in 36.9 min) |
| Dispersion-corrected | 117 [24, 342] |
| Markov-exact | 273 [110, 563] |
| Bernoulli-exact, 4.8–8.1 min commissioning | 410 [254, 627] |

## Adaptive attacks (`out/episodes_adaptive*.csv` → `results/summary.json["adaptive"]`)

| Attack | Detected |
|---|---|
| A1 ACK inflation + hide | 16/60 (same as plain hiding) |
| A2a / b / c withhold ACKs / jam / change channel | 53 / 7 / 1 of 60 |
| A3 known-endpoint leak, 1/60, 1/20, 1/5 s | 5 / 6 / 30 of 30 |
| A4 32-bit counter: no / software / exact ACK | 26 / 30 / 12 of 60 |
| A7 / A7′ masked hiding | 0 / 0 of 60 (14% / 12% of reports hidden) |
| A8 chaff +50% / +100% | 31 / 48 of 60 |
| A9 silencing (with LIVE) | 60/60 |
| A10 low-coverage hiding, long horizon | 10/60 |

## Matched alarm rates and cold start (`out/sweep.csv`, `out/coldstart.csv` → `results/summary.json["matched"]`, `["coldstart"]`, Fig. 2)

- GateWitness has no operating point below 156 benign alarms/day.
- At ≤ 100/day the radio-to-hub comparator detects 0.33 of hiding beyond placebo.
- On forgery, GateWitness reaches 1.00 and the best baseline 0.33.
- Cold start at 1 min of commissioning: 5.5 tests admitted per fold without the other home's prior and 13.2 with it. Detection rises from 0.49 to 0.54.

## Derived text numbers (`results/derived.json`)

- `ln_ud_min`: the smallest log-evidence any daily budget needs (9.9 nats).
- `episode_nats`: the log-evidence a short episode carries (2.7–27.9 nats).
- `phi_range`, `b_minus_a_median`: dispersion and persistence of the SUP streams.
- `log_us_per_decision`: cost of the evidence log (1.2 µs per decision).
- `tamper_detected`: modification, insertion, deletion and reordering were all detected.
