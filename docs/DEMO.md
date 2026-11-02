# Live demo script (about 15 minutes)

What the demo shows, in order:

1. The verifier working live on real Home Assistant traffic, in the terminal and
   in a browser dashboard that draws each episode on a timeline.
2. Any attack episode the reviewer chooses, replayed live, giving exactly the
   result recorded in the file the paper's table is computed from.
3. Every number in the paper's tables, compared automatically with the code's
   output.
4. The whole evaluation regenerated from the public dataset while you talk.
   Afterwards git shows the outputs did not change.

All commands below are for Windows PowerShell. On Linux or macOS, use `python3`
and `export GW_DATA=...` instead.

## Before the meeting (once, about 15 minutes)

```powershell
cd C:\Users\mail2\OneDrive\internship\gatewitness-artifact\gatewitness-artifact
pip install -r requirements.txt
# point GW_DATA at the folder that contains ZIOTP2025's "Data" folder,
# e.g. where you unzipped Data.zip
$env:GW_DATA = "C:\path\to\ZIOTP2025"
python demo.py           # first run parses the captures (~30 s); later runs take a few seconds
python check_paper.py    # should end with "33/33 rows agree with the paper"
```

Have the 6-page PDF open next to the terminal on the Results page (Tables II–IV).

`$env:GW_DATA` only lasts for one PowerShell window. Set it again in every new
window.

## During the meeting

### Step 0: start the full regeneration in a second window (runs 10–15 min)

```powershell
$env:GW_DATA = "C:\path\to\ZIOTP2025"
python run_all.py
```

Say: *"This reruns every experiment in the paper from the raw captures:
parsing, all attack injections, the baselines and the statistics. I'll come
back to it at the end."*

### Step 1: the live walk-through (about 3 min)

```powershell
python demo.py
```

What each part shows, and what to say:

| Output | What it means |
|---|---|
| **[1] Load** | 12 real paired captures: radio frames from a CC2531 sniffer and the hub's IP traffic, from two homes. |
| **[2] Commissioning test** | Before trusting the IP tap, the verifier tests statistically that hub reports follow radio uplinks (permutation test). Both homes are admitted with D = 0.466 / 0.379 and p = 0.002, the same numbers as in the paper. |
| **[3] Learn the tests** | It learns its tests on 4 captures only: suppression (SUP), size (SIZE), emitter (EMIT) and volume (VOL). It also learns the ACK-turnaround band (190.9 µs). Thresholds are derived from a budget of 1 alarm/day, not tuned. |
| **[4] Clean run** | It runs on the 2 held-out captures with no attack. The benign alarms here are the honest failure the paper reports: the budget is not met (234/day pooled). |
| **[5] Live attacks** | Each attack is injected into the held-out traffic and the verifier is rerun. The right-hand columns compare the live result with the committed row of `out/episodes_main.csv`, which Table II is counted from. Every row says `yes`. |

### Step 2: the visual replay dashboard (about 5 min)

```powershell
python dashboard.py
```

It opens http://127.0.0.1:8050 in your browser after about 10 s (about 30 s the very first time). Press Ctrl+C in
the terminal to stop it. Ask the reviewer to pick a fold, an attack, a device and
a seed, then press **Replay**. Each replay takes a second or two and is
recomputed live.

What the page shows:

| Part | What to point at |
|---|---|
| **Four tiles** | The live verdict and latency, whether the same test fired on the clean traffic (placebo), the committed row behind Table II, and whether the two are identical. |
| **Radio lane** | The device's real uplinks (blue ticks). For forgery, the forged frames (orange triangles). |
| **Hub reports lane** | The hub's reports for that device. Hidden reports are hollow circles; altered, delayed or invented reports are orange squares. For a leak, a third lane shows the records sent to the new host. |
| **Statistic panel** | The test's CUSUM statistic. It rises with every uplink that has no matching report, or every anomalous frame. The dashed line is the threshold h, derived from the budget of 1 alarm/day. The red line marks the first alarm, which decides the episode. The grey dashed line is the same test on the clean traffic. |
| **Hover** | Move the mouse over the chart to read the statistic at any moment. |
| **Replay all 80 in this fold** | Replays every Table II episode of the fold. It shows `80/80 ... identical` and one card per episode. |
| **Paper vs code tab** | The same 33 rows as `check_paper.py`, as a table: printed in the paper, recomputed, match. |

Good episodes to show:

- A0, Forge (host-timed), device 0, seed 0: detected after 50.5 s; the statistic jumps at the forged frames.
- A0, Hide 80%, device 0, seed 0: missed. The statistic rises but cannot reach h before the window closes. This is the detection floor the paper predicts.
- A0, Leak, device 0, seed 0: the new-endpoint rule fires after 40.1 s.
- A0, Add 60 B, device 0, seed 0: detected, but the clean traffic also fired (placebo), which is why Table II reports the placebo column.

### Step 3: let the reviewer choose in the terminal (optional)

The same replay without the browser. Ask the reviewer to pick a home and fold (A0–A2, B0–B2), an attack, a device
(0 or 1) and a seed (0–4):

```powershell
python demo.py --fold B2 --attack forge --device 1 --seed 3
```

Attack names: `hide delay change swap invent leak forge forgeX`.

Then replay every episode of one fold, 8 classes × 2 devices × 5 seeds = 80
episodes:

```powershell
python demo.py --fold A1 --all
```

It ends with `80/80 live episodes identical to the committed rows behind
Table II`. Table II pools the six folds, so 480 episodes in total.

### Step 4: are the paper's values accurate? (about 2 min)

```powershell
python check_paper.py
```

Each row prints the value exactly as written in the paper, recomputed from the
result files with the same rounding. It covers Section V's text numbers, every
row of Tables II, III and IV, and the frame-counter statistics. It ends with
`33/33 rows agree with the paper`. Point at a few rows in the PDF while it is on
screen.

### Step 5: the regeneration has finished

In the second window, `run_all.py` ends by running `check_paper.py` on the
freshly computed results: `33/33 rows agree`. Then:

```powershell
git status
```

Only `results/derived.json` shows as changed. That file holds one wall-clock
timing (the evidence log's µs per decision), which depends on the machine:

```powershell
git diff results/derived.json
```

Every other output, including all CSVs, `summary.json` and the figure, is
byte-for-byte identical to the committed version.

If many files show as modified, git is reporting Windows line endings, not
different values. Check with `git diff --ignore-cr-at-eol --stat`.

## Questions to expect, and honest answers

- **"Are these attacks real?"** They are injected into real recorded traffic,
  not produced by a compromised hub. That is the paper's main stated
  limitation, and a hooked ZHA hub testbed is the next step.
- **"Why did a detected attack say (placebo)?"** The same test also alarmed on
  the clean traffic in that window. The paper reports that rate in its own
  column, so chance alarms are visible rather than counted as wins.
- **"Why do some attacks get missed?"** The theory predicts it. Short hiding
  episodes carry too few uplinks to detect (35 of the 39 predicted misses were
  missed), and masked hiding and substitution leave no metadata trace at all.
- **"Why is the result identical every time?"** Every random choice is seeded
  from the home, fold, attack, device and seed, so the code is deterministic.
  That is what makes the replay comparison possible.
- **"Where is each number computed?"** `docs/RESULTS.md` maps every number to
  its file and field.

## If something goes wrong

- `FileNotFoundError ... Data`: `GW_DATA` is not set in this window, or points
  one folder too deep or too shallow. It must contain `Data\Web_Interaction`
  and `Data\Extra\IP-Traffic`.
- Scapy prints a warning about libpcap or Npcap: harmless. The code only reads
  files.
- No dataset on the demo machine: `python check_paper.py` still works, because
  it uses the committed results.
