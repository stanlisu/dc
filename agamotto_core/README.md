# agamotto_core — private C++ core for the agamotto sentinel port

The **IP side** of the agamotto port. Implements the opaque contract declared in
the public `sentinel` repo
(`Strategy/ltp_release/ltp_strat_sdk/stan_code/agamotto_core.hpp`) and builds as
`libagamotto_core`, which the public strategy links against.

**Nothing in here may be copied into the sentinel repo.** Same split, and same
reasoning, as [`../sentinel_core`](../sentinel_core/README.md).

## Status: Phase 5 — the port DECIDES (and still never sends)

| stage | state |
|---|---|
| 15m kline builder from ticks | ✅ 102/102 self-test assertions (84 + rule 7) |
| Backfill ingestion + validation | ✅ (loader lives in the strategy; CSV is the seam) |
| Parity vs Binance's own klines | ⚠️ **7/9 columns exact; the two taker_buy_\* miss** (below) |
| 2.0 `Table` + `pdops` scaffolding, ta-lib linked | ✅ |
| 2.1 numeric primitives vs pandas 2.3.3 | ✅ 70 gated + 6 negative + 19 probe specs; byte-identical report on clang/arm64 and gcc 8.5/x86-64 |
| 2.2 OHLC / returns / MA / volume columns (24) | ✅ vs the REAL `research.py`; 4 scenarios x 699 rows, clang/arm64 and gcc 8.5/x86-64 |
| 2.3 TA-Lib indicator block (25 calls -> 29 columns) + `parkinson_vol` | ✅ vs the REAL `research.py`; 54 columns x 4 scenarios x 699 rows, both toolchains |
| 2.4 rolling stats (`std`/`skew`/`kurt`/`acf_lag1`) | ✅ vs the REAL `research.py`; on `hist_return`, not close. `std` **is** `f085` — see below |
| 2.5 scale-free levels (7 columns) | ✅ vs the REAL `research.py`; **65 columns x 5 scenarios x 699 rows, both toolchains, 5/5 negative controls** |
| 2.6 the engine WIRED into `RealCore` | ✅ 42/42 integration assertions; panel computed per live bar; cost + shape cross the ABI |
| 2.7 live reconciliation vs the Python bot | ✅ `tests/live_reconcile.py` — feed -> builder -> engine vs knull, attributed |
| Rule 7: the DROPPED-TRADE detector | ✅ 102 kline self-test assertions (was 84); per-bar + per-run, across the ABI |
| 3.0 the regime gate (30 atom predicates, codes only) | ✅ **62 deployed regimes + 25 probes x 699 rows x 5 scenarios, EXACT, both toolchains, 5/5 negative controls** |
| 3.1 the gate WIRED into `RealCore` + the strategy | ✅ 75/75 integration assertions (was 42); `[AGGATE]` per bar and at shutdown |
| 4.0 the linear model runner (per regime, N from the artifact) | ✅ **62 deployed regimes x 699 rows x 5 scenarios x 2 price scales, both toolchains, max rel dev 6.7e-14 against the DEPLOYED sklearn pipeline; 6/6 negative controls; 12 refuse-to-load cases** |
| 4.1 the runner WIRED into `RealCore` + the strategy | ✅ 119/119 integration assertions (was 75); `[AGMODEL]` at boot, `[AGDEC]`/`[AGPRED]` per bar |
| 5.0 the centred per-leg threshold gate + the vote | ✅ **62 deployed regimes x 699 rows x 5 scenarios x 2 price scales, both toolchains, `fired`/`side` EXACT against the imported reference; 8 negative controls; 12 gate refuse-to-load cases** |
| 5.1 the decision WIRED into `RealCore` + the strategy | ✅ 153/153 integration assertions (was 119); `[AGDEC]` gate at boot, decision + votes per bar, `bar_to_signal_us` at shutdown |
| 5.2 live decision reconciliation vs the Python bot | ✅ `tests/live_reconcile.py` STEP 6 — 3 shared 15m bars on hydra, FOUR sources per bar, PORT 0 / FEED 0 / TRANSCRIPTION 0 |
| an order path | ❌ **and deliberately never** — see below |

`decide()` now populates every field and `fired` can be true. **There is still no
order path**, in this core, in `AgamottoStrategy`, or behind any flag: `nm -DC`
on the shipped `libtsAgamotto.so` finds zero `sendOrder` references. Shadow is
STRUCTURAL here, exactly as it has been since Phase 1. Arming is an
operator-gated decision that this port does not make.

ta-lib 0.6.4 is linked, version-pinned, and CALLED. There is still **no LightGBM
dependency** — agamotto's `SWEEP_MODELS` is Ridge, so the model is a dot product
(see below).

## Phase 5 — the decision rule, transcribed not invented

`src/decision_rule.cpp` is a line-by-line transcription of three reference files
and nothing else. The full derivation lives in `src/decision_rule.hpp`; the
short form:

| step | reference | rule |
|---|---|---|
| the leg edge | `marvel/gauntlet/thresholds.py::signed_threshold` | long `C+T`, short `C-T` |
| a regime fires | `dc/agamotto_pkg/.../trading.py::dual_gate_filter:42-49` | long `y_pred > edge`, short `y_pred < edge` |
| the vote | `trading.py::make_decision:846-861` | `net = long_count - short_count`; `final_qty = base_size * net * REVERSE` |
| the side | `knull/orb_bridge.py::_decisions_to_signals:156,168` | `abs(qty) < 1e-9` -> FLAT, else `sign(qty)` |

**The edge this core computes is BIT-IDENTICAL to the deployed stack's
`optimal_threshold` column.** Verified: `-0.00012442 + 0.00070083` is exactly the
`0.0005764100000000001` every one of the 39 long rows carries, and
`0.00030606 - 0.0013514` is exactly the `-0.00104534` on all 23 short rows. The
core reads the five numbers from `algo_params` and the bot reads one number from
the stack CSV, so the two are independent expressions of the same quantity and
`tests/decision_parity.py` grades one against the other.

### What is NOT in the rule, and was looked for

* **No `MIN_SIGNAL_COUNT`.** It is a MJOLNIR key (`mjolnir/README.md` line 130);
  every occurrence in marvel is a mjolnir test fixture. It is absent from
  `pred_agamotto.base.15m_1/setting.json` and no agamotto or orb code path reads
  it. The effective minimum is ONE net vote.
* **No hold-TTL.** There is no signal hold anywhere in the agamotto path. The
  only TTL on the arm is `EXECUTORS[<venue>].SIGNAL_TTL_SEC = 300.0`, an ORDER
  lifetime inside the executor that never reaches a decision. Every bar
  re-decides a TARGET position from scratch.
* **No 2-bar arm.** `dual_gate_filter` engages its dual-horizon branch only when
  the frame carries BOTH `opt_threshold_2bar` and `prediction_2bar`. The deployed
  `filtered_optimal_regime_stack.csv` has neither, so the reference takes the
  1-bar branch on every bar of this arm.

A core that added any of the three would be quieter than the reference and would
look exactly like one whose regimes had stopped firing.

### The 2 bps floor is REFUSED, never clamped

`GateParams::validate()` (called from `createCore`, before a 7.3-day warmup)
throws on a non-finite bound, a negative width, a width under
`kAbsThreshFloor = 0.0002`, and a `reverse` outside `{-1,+1}`. Nothing is raised
up to the floor: clamping would leave a strategy running at a gate nobody chose
with nothing reporting the difference — the banned `max()`-on-a-derived-quantity
shape.

The constant cannot be `import`ed into C++, so `tests/decision_parity.py`
asserts `kAbsThreshFloor == gauntlet.thresholds.ABS_THRESH_FLOOR`. That makes it
a **gated copy of one source**, not a second declaration.

### `Decision::y_pred` is REPORTING, not the decision

The reference produces a VOTE COUNT and keeps no single prediction. `y_pred` /
`threshold` / `threshold_center` / `winning_regime_code` therefore describe one
REPRESENTATIVE regime — the largest `|y_pred|` among the voters on the MAJORITY
leg — chosen so the logged triple explains itself. They can never change `fired`
or `side`. Phase 4's "largest `|y_pred|` over everything that predicted" is
replaced, not extended: it could name a SHORT regime as the winner of a LONG
decision.

### The live reconciliation (5.2) — four decisions per bar

`tests/live_reconcile.py` STEP 6 puts four decisions side by side on every
shared bar, because "the bot went long and the core went flat" is not a finding
until you know which seam it came from:

| | source | what a difference means |
|---|---|---|
| (a) | the C++ core's own `[AGDEC]`/`[AGPRED]` | — |
| (b) | the REFERENCE path over the C++ PANEL row | **(a) vs (b) is the PORT.** Same features, same weights, same gate: nothing else can explain a difference |
| (c) | the REFERENCE path over the BOT's `debug_features` row | **(b) vs (c) is the FEED.** Identical code, two chains' features |
| (d) | the BOT's OWN `Decision for <sym>:` log line | **(c) vs (d) is the TRANSCRIPTION.** The only in-band record of what production actually decided |

The reference path is imported, never rewritten: `research_filters
.apply_filter_mask` for the gate, `utils.weights_io.load_regime(...).predict`
for the model, `agamotto.trading.dual_gate_filter` for the fire, and
`gauntlet.thresholds.signed_threshold` for the edge. Only `make_decision`'s four
lines of vote arithmetic are transcribed — and (d) is what grades them.

**The gate must run in (b) and (c).** A first draft omitted it and reported 23
voters on a bar where the bot reported none: without the filter, all 62 regimes
vote, including the 53 that cannot fire live. The one-row mask evaluation is
exact *only* while `price_range_pct_q50` is a column of the row —
`research_filters` otherwise falls back to a rolling quantile that, on one row,
returns the value itself and makes every vol regime read as "did not hold". The
script asserts the column rather than trusting it.

Measured on hydra 2026-08-20, bars 03:15 / 03:30 / 03:45Z: inputs 9/9 exact on
all three, features 65/65 agree, and **all four decision sources agree on all
three bars** — including the firing one, where the bot's own log names
`r039_and_r008_long` and nothing else.

### The trade-id gap detector FIRES ON BIT-EXACT BARS

Reported, not fixed. Rule 7's detector flagged 291 "missing" ids across four bars
whose **nine columns match Binance to 6e-15**. Both cannot be true: a bar that
lost 198 trades cannot reproduce Binance's volume, quote_volume, trade count and
both `taker_buy_*` to fifteen digits. The publisher subscribes to BOTH
`btcusdt@trade` and `btcusdt@aggTrade`, and an aggTrade id is not a trade id — so
a gap in the merged sequence is not evidence of a dropped ring slot.
`live_reconcile.py` therefore reports it and lets the **arbiter** (Binance's own
kline) decide whether a bar is dirty; letting the hint decide made every bar
UNATTRIBUTED and printed "65 live differences" over a table of exact zeros. The
builder fix is to track the two id sequences separately.

### `reverse` is refused outside `{-1, +1}`

The reference multiplies a QUANTITY by `REVERSE` and never validates it
(`trading.py:861`), so `REVERSE: 0` is a permanently flat bot and `REVERSE: 2`
silently doubles live size — `_decisions_to_signals` recovers `net_count` back
out of the qty. A shadow core reports a SIDE and can represent neither, so both
are refused at `createCore`. **Reported as a reference weakness, not fixed
there.**

## Rule 7 — the SHM ring drops trades, and now it says so

Phase 2.7 reported (below, "FINDING") that the SHM tick ring loses events in a
burst and that the loss is **completely silent**: on hydra 2026-08-19 the 15:15
15m bar lost **5.04%** of its trades and **missed the high by 562 points**
(69888 against Binance's 70450), while `unclassified` was 0, `aggressor_source`
read `exact` and `conv_err` was 0. Every counter in the core tallies what
ARRIVED, and a ring slot that was overwritten never arrives.

Binance trade ids are monotonic per symbol, and the builder already parses
`last_trade_id` — rule 2 dedupes on it. So `id > expected` is the one piece of
in-band evidence the loss leaves behind, and rule 7 counts it:

| where | field |
|---|---|
| `KlineBar` | `trade_id_gaps` (discontinuities in this bar), `n_trades_missing` (ids they skipped) |
| `CoreDiagnostics` | `trade_id_gaps`, `trades_missing` — the run totals |
| strategy | `[AGBAR] ... gaps=N missing=M`, a per-bar `[AGDROP]` **ERROR** with the loss %, and a run-level `[AGDROP]` at shutdown |

**Nothing is recovered.** The trades are gone from the ring and cannot be
reconstructed from it; the fix belongs in the feed publisher. This converts a
silently wrong bar into a loud one, which is what makes a Phase-4 signal
computed off it identifiable as suspect rather than taken at face value.

Three things are deliberately **not** gaps, because each would manufacture a
loss the ring did not cause, and each is pinned by a self-test:

* **the FIRST id of a run** — no predecessor to be missing from. Without this
  guard the first bar of every session reports ~4.1e9 missing trades on
  BTCUSDT, which is not a big number so much as a meaningless one;
* **an id we dropped ourselves as a duplicate** (rule 2) — it arrived;
* **the expected id** — the ordinary case.

**Declared limitation.** An id that arrives OUT OF ORDER is first counted
missing by the jump that overtook it and is then dropped by rule 2 when it turns
up. That is the right answer for "how many trades are not in this bar" — rule 2
means the late one never contributes either — and the wrong answer for "how many
did the ring lose". The two coincide on Binance's trade stream, which is ordered.
Pinned by self-test [18] rather than left to surprise someone.

```
./build/kline_parity_driver --selftest     # 102 assertions; [16]-[19] are rule 7
```

| self-test | property |
|---|---|
| [16] | a clean id stream reports **0** gaps and 0 missing, and the ~4.1e9 first id is not read as loss |
| [17] | an injected 7-id hole plus a 1-id hole report **exactly 8 missing across 2 gap events** |
| [18] | a duplicate and an out-of-order replay contribute **nothing**; a hole opened in bar N is charged to bar N and not to bar N+1 |
| [19] | a backfilled (REST) bar carries no id sequence, so it claims 0/0 rather than an invented figure |

## Phase 4 — the model runner

`agamotto`'s `SWEEP_MODELS` is **Ridge**, so there is no booster and no tree
walk. The whole model, for one regime, is

```
y = intercept + SUM_i coef[i] * ((x[i] - center[i]) / scale[i])
```

over the features `features.txt` names, read from the panel's **newest row** —
the bar that just closed, the same row the gate classified. `REVERSE` and the
threshold gate are applied DOWNSTREAM by the strategy out of `algo_params`; they
are not the weights' business and are not applied here.

The weights arrive as three text files per coded regime, written by
`marvel/gauntlet/export_agamotto_sentinel_weights.py` (which lives in marvel, not
here, so no machine that exports weights needs the private checkout):

```
<weights_dir>/r029_and_r001_and_r073_long/
    model.txt     model_kind linear / format_version 1 / n_features N /
                  intercept <double> / coef / N doubles, %.17g
    scaler.txt    "<N>" then N lines of "<center> <scale>"   (RobustScaler)
    features.txt  N CODED feature names, in model input order
```

### `model_kind` is the first token so a LightGBM dump cannot be loaded

`sentinel_core` (mjolnir) reads a `model.txt` that is LightGBM's own native text
dump and opens with the token `tree`. **Same filename, completely different
format.** `loadLinearModel` refuses any file whose first token is not
`model_kind`, and says so naming both formats. Without that check, pointing
agamotto at a mjolnir weight tree parses a booster header as coefficients and
predicts plausible numbers — the failure mode with no symptom.
`tests/model_parity_driver.cpp --selftest` builds exactly that file and requires
the refusal, along with 11 other malformed artifacts.

### *** `window_2026_07_31` IS NOT ONE TRAINING RUN, AND THE SPLIT IS THE DANGEROUS ONE ***

`TOPN_ICS` is 5. Measured 2026-08-20 across all 109 exported regime directories
of the deployed window:

| feature count | regimes | fitted | `window_id` |
|---|---|---|---|
| 5 | 53 | 2026-08-16 | 33 |
| 16 | 56 | 2026-08-13 | 33 **and** 34 |

`window_id` 33 and 34 are a **different train/test split** (7 of the 9 deployed
16-feature regimes are `window_id` 34), inside one window label. And across the
62 rows of the deployed stack the split falls exactly on the fault line that
matters:

> **All 53 vol-quantile-gated (inert) regimes carry 5-feature models.**
> **All 9 FIRABLE regimes carry 16-feature models.**

So a runner that hardcoded `TOPN_ICS = 5` would not fail on a corner case — it
would be **wrong on every regime that can actually trade and right on every
regime that cannot**, which is the worst possible arrangement for noticing.
`n_features` is therefore read per regime from the artifact, `features.txt` is
the sole authority on which columns to take, and three things make the mixture
impossible to miss:

* `CoreDiagnostics::model_feature_count_variants` / `model_features_min` /
  `model_features_max` cross the ABI, and the strategy prints
  `ModelBook::inventory()` as `[AGMODEL]` at boot;
* `tests/model_parity.py` **fails** if the deployed stack does not contain both
  widths, and reports the inert-vs-firable split explicitly;
* the integration fixture spans three different widths on purpose, so a core
  that assumed one width cannot pass 4.1 either.

### A FINDING: six scaler rows have `scale == 1.0` from a ZERO IQR

sklearn's `_handle_zeros_in_scale` substitutes `1.0` for a `RobustScaler`
`scale_` entry whose training-window IQR is exactly zero — i.e. a feature that
was **constant** over the train period. Six such rows exist across the 109
exported regimes, and all six are the same column, `f089`:

| regime | centre | in the deployed stack? |
|---|---|---|
| `r029_and_r019_and_r074_long` | 100 | yes (inert) |
| **`r069_and_r001_long`** | 100 | **yes, and it is FIRABLE** |
| `r069_and_r019_long` | 100 | no |
| `r069_and_r045_long` | 100 | no |
| `r069_and_r045_short` | 0 | no |
| `r069_and_r066_long` | 100 | no |

`f089` is bounded in `[0, 100]`, so a constant train window pins it at a rail.
The consequence is **not** a divide-by-zero: it is that the feature enters the
model **unscaled** while every neighbour is divided by its own IQR, so
`coef * (x - 100)` swings over the full `[-100, 0]` range at whatever weight the
fit assigned to it. Exported faithfully — those numbers *are* the deployed model
— but counted (`model_unit_scale_features`) and named at boot, because a scale of
exactly 1.0 next to IQR-scaled neighbours is a fitted-on-a-constant artifact
rather than a modelling choice. `r069_and_r001_long` is the regime the live hydra
bot was running on 2026-08-07. **Reported, not fixed** — the fix belongs in
training.

### Two DECLARED DIVERGENCES from `trading.py`

**1. The prediction clip is not reproduced.** `trading.py:705-713` does

```python
if np.abs(preds).max() > 1.0:
    logger.warning("prediction overflow ... clipping to [-1, 1]")
    preds = np.clip(preds, -1.0, 1.0)
```

which is a `np.clip` on a **derived quantity** — the shape `CLAUDE.md` bans,
because an out-of-range prediction is the degenerate-scaler bug signalling
itself and the clamp converts it into a plausible in-range number. It is also
not part of the exported contract: `utils.weights_io.RegimeArtifacts.predict` —
the loader this port's parity gate grades against, and the one tesseract uses —
does **not** clip. Reproducing it here would make the C++ disagree with the
reference. `tests/model_parity.py` reports how often it would have bitten (12,
4 and 76 predictions on the three scenarios that produce extreme features).
Since the deployed thresholds are ~5.8e-4, clipping never changed *whether* a
leg fired; it changed the magnitude, and therefore any ranking built on it.
**Reported, not fixed** — the fix belongs in `trading.py`.

**2. An `inf` feature: the reference cannot be asked, the core answers.**
`trading.py:697-700` fills NaN with `0.0` across the selected model columns and
does **not** touch `inf`. sklearn's `validate_data(..., ensure_all_finite=
"allow-nan")` inside `RobustScaler.transform` allows NaN and **rejects `inf`
outright** (`ValueError: Input X contains infinity...`). In production that
`ValueError` is swallowed by `trading.py:744`'s bare `except Exception`, logged,
and the regime returns an empty frame — so **an infinite feature makes the regime
silently produce no prediction for that bar**. The core instead propagates the
`inf`, reports a non-finite `y_pred`, counts it (`nonfinite_predictions`) and
**excludes it** from `n_triggered` and from the winning-regime pick. Same trading
outcome (no signal), reached by a route that leaves a number behind. The gate
asserts the divergence is confined to exactly those rows: on every `(regime, row)`
whose model input carries `±inf`, the C++ `y_pred` **must** be non-finite — a
finite one there would mean an `inf` had been absorbed into a plausible number.

The NaN fill itself **is** reproduced, exactly: a filled cell contributes
`coef * ((0 - center) / scale)`, which is emphatically not "no contribution".
`nan_features_filled` counts them, and the `--negative` control that skips the
term instead goes red.

### Codes only, and the directory name is reconstructed rather than carried

`RegimeSpec` crosses the ABI as an array of `uint16` atom codes; the exporter
names directories by the coded regime. The two are bridged by
`regimeDirName({29,1,73}, LONG) -> "r029_and_r001_and_r073_long"` — verified
2026-08-20 to reproduce all 109 exported directory names and all 62 deployed
stack rows exactly, with atom **order preserved, never sorted**. No regime name
and no feature name is a string literal in `model_runner.cpp`, and
`build_linux.sh`'s artifact leak audit passes on the built `.so`.

### Column indices are resolved ONCE, at boot

`features.txt` codes are resolved to panel column indices when the stack is
installed, against `canonicalPanelColumns()` — which is obtained by **running
`engineerFeatures` once** on a synthetic panel rather than from a list kept
alongside it. A hand-maintained list is one edit away from disagreeing with the
engine, and the disagreement would surface as a model reading the wrong column:
a plausible number for the wrong feature, which nothing downstream can detect.
It costs one ~55 ms engine pass per process.

A code the panel does not carry is a **boot** error. Agamotto warms for 700 15m
bars (7.3 days), so a runner that discovered a missing column on its first panel
would have booted clean, warmed for a week, and only then said its weights were
unusable. `ModelBook::assertPanelLayout` re-checks the layout once per panel, so
a panel whose columns moved cannot feed every model its neighbour's numbers.

### `weights_dir` is a NEW REQUIRED config key, and there is no `""` escape

`createCore` gained a fourth parameter and `agamotto_algo_params.json` a
`weights_dir` key. Unlike `backfill_csv` and `regime_stack_csv` — where `""`
means "explicitly none" — an empty or non-existent `weights_dir` **throws**: a
core that gated bars and predicted nothing would produce a silent no-signal day
indistinguishable from a working one. A stack entry with no directory under it
throws out of `setRegimeStack`, **naming the regime**, which is the same failure
the Python bot raises (`FileNotFoundError: Regime folder r060_and_r075_long not
found`) and the one that caught a real stack/weights mismatch.

### What `decide()` does, and what it deliberately does not

| field | Phase 4 |
|---|---|
| `bar_ts_ms` | the panel's bar |
| `n_triggered` | regimes that BOTH fired AND predicted finitely |
| `y_pred` / `winning_regime_code` | that set's largest `\|y_pred\|`, ties -> lowest stack index |
| `signal_emit_ns` | stamped when anything was predicted |
| `fired` / `side` / `threshold` / `threshold_center` | **untouched (false / 0 / 0 / 0)** |

The winner is a **reporting order so the log line names something — not a
selection rule.** Phase 5 must replace it with the threshold/vote rule
(`thresholds.read_threshold` / `read_threshold_center`, the per-leg centred gate,
`REVERSE`), not build on it. `winning_regime_code` carries only the conjunction's
**leading atom**, because a 3-atom regime has no single code and the ABI field is
one `uint16`; `ICore::winningRegimeIndex()` is the unambiguous identity and the
caller already holds the stack.

Per-regime predictions are available individually via
`ICore::regimePrediction(i)`, which returns **NaN** — never 0.0 — for a regime
that did not fire. 0.0 is a legitimate prediction and would read as a confident
flat call.

**Expect at most 9 predictions per bar on live data.** 53 of the 62 deployed
regimes cannot fire (PR #532, above), and a regime that does not fire is never
scored — the reference predicts only on rows its filter let through
(`predict(filtered_signals)`), and scoring a gated-out bar spends the time to
produce a number nothing may act on.

### The gate

```bash
tests/run_model_parity.sh              # macOS / clang -O2
tests/run_model_parity.sh --linux      # rocky8 / gcc 8.5, driver in the build image
tests/run_model_parity.sh --both
tests/run_model_parity.sh --negative   # 6 controls, each must go red
```

The reference side is **`utils.weights_io.load_regime(...).predict`** — the same
loader the live bot and tesseract call, imported from the marvel tree and handed
the same `ridge_*.pkl` files production loads. It is never reimplemented: a
hand-written `X @ coef + intercept` on the Python side would grade the C++
against a second copy of the same idea, and the two would agree happily while
both disagreed with sklearn.

The C++ side reads the **text export**. So this gate closes the loop the exporter
opens: the exporter proves *pickle -> text* round-trips (measured 1.1e-13 on
random probes), and this proves *text -> C++ prediction* matches the pickle **on
a real engineered panel**. `--weights` and `--raw-weights` must be different
directories, and the harness refuses if they are not — otherwise it would be
comparing the export with itself.

The driver emits the panel it engineered **and** the predictions it computed from
it, and the harness feeds *that* panel to the reference — the same discipline
`tests/regime_parity.py` uses. Engineering a panel in Python alongside would fold
feature parity's own 1e-9 tolerance into a comparison that is supposed to be
about the dot product alone.

**Every row, not just the scored one.** Live scores one row per bar, so grading
only that row would compare five numbers per regime across the whole suite —
nowhere near enough to separate a correct prediction from one that is right near
the mean and wrong in the tails, or that has a coefficient's sign wrong on a
feature that is rarely large. `predictRow` is row-independent, so all 699 rows
are graded and the newest row is reported separately.

**The tolerance denominator is the regime's own signal scale**
(`max(|a|, |b|, rms(reference))`), not the individual cell. These predictions are
returns and cross zero constantly; a pure per-cell relative gate would be
measuring the denominator. The `rms` is taken over the *reference*, so the C++
cannot influence its own tolerance.

Beyond equality the harness asserts that the panel actually **moves** each
model (a runner emitting only its intercept would match a reference that did the
same), that both feature widths are present, and that the firable regimes are the
16-feature ones.

**Measured 2026-08-20**, 62 regimes x 699 rows x 5 scenarios x 2 price scales:

| toolchain | worst relative deviation |
|---|---|
| macOS / clang -O2 (arm64) | **6.670e-14** |
| rockylinux:8 / gcc 8.5 (x86-64) | **6.649e-14** |

i.e. ~15,000x inside the 1e-9 gate, and ~1e10 times smaller than anything that
could move a decision (the deployed thresholds are ~5.8e-4). The residue is BLAS
summation order — sklearn's `Ridge.predict` goes through `dgemv`, the core sums
left to right — not a difference in the model.

### Negative controls (6/6 caught)

Two are **driver flags** rather than source mutants, because what they must prove
absent is a whole missing *step*, not a wrong operator:

1. `--no-scaler` — predict `intercept + sum(coef * x_raw)`, dropping
   `(x - center)/scale` entirely. On features whose centre is near 0 and whose
   scale is near 1 the two are nearly the same number, so without this a runner
   that ignored the scaler could look correct on a lucky regime.
2. `--perturb-coef 0:0:1e-6` — nudge ONE coefficient of ONE regime by 1e-6, five
   orders of magnitude **below** the deployed threshold. This is not a control
   that only catches vandalism; it pins the gate's resolution well under anything
   that could change a decision.

The other four are `sed` mutations of `src/model_runner.cpp`, each guarded twice
(the new text must appear, the old must be gone) so a drifted `sed` cannot report
a green control over an unmutated binary:

3. `(x - center)` -> `(x + center)` — the most plausible transcription error in a
   RobustScaler port, and nearly invisible on a centre near zero.
4. divide by `scale` -> multiply by `scale` — and note this changes **nothing**
   on the six deployed rows whose scale is exactly 1.0, so it must be caught by
   the other rows.
5. the NaN fill contributes `coef * ((0 - center)/scale)` -> skip the term. The
   "treat missing as absent" reading, which is not what `trading.py` does.
6. `features.txt` order reversed — pairs every coefficient with the wrong column
   while keeping the count, the scaler shape and the arithmetic all valid.

### 12 refuse-to-load cases

`./build/model_parity_driver --selftest`, run before the parity gate. Every one
is a way a wrong weight tree could load and **predict anyway**: a LightGBM
`model.txt`; a missing regime directory; a `features.txt` code absent from the
panel; a `scale` of 0; a `scaler.txt` row count disagreeing with `n_features`; a
short `features.txt`; a duplicate feature code; an unsupported `format_version`;
a non-finite coefficient; a trailing token after the coefficient block; an empty
conjunction (`baseline`); an empty `weights_dir`. Plus a well-formed control that
must **load**, so the suite cannot pass by refusing everything.

### The integration gate (4.1 / 5.1)

`tests/core_integration_driver.cpp` grew from 75 to 119 to **153** assertions —
Phase 5 added the gate's refuse-to-load cases at `createCore`, the vote
recomputed from the ABI's own `regimeTriggeredLatest()` flags, the
`side == sign(net_count * reverse)` identity, the representative regime's
self-consistency, `decide()`'s IDEMPOTENCE, and the first real measurement of
the kline-built -> signal-generated span. It writes
its **own** tiny weight tree — feature counts 1, 2 and 3, coefficients it chose —
because this file grades the **wiring** and an expectation it can state exactly
beats one it has to trust. That also keeps it hermetic, which is why
`build_linux.sh` runs it inside the build image with no weights mounted;
`--weights DIR` overrides it for a smoke test against a real export.

The load-bearing assertion is a **closed form**: for every firing regime, the
prediction must equal `intercept + sum coef*((x - center)/scale)` computed from
`panelLatest()` — the ABI's own accessor, a *different* code path from the one
the core predicted through. That catches a model reading the wrong row, the wrong
column, or another regime's coefficients. It is graded on every bar of the bench
run, not only the first warm bar (which gates everything out on the fixture
walk), and the run asserts that the number of closed-form checks **equals**
`predictions_computed` — so "0 fired" can never read as a pass.

## Phase 3 — the regime gate

`src/regime_gate.{hpp,cpp}` implements the predicates of
`agamotto_pkg/src/agamotto/research_filters.py` `apply_filter_mask` /
`allowed_positions`, transcribed branch by branch. **30 atom predicates**, which
is every atom in `map.json` that `apply_filter_mask` can evaluate on a 15m kline
panel.

### The inventory of what is actually deployed

`pred_agamotto.base.15m_1/filtered_optimal_regime_stack.csv`, copied verbatim
into `tests/regime_stack_deployed.csv` (md5 `c90264a1…`) so the gate grades the
stack that is RUNNING rather than one somebody retyped:

* **62 regimes** — 39 long, 23 short;
* **28 are 2-atom conjunctions, 34 are 3-atom**. There are no single-atom
  regimes, which is why mjolnir's `^r\d{2,}_(long|short)$` matches none of them;
* **16 distinct atoms**: r001, r003, r008, r019, r029, r039, r040, r045, r048,
  r060, r065, r066, r069, r073, r074, r075;
* the busiest are r069 and r029 (20 regimes each), then r075 (19), r074 (17),
  r073 (17), r008 (13).

### *** 53 of the 62 CANNOT FIRE, AND THE PORT KEEPS IT THAT WAY ***

r073 / r074 / r075 compare `price_range_pct` against
`price_range_pct_q80 / q90 / q95`, which research.py:371-376 builds as
`rolling(VOL_Q_WINDOW=700, min_periods=700)`. The live panel is 699 rows, so
min_periods is never met, all three cutoff columns are NaN on every row, and
`x > NaN` is False. Every regime carrying one of those atoms is **inert live**.

That is today's production behaviour under an open finding — marvel PR #532,
`docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md` — and reproducing
it is the point. A port that "fixed" the min_periods, or widened the panel to
700, would start 53 regimes firing against Ridge weights never trained on a
firing regime, and the port would look like the cause of whatever followed. When
the finding is resolved it is resolved in research.py FIRST and mirrored here.

**The 9 that CAN fire**, all of them 2-atom and none carrying an r07x gate:

| regime | position |
|---|---|
| `r029_and_r045` | long |
| `r029_and_r019` | long |
| `r029_and_r066` | long |
| `r069_and_r008` | long |
| `r069_and_r065` | long **and** short |
| `r039_and_r019` | short |
| `r069_and_r001` | long |
| `r039_and_r008` | long |

(eight names, nine (regime, position) rows — `r069_and_r065` is deployed on
both sides.)

### Codes only, and it is now PROVEN of the artifact

Atoms cross ICore as `uint16` codes in a `RegimeSpec{atom_codes[8], n_atoms,
position}` — an array, not a string, because 34 of the 62 are three atoms and a
parser on the public side would have to know the grammar. The public strategy
splits `rNNN_and_rNNN_..._long` into separators, a position suffix and digits;
it never learns what a code means.

`build_linux.sh` now runs `obfuscation/audit_public_surface.py --repo
build-linux --binaries` and **fails the build** on a hit. `--binaries` is
load-bearing: without it the scanner skips `.so` by suffix and passes without
ever opening the library. Two leaks were found and closed the first time it ran
(2026-08-20), both **feature** names in `throw`/assertion string literals —
`price_range_pct_q50` in `engineerFeatures` and `price_range_pct_q80/q90/q95` in
the integration driver. Zero regime names were ever present. Exception text is
the easiest place for a name to re-enter a compiled artifact, precisely because
it does not look like data.

### Reference quirks reproduced verbatim

Each is a place where writing what the regime's NAME suggests gives a different
column of booleans:

| atom | the trap |
|---|---|
| `mom_positive` | the SHORT branch is `mom < 0`. It is not in `SHORT_ONLY_FILTERS`, so short is allowed and the predicate is INVERTED rather than the regime refused. |
| `stoch_bullish` | the reference's short branch is `stoch_k > stoch_d` — the SAME predicate, not the mirror — and is DEAD, because `allowed_positions` returns `["long"]`. Ported as unreachable rather than "corrected" to `<`, which would make it fire. |
| `near_ma` | a `<` on a SIGNED relative distance, not a \|distance\| band. On the long side EVERY bar below `mvg1` satisfies it — see the finding below. |
| `bb_rebound` | long reads `bb_lower`, short reads `bb_upper`. Two different columns; a mirrored one-column implementation is wrong on one side and plausible on both. |
| `buy_pressure` | 0.55 / 0.45, asymmetric — not one threshold mirrored. |
| volume ratio | `quote_vol_ratio` FIRST, `vol_ratio` second. Both exist in a live panel and they are different numbers. |
| `low_vol` / `high_vol` | duplicated verbatim into both position branches with the same predicate, while the `high_vol_q*` atoms resolve ONCE above the split. |

Not reproduced, declared: `trend_aligned` and the `combined_*` family are
reachable in `apply_filter_mask` but have **no entry in `map.json`**, so they
have no code and cannot cross the boundary; they are not implemented and an
unknown code THROWS. The reference's `strict_filters=False` all-False fallback
is also not reproduced — a live core that quietly gated a typo'd regime to
"never fires" is a strategy that is off with nothing in the log saying so.

### The gate

```bash
tests/run_regime_parity.sh              # macOS / clang -O2
tests/run_regime_parity.sh --linux      # rocky8 / gcc 8.5, driver in the build image
tests/run_regime_parity.sh --both
tests/run_regime_parity.sh --negative   # 5 mutants, each must go red
./build/regime_parity_driver --selftest # atomIsKnown vs atomMask, all 4096 codes
```

`tests/regime_parity_driver.cpp` prints the panel it engineered **and** the
masks it computed from it; `tests/regime_parity.py` hands **that same panel** to
the REAL `research_filters.apply_filter_mask` and diffs. That is what makes the
comparison exact:

> **TOLERANCE IS ZERO. A MASK IS A DECISION, NOT A MEASUREMENT.**

Engineering a panel in Python alongside would drag the feature engine back into
scope, and 1e-9 relative is enormous next to a boolean — a cell 1e-12 from
`adx > 25` would land on opposite sides and the run would go red for a reason
that is not the predicate.

The regime names handed to the reference are the CODED ones the live stack
carries (`r029_and_r001_and_r073_long`); `apply_filter_mask` decodes them itself
through `decode_regime_tolerant`, exactly as production does. Neither side is
ever shown a real name.

**Four assertions, because mask equality alone is vacuous here.** A gate
hardwired to all-False agrees with the reference on 53 of 62 regimes:

1. exact equality, per regime, per row;
2. the 53 r07x-gated regimes are ALL-FALSE **on both sides** — and the q80/q90/
   q95 columns of the panel the gate actually read are asserted all-NaN, so the
   inertness is pinned at its CAUSE and not only at its effect;
3. **the causal control**: each of the 53 is also evaluated with its
   vol-quantile atom STRIPPED (25 distinct "probe" regimes, not deployed), and
   those must fire. This separates "inert because the cutoff is NaN" from
   "inert because the gate is broken", which are identical from the mask alone;
4. nothing deployed may be all-TRUE (that is `baseline` under another name), and
   every firable regime and every probe must fire on at least one scenario.
   All-False is judged ACROSS the five scenarios, not within each: two of them
   are deliberately degenerate (injected NaN runs, a 26-bar flat run, a close of
   1e-305), and requiring `bb_rebound` to fire on those would assert something
   about the synthetic data rather than about the gate.

Verified 2026-08-20 — **87 regimes (62 deployed + 25 probes) x 699 rows x 5
scenarios, 0 differing cells, on macOS/clang and rocky8/gcc 8.5**, with
identical fire rates on both.

| mutant | caught by |
|---|---|
| `mom > 0` -> `mom >= 0` (the boundary) | scenario 5's 26-bar FLAT run, where `mom` is exactly 0.0 |
| `mom_positive` short: `mom < 0` -> `mom > 0` (believing the name) | mask diff on every scenario |
| `high_vol_q95` reads the q50 cutoff (the tempting "fix") | the inert-regime assertion — 53 regimes start firing |
| volume ratio prefers `vol_ratio` over `quote_vol_ratio` | mask diff; r029/r039/r069 appear in 45 of the 62 |
| `near_ma` signed -> `\|signed\|` (believing the name) | mask diff |

**On `>` vs `>=`, honestly:** a boundary mutation only changes a mask where the
two operands are EXACTLY equal. On `adx > 25` against a continuous double that
never happens, so `>=` there is genuinely indistinguishable on any data — a real
limit of this gate, recorded rather than papered over. The controls above sit
where ties DO occur (the flat and zero-volume runs make `mom` exactly 0,
`close == mvg1 == mvg2 == mvg3`, and `stoch_k == stoch_d`).

### Deploying it: `regime_stack_csv` is a NEW REQUIRED config key

`config/agamotto_algo_params.json` gains `regime_stack_csv` — the path to the
deployed `filtered_optimal_regime_stack.csv`, or `""` for an explicitly ungated
run. Required, like `backfill_csv` and for the same reason: a missing key
defaulting to "no stack" produces a run that classifies nothing and looks
exactly like one whose regimes never held. **An existing algo_params.json
without this key will fail to boot** — `root_.get<std::string>` throws — which
is the intended fail-loud, but it does mean every deployed config must be
updated alongside the plugin.

Plugin and core must be rebuilt TOGETHER: `CoreDiagnostics` and `KlineBar` both
grew and `ICore` gained five methods. Acceptable for the same reason 2.6's ABI
change was — `AgamottoStrategy` is shadow-only with no order path compiled in —
and still not a precedent for `mjolnir_core.hpp`, which has one.

### Measured gate cost

Per bar, 6 regimes over a 699-row panel, from `core_integration_driver`:

| host / toolchain | last | worst |
|---|---|---|
| macOS arm64, Apple clang | 25 us | 55 us |
| dev105 x86-64, rocky8 gcc 8.5 (the SHIPPING build) | 236 us | 244 us |

Against the 55 ms the feature panel costs on the same host, the gate is ~0.4% of
the per-bar work, and the full 62-regime stack extrapolates to ~2.5 ms — still
two orders under the panel.

### A FINDING in the reference: `near_ma` alone is `baseline`-shaped

`research_filters` `near_ma` (long) is `(close - mvg1) / mvg1 < 0.02`. The
distance is SIGNED, so every bar trading *below* its 7-bar MA satisfies it
automatically, and the predicate only excludes bars more than 2% ABOVE the MA.
Measured on the parity panels: `r048_long` alone is **all-True on 699/699 rows**
on the BTC and PEPE scenarios — i.e. the unconditional fire-on-every-bar gate
CLAUDE.md removed forever on 2026-06-18, arriving under another name.

The C++ reproduces it exactly (that is what the cell-for-cell equality shows),
so this is a statement about the reference, not about the port. It is not
currently reachable in production: all three deployed regimes containing r048
(`r048_and_r075_long`, `r048_and_r074_long`, `r069_and_r048_and_r074_long`) also
carry an r07x atom and are inert. **It becomes reachable the day PR #532 is
resolved**, at which point `r048_and_r07x` degenerates to the vol-quantile atom
alone. `tests/regime_parity.py` prints it as a FINDING on every run rather than
allowlisting it into silence. Reported, not fixed.

### A FINDING in the tooling: the "artifact leak audit" was not reading the artifact

`obfuscation/audit_public_surface.py` skips `.so`/`.o`/`.a` by suffix and
`continue`s on anything that is not valid UTF-8 — correct for a source-tree
audit, and exactly wrong when the thing being audited is the library.
`sentinel_core/build_linux.sh:75` points it at `build-linux/` and prints
`artifact leak audit: PASS`; that run reads `CMakeCache.txt` and the build logs
and **never opens `libmjolnir_core.so`**. It has been reporting a pass it did
not measure since 2026-07.

A `--binaries` flag was added (printable runs, as `strings(1)`) and agamotto's
`build_linux.sh` uses it. `sentinel_core`'s was deliberately left alone —
turning it on there is a separate change that may turn that build red, and it
should be made by someone who can act on the result. **`libmjolnir_core.so` has
not been audited by this mechanism and should be.**

## Phase 2.6 — the engine wired into the live core

Through 2.5, `engineerFeatures()` existed and **nothing called it**. Every gate
was green throughout, because all of them call the engine directly. 2.6 wires
it into `RealCore` and, just as importantly, makes the fact that it ran
OBSERVABLE — a panel computed and never looked at is how a silently wrong one
survives to Phase 4.

**Where it runs.** In `barReady()`, on the POP, not on the emit. The builder
emits into a queue and the caller drains it, so "a bar completed" and "a bar was
handed over" are different moments; the panel must describe the one the caller
is about to look at.

**The 699 / 700 split is deliberate and is not an off-by-one.** `PANEL_BARS` is
699 because that is what live engineers (`trading.py:443` fetches 700,
`:485` drops the incomplete one), and `engineerFeatures` throws on any other
width — `price_range_pct_q50` is `rolling(700, min_periods=1)` and is therefore
EXPANDING on a shorter frame, so a 700-row panel moves every cell of it.
`warmup_bars` stays at the contract's **700**, one bar more conservative, so the
retained ring is always strictly longer than the slice and there is no width to
get wrong at the boundary. `createCore` now **throws** if `warmup_bars <=
PANEL_BARS`: a core that could never slice a panel would look exactly like a
quiet market.

**A burst gets ONE panel, not four.** A quiet stretch drains several flat bars
out of a single tick (bar rule 4), and all of them are in history before the
first pop returns. The panel ends at the NEWEST retained bar, so pairing it with
an older popped bar is LOOKAHEAD — the kind that flatters a Phase-4 backtest
rather than failing it. Superseded bars are skipped and COUNTED
(`panels_skipped_stale_bar`), never re-sliced.

**ABI change.** `CoreDiagnostics` grew (`panel_rows`, `panel_cols`,
`panel_bar_ts_ms`, `feature_compute_us[_max|_total]`, `panels_computed`,
`panels_skipped_not_warm`, `panels_skipped_stale_bar`, `panel_errors`) and
`ICore` gained three accessors (`panelColumnCode`, `panelLatest`,
`lastPanelError`). **Plugin and core must be rebuilt together.** Acceptable only
because `AgamottoStrategy` is shadow-only with no order path compiled in — there
is no live position a size-mismatched read could mis-manage. Not a precedent for
`mjolnir_core.hpp`, which does have one.

An engine throw is CAUGHT (an exception escaping the SDK's quote handler kills
the process) but never swallowed: the previous panel is **destroyed** so a stale
one cannot be scored, `panel_errors` moves, and the strategy `LOG_ERROR`s with
`lastPanelError()`.

The strategy logs `[AGFEAT]` on **every** bar, panel or no panel — a line only
when a panel was produced would make "not warm yet" and "warm but producing
nothing" the same silence — appends the panel's newest row to
`<logger_file_path>_panel_<rundate>.csv` (codes as keys, `%.17g`, NaN/inf
verbatim), and repeats the shape/cost distribution in the shutdown `[AGDIAG]`.

### Measured per-bar cost

`tests/core_integration_driver.cpp` times the panel on the real path, so this
replaces Phase 1's estimate (which was a TA-Lib microbenchmark plus arithmetic).

| host / toolchain | min | p50 | p95 | max |
|---|---|---|---|---|
| macOS arm64, Apple clang 21 | 3.4 ms | 3.8 ms | 6.4 ms | 7.9 ms |
| dev105 x86-64, rockylinux:8 gcc 8.5 (the SHIPPING build) | 55.0 ms | 55.1 ms | 55.4 ms | 55.5 ms |

The shipping build is ~15x slower than the dev build and that gap is **not**
noise — the rocky8 numbers are flat to three digits across runs. Against a 900 s
bar period 55 ms is 0.006%, so it is a non-issue for scheduling; it is recorded
because it is the number a Phase-4 latency budget must start from, and because
"3.4 ms" measured on a laptop would have been wrong by an order of magnitude.

### The integration gate

The 65-column parity harness cannot see any of the above: it never constructs a
core, never builds a bar, and would pass unchanged if `RealCore` called the
engine on the wrong window, at the wrong moment, or not at all.

```bash
./build-linux/core_integration_driver          # 75 assertions
./build-linux/core_integration_driver --bench 50
```

It drives synthetic ticks through `createCore()` -> `KlineBuilder` ->
`engineerFeatures`, reproducing the REAL boot path including the structural
seam: 699 backfilled bars, attach mid-bucket, the discarded partial, the
quarantine, the one-bucket fill, the splice. It asserts no panel before warm,
exactly 699 x 65 on the first warm bar, the panel stamped with the bar that was
popped, one panel per burst, and that the q80/q90/q95 columns are NaN. It runs
inside `build_linux.sh`, so a core that computes no panel cannot ship.

## Phase 2.7 — live reconciliation against the Python bot

Everything above compares the two sides on the SAME bars. 2.7 compares the two
CHAINS:

```
SHM tick feed ---> KlineBuilder ---> engineerFeatures ---> panel CSV
Binance WS kline -> knull bridge --> research.py -------> debug_features CSV
```

```bash
python tests/live_reconcile.py --ssh-host hydra --bars 3
```

**The method is the deliverable.** Two chains can disagree for two unrelated
reasons, and a report that gives one number per column cannot tell them apart:

* **input divergence** — both sides computed CORRECT features over DIFFERENT
  bars. The bot's WS-fed klines drift from Binance's official klines by one tick
  intermittently (measured 2026-08-19).
* **engine divergence** — different features over the SAME bars. Only this is a
  port bug.

So the script reports **inputs first**, using Binance's own klines
(`fetch_binance_klines.py`'s CSV, which the strategy already maintains) as the
**arbiter** — "C++ != bot" alone says nothing about which one is wrong — then
compares features and ATTRIBUTES every disagreement. A column that differs only
on bars whose inputs already differed is listed under the FEED, not the engine.
Input divergence is reported loudly and is never fatal: it is a property of the
two feeds, and an exit code that goes red on something this port cannot fix is
an exit code everyone learns to ignore.

The bot's CSV carries REAL names and the engine emits CODES; the mapping is
`dc/obfuscation/map.json`, applied on the private side, with the same four
uncoded exceptions the parity harness declares (`close`, `mvg1/2/3`). The bot's
11 non-feature columns (7 lookahead targets, `year`/`month`, `symbol`,
`timestamp`) are DECLARED, so they read as expected absences rather than as
missing columns. Coverage measured: **65 engine columns, 65 bot feature columns,
65 compared, 0 engine-only, 0 bot-only, 0 unmapped.**

### A per-bar verdict is NOT enough — STEP 1b and the counterfactual

The first live run made the trap obvious. On 2026-08-19 15:30Z the C++ bar was
**9/9 columns exact** against Binance and the bot's close was exact too, and 23
of 65 columns still disagreed. Not an engine bug: almost every column here is
ROLLING — the TA-Lib block is 14 bars deep, the scale-free transforms 20, `mvg3`
99, `price_range_pct_q50` a 700-bar median — so the *previous* bar being wrong
poisons ~20 bars of everything that reads it, on bars whose own inputs are
perfect. The 23 that moved were exactly the ones reading high/low/volume/trades;
the 42 that read only `close` agreed to 1e-9.

A per-bar cleanliness test would have called that a port bug. So the script adds:

* **STEP 1b** — checks EVERY live-built bar against Binance and names the dirty
  ones, because a rolling column reads many bars, not one.
* **STEP 5, the counterfactual** — runs the SAME engine binary over the PURE
  BINANCE 699-bar window ending at each bar and diffs THAT against the bot's row.
  Agreement there plus disagreement live means the ENGINE is right and the BARS
  differ; disagreement there is a real port bug on inputs nothing can be blamed
  for. It is the authoritative verdict and it drives the exit code.

This is a stronger claim than `tests/feature_parity.py` makes. That harness diffs
the engine against `research.py` run OFFLINE by the harness on synthetic panels.
STEP 5 diffs the engine against what the PRODUCTION BOT actually emitted, in
production, for that bar.

### Measured, 2026-08-19, BINANCE_PERP_BTC_USDT, 3 shared 15m bars

| step | result |
|---|---|
| STEP 1 inputs, C++ vs Binance | 15:30 and 15:45 **9/9 exact** (3.0e-12 / 2.1e-12); 15:15 DIRTY |
| STEP 1 inputs, bot close vs Binance | **exact on all 3 bars** (0.00e+00) |
| STEP 2 coverage | 65 compared, 0 engine-only, 0 bot-only |
| STEP 3 live panel vs bot | 42 agree to 1e-9; 23 diverge, all reading high/low/volume/trades |
| STEP 4 `buy_pressure` | agrees to **2.2e-14**, `aggressor_source=exact` — the maker flag IS populated |
| STEP 4 q80/q90/q95 | both sides NaN, as pinned |
| **STEP 5 counterfactual** | **3/3 bars: ALL 65 columns agree with the LIVE BOT to rel <= 1e-9** |
| panel shape / cost, live | 699 x 65 every bar; 47.8 / 48.1 / 48.9 ms |

**The engine is exonerated against production.** Every live difference traces to
the bar layer, and specifically to the finding below.

### FINDING (reported, not fixed): the SHM tick ring drops events in a burst

The two dirty bars fell inside a 65k -> 70k move. Against Binance's own klines:

| bar (UTC) | C++ trades | Binance trades | avg rate | trades lost | high |
|---|---|---|---|---|---|
| 15:00 | 269,078 | 271,658 | 302/s | **-0.95%** | exact |
| 15:15 | 495,819 | 522,121 | 580/s | **-5.04%** | 69888 vs **70450** |
| 15:30 | 450,818 | 450,818 | 501/s | 0.00 | exact |
| 15:45 | 201,344 | 201,344 | 224/s | 0.00 | exact |

It is not average rate — 15:30 at 501/s was bit-exact while 15:15 at 580/s lost
5% — it is BURST rate: the 15:15 bar contains the spike to 70450, which the
tick-built bar **missed entirely** (562 points, 0.80%, low). That is consistent
with the SHM ring (`container_size: 1024`) overflowing while the consumer is
between reads; a dropped slot is invisible to the consumer, so nothing in the
core can count it — `unclassified` was 0 and `aggressor_source` was `exact`
throughout. Volume, quote_volume, trades and both taker_buy_\* run LOW together
and roughly proportionally, which is why `buy_pressure` (a RATIO of two of them)
survives at 4.9e-04 while `vol_ratio` and `mfi` do not.

Consequence for later phases: a Phase-4 signal computed off tick-built bars will
differ from research on exactly the bars where it matters most — the violent
ones. Reported here; the fix belongs in the feed/ring layer, not in this core.

## Phase 2.1 — the numeric primitives

`src/feature_engine.{hpp,cpp}` holds the `Table` column panel and `pdops`, the
pandas-equivalent primitives every feature column is built from: `diff`,
`diffN`, `shift`, `pctChange`, `rollSum`, `rollMean`, `rollStd`, `rollVar`,
`rollSkew`, `rollKurt`, `rollCorr`, `rollQuantile`, `rollQuantiles`.

They are transcribed from pandas 2.3.3
`pandas/_libs/window/aggregations.pyx` — **read, not inferred**. Three details
in there are invisible from the docs and each is worth a wrong column:

- **`rolling().corr()` pairwise-masks BOTH series FIRST.** `rolling.py`
  `prep_binary` computes `X = x + 0*y; Y = y + 0*x` before any rolling stat, so
  each mean AND each variance is over the *pairwise* observations. Taking each
  series' own non-NaN values gives a plausible number that is a different
  statistic — on `acf_lag1` (`hist_return` vs `hist_return.shift(1)`, whose
  masks differ by one row at every hole) it is off by 100%.
- **`roll_skew`/`roll_kurt` pre-centre the WHOLE array** by `round(nanmean)`,
  guarded on `nanmin - mean > -1e5` for skew and `> -1e4` for kurt. Different
  constants, deliberately. Without it the raw 4th moment of a price-scale
  series cancels ~1e12 to 1.
- **The constant-window guards.** pandas forces `std -> 0`, `skew -> 0`,
  `kurt -> -3`, `sum -> prev*nobs`, `mean -> prev` when every observation in
  the window is equal, instead of returning the floating-point residue. Skip
  them and a flat-bar window produces a variance of ~1e-17, which flips the
  sign of anything that divides by it.

`min_periods` counts **non-NaN observations**, never rows — the difference
between `rolling(700, min_periods=1)` (research.py:363) and
`rolling(700, min_periods=700)` (research.py:373) is the entire fail-closed
warmup behaviour of the `high_vol_q*` cutoffs.

`EPS` is **1e-8, written inline per expression** (research.py:362, :378, :475).
mjolnir's shared `1e-10` must not be reused here, and there is deliberately no
shared constant to tempt anyone.

### The gate

```bash
tests/run_pdops_parity.sh          # golden on the host, diff inside the image
```

`tests/pdops_golden.py` emits ~2000 rows of 15m-shaped data with injected NaN
holes (singletons, runs shorter than `w`, a run LONGER than `w`, a trailing
hole, constant runs, and deliberately DIFFERENT masks on the two corr inputs)
plus the pandas answer for every primitive at every `min_periods` the reference
actually uses (1, `w`, 700). The **golden CSV header is the spec** —
`rollskew|ret|14|14` — and `tests/pdops_parity_driver.cpp` parses it and
dispatches, so the two sides cannot drift and an unparseable spec is a hard
failure rather than a skipped column.

- **Gated:** identical NaN masks (zero tolerance) *and* max rel diff ≤ 1e-12.
- **`rollskew`/`rollkurt` gated on `|a-b| <= 1e-12 * max(|b|, 1)`.** They are
  dimensionless statistics that legitimately pass through zero — a rolling skew
  of returns sits at ~2e-3 — so a pure relative metric divides a 1e-14
  agreement by a near-zero denominator and reports 1e-11. Measured agreement on
  the shapes the reference actually feeds them is **1.7e-14 absolute**.
- **`NEG_` specs are inverted:** the golden holds a deliberately WRONG
  computation (population moments, nearest-rank quantile) and the driver fails
  if the C++ *matches* it. Population skew is 12.4% off at n=14 and population
  kurt 1050x off; without these a positive-only harness would bless either.
- **`PROBE_` specs skip the VALUE gate; their NaN masks are still gated at zero
  tolerance.** Two groups. (a) Price-scale skew/kurt, which the reference never
  computes: `pdops_golden.py` prints the reason — pandas cannot reproduce
  *itself* there, differing by 5.6e-8 on the same 14-bar window when the frame
  start moves. (b) The `flat` / `nearflat` **zero-variance regression series**
  (below), whose values ride on a rolling variance whose last bit is not
  portable, but whose masks are exactly the thing being pinned.

### The zero-variance regression series, and the FP-contraction defect

`flat` (piecewise-constant blocks) against `wobble`, plus `nearflat`
(near-constant blocks), exist to hit — deliberately, not by luck of the random
data — the two mask predicates that a compiler can silently change:

1. **`rollCorr` over a constant window.** Variance is exactly 0, so the result
   is decided purely by whether the numerator cancelled to exactly 0.0 (→ NaN)
   or landed 1 ULP off (→ ±inf). pandas produces **both**, and the generator
   asserts the golden contains both (~620 zero-denominator rows per spec, split
   roughly 55/45) rather than passing vacuously. Apple clang on arm64 defaults
   to `-ffp-contract=on` and fused `mXY - mX*mY` into one `fnmsub`, flipping
   NaN to ±inf; gcc 8.5 on the baseline-x86-64 target has no FMA instruction to
   contract into, so the same source and golden were **RED on macOS and GREEN
   on Linux**. Fixed at source with `pdRound()` (a volatile round-trip) at
   every mask-deciding site — see the banner in `src/feature_engine.cpp`.
   There is **no** "zero variance → NaN" shortcut: pandas emits ±inf on rows
   where the numerator is 1 ULP from zero, so only its own arithmetic is right.
2. **`rollStd` over a near-constant window.** The streaming Welford residue
   goes negative (measured −1.4e-10 … −3.5e-9 on 909 of 2000 rows) and
   `calc_var` has **no** clamp — but `Rolling.std` is `zsqrt`
   (`common.py:149-161`), which maps negative → **0.0**, not NaN. A bare
   `std::sqrt` gives NaN on every one of those rows. Gateable because only the
   *sign* is consulted and the sign survives an FMA (0 sign changes in 1965
   rows).

Near-constant `skew`/`kurt` are **absent from the suite, not PROBE**: their
`B <= 1e-14` guard is pure cancellation residue there and pandas cannot
reproduce that mask against *itself* — moving only the frame start flips
101–113 rows. Gating it would pin a coin flip. Do not add them back.

**`roll_var`'s last bit is not portable.** `add_var` is Cython → C, so the
wheel's compiler decides: the pandas 2.3.3 **arm64** wheel contracts its
`ssqdm` update into an FMA (a bit-exact Python replica matches it on 0/1965
rows with separate rounding and 1965/1965 with `math.fma`), while a
baseline-x86-64 wheel cannot. The port takes the form the reference *source* is
written in and pins it with `pdRound`, so clang and gcc agree with **each
other**; the residual ~1 ULP against any given wheel is what the 1e-12 gate
absorbs, and the specs where conditioning amplifies it past that are PROBE.

That last measurement is the honest ceiling on this whole layer: pandas'
rolling var/skew/kurt stream, and skew/kurt additionally centre on a
whole-array constant, so they are **not window-local**. Live runs a ~700-bar
panel while research runs years of bars, so the last digits cannot agree
however the C++ is written. On the return-scale columns the reference actually
computes it is ~1e-14; it is not 1e-16 anywhere.

**Reading these CSVs back from Python needs
`pd.read_csv(..., float_precision="round_trip")`.** The default C parser is not
exactly rounded and shifts values by ~1 ULP, which surfaces as a 1e-11 "parity
failure" that is entirely the parser. `%.17g` on the way out and `strtod` on
the way in are both exact.

## Phase 2.2 — the OHLC / returns / MA / volume columns

`agamotto::engineerFeatures(const RawBars&)` emits **exactly 24 columns** —
research.py `engineer_features` lines 361-380, 382-387, 461-468 and 470-499,
and nothing else:

| block | columns |
|---|---|
| passthrough | `close` |
| OHLC (:361-380) | `price_range`, `price_range_pct`, `price_range_pct_q50`, `price_range_pct_q80/q90/q95`, `open_close_diff`, `open_close_pct`, `high_open_pct`, `low_open_pct` |
| returns (:382-387) | `ret_lag1`, `ret_lag2`, `ret_lag3` |
| MAs (:461-468) | `mvg1`, `mvg2`, `mvg3` = `close.rolling({7,25,99}, min_periods=1).mean()` |
| volume (:470-499) | `vol_ratio`, `vol_ret_lag1/2/3`, `quote_vol_ratio`, `buy_pressure`, `trade_intensity` |

Column keys are the obfuscation codes, except `close` and `mvg1/2/3`, for which
`dc/obfuscation/map.json` **has no entry** — see "The mvg gap" below.

### PANEL_BARS = 699 is a correctness parameter, not a buffer size

`trading.py:443` `load_data(limit=700)` → `:480` `tail(limit)` → `:485`
`iloc[:-1]` (drop the incomplete bar) = **699 closed bars**, always.

`price_range_pct_q50` is `rolling(700, min_periods=1)`, so below 700 rows it is
an **expanding** median: the value at row *i* is the median of rows 0..*i*.
Feed 700 rows instead of 699 and the numbers change; feed 1000 and they all do.
`engineerFeatures` therefore *rejects* any other width rather than producing
plausible numbers live would never see, and the harness reads `PANEL_BARS` out
of the header rather than retyping it.

### q80/q90/q95 are ALL-NaN, deliberately, and the gate pins it

They are `rolling(VOL_Q_WINDOW=700, min_periods=VOL_Q_WINDOW)`
(research.py:371-376). At 699 rows the min_periods is never met, so all three
columns are NaN on every row. That is **live behaviour under an open production
finding** — marvel PR #532,
`docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md`, which measures
that **53 of 62 deployed regimes cannot fire live** because `x > NaN` is False.
The port reproduces it and must not "fix" it: lowering min_periods or widening
the panel would start those regimes firing against models never trained on a
firing regime, and the port would look like the cause. `feature_parity.py`
asserts the all-NaN property on **both** sides, so it is pinned rather than
incidental.

### Two things this file does NOT do

- **No target column, ever.** `return`, `return_long`, `return_short`,
  `return_long_raw`, `return_short_raw`, `return_dip`, `return_rip` and the
  `ret_2bar*` family all read `shift(-1)`/`shift(-2)`: lookahead target
  construction for the trainer, not features. They are listed in
  `feature_parity.py` `EXPECTED_ABSENT_TARGETS`, which fails both if one
  appears in the engine's output and if one disappears from the reference (a
  stale declaration).
- **No inf/NaN sanitisation.** mjolnir replaces non-finite cells with 0.0
  panel-wide; agamotto's only fill is `X.fillna(0.0)` on the selected model
  columns of the single scored row (`trading.py:700`), after the gate has run,
  and inf is never touched. NaN/inf propagate out of `engineerFeatures`
  untouched.

### The epsilons

`+1e-8`, written **inline per expression** at all eight sites
(research.py:362, :378, :379, :380, :475, :485, :491, :498). There is no shared
`EPS` constant in `feature_engine.cpp` on purpose — mjolnir's is `1e-10`, and a
single named constant is one "tidy-up" away from silently re-scaling every
ratio feature in one of the two algos.

The epsilon is **absolute**, so its effect is relative to the symbol's price:
1.6e-13 on BTC (~64000), **2.2e-6 on 1000PEPE (~0.0045)** — a
sixth-significant-figure move in `price_range_pct`, a top-5 IC-selected
feature. Measured with a mutant driver built with mjolnir's `1e-10`: the
BTC scenario **passes**, and the PEPE scenario fails on 7 columns with max rel
diff **2.212e-06**. A BTC-only harness cannot tell `+1e-8` from `+0`.

### The gate

```bash
tests/run_feature_parity.sh            # macOS / clang -O2
tests/run_feature_parity.sh --linux    # rocky8 / gcc 8.5, driver in the build image
tests/run_feature_parity.sh --both
```

`tests/feature_parity.py` constructs a real `AgamottoResearch`, sets `.raw`
directly (bypassing `load()`, which only reads CSVs off disk) and calls
`engineer_features()` — **the reference is not reimplemented anywhere in the
harness**, so a change in research.py is visible to the gate. It then pipes the
same raw panel to `tests/feature_parity_driver` and compares.

Three scenarios, 699 rows each: BTC-like (~64000), 1000PEPE-like (~0.0045), and
BTC-like with injected NaN holes (singleton, sub-window run, super-window run,
trailing), a zero-volume bar (→ `+inf` in `vol_ret_lag*`), a zero
`quote_volume` bar and a flat bar.

**Cells are classified `finite / NaN / +inf / -inf` and the classifications
must match EXACTLY before a single value is diffed.** This is the one place the
harness deliberately departs from `sentinel_core/tests/feature_parity.py`,
which does `np.where(np.isfinite(x), x, 0.0)` on both sides first — correct for
mjolnir, which sanitises, and wrong here: under that rule an engine returning
0.0 where pandas returns NaN would PASS, and a NaN that fails to propagate is
exactly what turns a regime that cannot fire into one that fires on every bar.
Classifying (rather than just `isnan`) also stops `+inf` on one side and NaN on
the other from cancelling out as "both non-finite". Finite cells are then gated
at **1e-9 relative** (achieved: 0 differing cells, all three scenarios, both
toolchains).

Coverage is pinned by **set equality**, not a floor: the C++ must emit exactly
the declared 24 and each must exist in the reference panel. sentinel_core's
harness records a refactor that once shrank its comparison from 155 columns to
55 while still printing PASS, and a `>=` floor alone would still bless an
engine emitting the right *number* of the wrong columns.

**Both toolchains, always.** Stage 2.1 was GREEN on gcc 8.5/x86-64 and RED on
clang/arm64 from identical source (FMA contraction flipped a NaN mask in
`rollCorr`). A single-platform pass is not evidence about the other platform.

Verified 2026-08-19 — negative controls, each built as a mutant driver and each
turning the gate red:

| mutant | caught by |
|---|---|
| `1e-8` → `1e-10` (mjolnir's epsilon) | PEPE scenario, 7 columns, max rel 2.212e-06 (**BTC scenario still passes**) |
| non-finite → 0.0 (mjolnir's sanitiser) | 9 columns on the NaN/inf classification + the all-NaN q80/q90/q95 assertion |
| `min_periods` 700 → 1 on the vol quantiles | the all-NaN assertion + 3 columns on classification |

### Reference quirks reproduced, and one declared divergence

Reproduced faithfully, because they are what production computes:

- **`trade_intensity` lives inside the `quote_volume` branch**
  (research.py:483-499). A feed carrying `number_of_trades` but no
  `quote_volume` therefore emits no `trade_intensity` at all — the nesting is
  almost certainly accidental, but the port matches it.
- **`price_range_pct_q50` writes the window as a literal `700`**
  (research.py:363) while q80/q90/q95 use `VOL_Q_WINDOW` (:373), even though
  the `VOL_Q_WINDOW` comment says the point is that "the new cutoffs and the
  incumbent median share one lookback". Changing `VOL_Q_WINDOW` would silently
  *not* move q50. Latent, not currently wrong (both are 700).

Declared divergence:

- **`df.get(open_col, close)`** (research.py:356-358) silently substitutes
  `close` for a missing `open`/`high`/`low`, which would make
  `high_open_pct`/`low_open_pct` exact zeros and `price_range_pct` zero, with
  nothing logged. It is unreachable through research.py's own loader (`:203`
  raises on a missing required column) and through the kline builder, but it is
  the `cfg.get(K, X)` shape CLAUDE.md bans. `engineerFeatures` **throws**
  instead. Reported, not fixed, in research.py.

### The mvg gap

`mvg1`/`mvg2`/`mvg3` are **not in `dc/obfuscation/map.json`**, so both the
vertical panel and this core carry their real names. They are not passthrough
market-data fields: they are engineered features, `MVG_DEPENDENT_FILTERS` gates
regimes on them (`research_filters`), and the codes header's preamble justifies
its uncoded set as "OHLCV, timestamps, depth_*, ofi_*, bids_*/asks_*" — which
these are not. The names leak little on their own; recorded here as a **gap in
the obfuscation map, not fixed as part of this stage** (adding a code is a
map.json change that has to move the Python packages and the parquet columns in
lockstep).

### Feature/regime codes

`src/codes_generated.hpp` is emitted by the committed generator:

```bash
python ../obfuscation/gen_codes_hpp.py --namespace agamotto \
    --out src/codes_generated.hpp
python ../obfuscation/gen_codes_hpp.py --namespace mjolnir \
    --out ../sentinel_core/src/codes_generated.hpp --check
```

The header has claimed "GENERATED … do not edit by hand" since 2026-07-30 with
no generator committed; `--check` is what keeps that claim true from now on.
It currently reports mjolnir's header as **13 entries stale** (`f101`-`f110`,
`r073`-`r075` — the scale-free twins and the vol-quantile atoms added to
`map.json` afterwards), with no other difference. Regenerating it is a
sentinel_core change, not an agamotto one, so it has been left alone.

## Phase 2.3 — the TA-Lib indicator block

`src/talib_block.cpp` emits **30 columns**: 25 TA-Lib calls producing 29, plus
`parkinson_vol`, which is *not* a TA-Lib call but lives inside the reference's
`try:` and so shares its fate. research.py:501-553, transcribed call for call.

| block | columns |
|---|---|
| momentum | `rsi`(14), `rsi_7`, `rsi_28`, `mom`(10), `roc`(10), `cmo`(14), `trix`(30), `willr`(14) |
| MACD(12,26,9) | `macd`, `macdhist` — **the signal line is discarded** |
| stochastics | `stoch_k`, `stoch_d` (STOCH 5,3,SMA,3,SMA); `stochrsi_k`, `stochrsi_d` (STOCHRSI 14,5,3,SMA) |
| directional | `adx`(14), `dx`(14), `plus_di`(14), `minus_di`(14) |
| oscillators | `cci`(14), `ultosc`(7,14,28), `mfi`(14), `bop`(o,h,l,c) |
| volume | `obv` = `OBV(c,v).diff(14).fillna(0.0)`, `ad` = `AD(h,l,c,v).diff(14).fillna(0.0)` |
| volatility | `atr`(14), `natr`(14), `parkinson_vol`, `bb_upper`/`bb_lower` (BBANDS **20**,2,2,SMA — **the middle band is discarded**) |
| trend | `sar`(0.02, 0.2) |

It calls **libta-lib directly** — the same C library the reference's Cython
wrapper calls — so the values are identical by construction rather than
reimplemented. The version is pinned to **0.6.4** in two places that
`tests/feature_parity.py` reads and cross-checks: `CMakeLists.txt`
`TALIB_PINNED_VERSION` (the library and the host driver) and
`../sentinel_core/Dockerfile.build` `ARG TALIB_VERSION` (the build image, i.e.
the linux driver). If those two drift, the two toolchain legs of the gate grade
against different math while both print PASS.

### This is NOT sentinel_core's talib_block with the namespace changed

`place()`, `init()` and the `TA_RetCode` -> all-NaN-on-failure shape are reused
verbatim, and so are the call lines whose parameters genuinely match. **Five
things differ, and copying mjolnir is wrong for each:**

| | mjolnir | agamotto | why it matters |
|---|---|---|---|
| BBANDS `timeperiod` | 5 | **20** (research.py:549) | different band entirely; `bb_upper`/`bb_lower` feed the `bb_rebound` predicate. This is the stage's negative control. |
| `f085` (`std`) | `TA_STDDEV(close, 14)` | **not emitted here** | agamotto's `std` is `hist_return.rolling(14).std()` (research.py:569) — a *different quantity under the same code* (return-scale vs price-scale). Stage 2.4. |
| volume into TA-Lib | `volume.fillna(0)` | **RAW** (research.py:535) | a NaN volume bar is *meant* to poison OBV/AD/MFI from that bar on. |
| TRIX(30), ULTOSC(7,14,28), STOCHRSI(14,5,3,SMA), BOP | absent | present | new code. |
| `open` input | not in the signature | threaded through | BOP needs it. |

### The wrapper skips leading NaNs — and so must the port

The reference is `talib.RSI(arr, ...)`, i.e. the **wrapper**, not the bare C
function. Every generated wrapper computes `begidx = check_begidx1..4(inputs)` —
the first index at which *every* input is non-NaN, i.e. the **MAX** over the
inputs' individual first-valid indices — calls the C function on `data + begidx`
over `endIdx = n - begidx - 1`, and writes the payload at `begidx + lookback`.
Interior NaNs are **not** skipped and **do** poison. Measured on the pinned
wrapper (0.6.8 / library 0.6.4):

| input | first valid output index |
|---|---|
| `RSI(x, 14)` | 14 |
| `RSI(x with 5 leading NaN, 14)` | **19** — skipped, not poisoned |
| `STOCH` with leads (h=3, l=5, c=2) | **13** = max(3,5,2) + 8 |
| `BOP` with leads (o=4, h=2, l=1, c=3) | **4** = max(4,2,1,3) + 0 |
| `RSI` with ONE interior NaN at row 40 | 14, then **zero** non-NaN cells after row 40 |

So `compute()` derives six per-call-site `begidx` values (`bC`, `bHL`, `bHLC`,
`bOHLC`, `bCV`, `bHLCV`) and places each result at `begidx + outBegIdx`. The
fourth parity scenario exists solely to make that observable: it gives each
column a **different** leading-NaN count (o5 h3 l7 c4 v9), which pins every
max() independently. Without it every `begidx` is 0 and a port that ignored the
skip entirely would pass.

### PANEL_BARS = 699 is what makes the unstable period match

RSI's Wilder smoothing, ADX, TRIX's triple EMA and SAR are **recursive** — they
never forget their start, so their values depend on where the input begins. The
port makes **no convergence correction**, and none is needed *for the gate*:
`engineerFeatures` refuses any panel that is not 699 rows and the harness drives
the reference over the *same* 699 rows, so the unstable period is identical by
construction. Matching the window **is** the mechanism.

What convergence does and does not buy is worth stating precisely, because the
figure that was circulating ("0.000e+00 on SAR/ADX/TRIX, <= 1.5e-11 on the
rest") holds only at the LAST row. Measured 2026-08-19, ta-lib 0.6.4, on a
5000-bar synthetic BTC-scale series, full history vs its trailing 699 rows:

- **At the last row** — the only row live ever scores — `rsi`, `adx`, `dx`,
  `trix`, `sar`, `macd`, `cmo`, `atr` and `natr` agree **exactly (0.0)**.
  `rsi_28` does **not**: 5.2e-11 absolute, 9.8e-13 relative. It has the slowest
  Wilder decay in the block, and it is the one column where a 699-bar live panel
  does not reproduce research's years-long one bit for bit. Still three orders
  under the 1e-9 gate, but it is not zero and should not be described as zero.
- **Over the whole 699-row window** the disagreement is large: max absolute
  `sar` **7.8e+02**, `macd` **4.0e+01**, `atr` **1.5e+01**, `adx` **9.3**,
  `cmo` **4.8**, `rsi` **2.4**. Exact agreement is only reached from row **414**
  (`macd`) through **567** (`trix`); `sar` recovers by row 22 and `rsi_28` never
  does within 699.

So a harness that fed the reference more history than the C++ would be comparing
numbers that differ **in the first digit** over the first half of the panel.
That is the whole reason `PANEL_BARS` is enforced on both sides rather than
treated as a buffer size.

### Two transcription traps

- **`obv`/`ad` are `.diff(14).fillna(0.0)`**, not the raw indicator. Rows 0..13
  are **0.0, not NaN** — and the fill also flattens the *poisoned tail* after a
  NaN volume bar to 0.0, which a head-only fill would leave as NaN. Both change
  the model input.
- **`parkinson_vol`** is `sqrt(1/(4 ln2) * log(high/low)**2)` **elementwise**,
  and only then `.rolling(14).mean()`. Three details decide the last bits:
  divide *then* log (`log(h) - log(l)` is a different double); numpy's `**2` on
  a float64 array is `x*x`, not `pow(x, 2.0)`; and the scalar `1/(4 ln2)` is
  evaluated once and multiplied onto the square. `.rolling(14)` takes its
  default `min_periods` = 14, and pandas maps ±inf to NaN first — `pdops::rollMean`
  reproduces both.

### The gate

Same harness, extended:

```bash
tests/run_feature_parity.sh              # macOS / clang -O2
tests/run_feature_parity.sh --linux      # rocky8 / gcc 8.5, driver in the build image
tests/run_feature_parity.sh --both
tests/run_feature_parity.sh --negative   # negative control, exit code INVERTED
```

The host driver now links ta-lib, so it needs the **pinned 0.6.4** built locally
(homebrew's 0.7.1 is refused — different indicator internals, silently). The
script prints the recipe and verifies the version from `ta-lib.pc` before
building; override the location with `AGAMOTTO_TALIB_PREFIX`.

Two checks are specific to this stage:

- **A first-valid-index assertion, per column.** Every TA-Lib output is
  *compacted*; place it at the wrong offset and the entire column is shifted by
  its lookback. A value diff of the overlapping region can still pass on a
  slow-moving indicator, and the NaN classification only moves if the head
  *length* changed. The first non-NaN row index moves always, so it is compared
  per column and printed as a table.
- **A TA-Lib version gate, plus proof the reference actually ran TA-Lib.**
  research.py:501-554 wraps the whole block in `except Exception` and only logs
  a warning, so a reference *without* TA-Lib silently returns a panel **missing
  these 30 columns** rather than failing — graded naively that reads as a pass
  over the remaining 24. The harness refuses to run unless
  `talib.__ta_version__` equals the pin, and asserts the 29 TA-Lib columns are
  **present** in the reference panel before comparing anything.

Verified 2026-08-19, **54 columns x 699 rows x 4 scenarios, both toolchains**,
0 differing cells at 1e-9 relative and identical first-valid-index tables.

| mutant | caught by |
|---|---|
| BBANDS `timeperiod` 20 -> 5 (mjolnir's value) | all 4 scenarios: `bb_upper`/`bb_lower` on first-valid-index (ref 19, cpp 4) **and** on NaN classification |

### Reference behaviour NOT reproduced — one declared divergence

research.py:554 catches **every** exception from the block and turns it into a
`logger.warning`, so a reference whose TA-Lib is missing, broken, or handed an
all-NaN column silently produces a panel that is simply **short 30 columns**.
`talib_block::compute` **throws** on an entirely-NaN input column instead. A
live core must not emit a panel whose missing columns are indistinguishable from
a dead feed. Reported, not fixed, in research.py.

A second, narrower divergence: on a `TA_RetCode` failure this file emits an
all-NaN column for that one indicator and carries on (mjolnir's pattern), where
the reference would drop **all 30**. Unreachable in practice — every parameter
here is a compile-time constant and TA-Lib returns `TA_SUCCESS` with
`outNBElement = 0` for a too-short range rather than an error — but recorded so
the difference is deliberate rather than assumed.

## Phase 2.4 — the rolling return-moment stats (4 columns)

`research.py:557-593`. `STATS_WINDOW` is **14**, carried explicitly in the
deployed arm's `setting.json`, and research.py:564 *raises* below 4 rather than
falling back. All four are computed on `hist_return`
(`close.pct_change(fill_method=None)`), **never on `close`**:

| column | code | reference | min_periods |
|---|---|---|---|
| `std` | `f085` | `hist_return.rolling(14).std()` | 14 |
| `skew` | `f083` | `.rolling(14).skew()` — SAMPLE (bias-corrected) G1 | 14 |
| `kurt` | `f040` | `.rolling(14).kurt()` — SAMPLE EXCESS G2 | 14 |
| `acf_lag1` | `f001` | `.rolling(13).corr(shift(1)).fillna(0.0)` — note **13**, and the fill runs *after* | 13 |

Nothing new is implemented here: all four call the stage-2.1 pdops primitives,
which are already pandas-exact and FMA-hardened.

### `f085` is a homonym across the two algos

This is the one thing in stage 2.4 that is not arithmetic.
`sentinel_core/src/talib_block.cpp:133` emits `codes::F_STD` — the **same code
`f085`** — from `TA_STDDEV(close, 14)`: the standard deviation of **price**,
~1e4 on BTC. agamotto's `f085` is the standard deviation of **returns**,
~4e-3. Same code, same window, quantities seven orders of magnitude apart.
Stage 2.3 therefore deliberately left `f085` out of `talib_block.cpp`; it is
emitted by the stage-2.4 block and **nowhere else in this core**. The two
definitions must never coexist in one binary: a panel carrying a price-scale
`f085` would be scored by weights trained on a return-scale one and nothing
would raise.

### The panel-start caveat did not bite

`feature_engine.hpp`'s stage-2.1 banner warns that a streaming accumulator plus
a whole-array centring constant makes skew/kurt depend on where the panel
*starts*, and that pandas disagrees with **itself** by up to 5.6e-8 on a
price-scale 14-bar kurt when the frame start moves. That is a statement about
comparing a 699-row frame against a multi-year one. This gate does not do that
— both sides see the same 699 rows — so all four columns clear the ordinary
**1e-9 relative** gate on every scenario and both toolchains, with **no probe
tier and no widened tolerance**. The 1e-12/PROBE treatment in
`tests/pdops_golden.py` grades a different comparison and stays where it is.

## Phase 2.5 — the scale-free level transforms (7 columns)

`agamotto_pkg/src/agamotto/features_scalefree.py:113-129`, called from
`research.py:639-646` with `window=20` and **`obv_is_cumulative=False`**. These
consume stage-2.3 outputs, so ordering matters; the sources are read back out of
the `Table` rather than recomputed, because agamotto's BBANDS is
`timeperiod=20` while mjolnir's is the talib default 5 and a recomputation here
would silently impose one engine's parameters on the other.

| column | code | reference |
|---|---|---|
| `sar_dist` | `f107` | `(close - sar) / close` |
| `bb_pctb` | `f102` | `(close - bb_lower) / (bb_upper - bb_lower)` |
| `bb_width` | `f103` | `(bb_upper - bb_lower) / close` |
| `macd_norm` | `f104` | `macd / close` |
| `macdhist_norm` | `f105` | `macdhist / close` |
| `obv_slope` | `f106` | `obv / volume.rolling(20, min_periods=20).sum()` |
| `ad_slope` | `f101` | `ad / volume.rolling(20, min_periods=20).sum()` |

**`obv_is_cumulative=False` is load-bearing.** `obv`/`ad` arrive **already
differenced** (`obv_raw.diff(14)`, research.py:538-539), so `_flow` must pass
them through untouched and only the *units* get normalised. Passing `True`
takes a second difference, and that failure is silent: on a steady flow it
returns ~0 on every row while still looking like a valid feature.

### `_safe` is two steps, and both are gated

```python
out = num / den.replace(0.0, np.nan)          # step 1
return out.replace([np.inf, -np.inf], np.nan) # step 2
```

Step 1 removes an **exactly-zero** denominator before the division (pandas'
`replace` matches with `==`, so `-0.0` is caught too). Step 2 catches an
infinite numerator or an **overflow** from a finite numerator over a
tiny-but-nonzero denominator — a *different* case. `if (den == 0) return NaN;`
implements step 1 only and is not equivalent.

On ordinary market-shaped data **neither branch ever fires**, so a harness built
only from the stage-2.2/2.3 scenarios would grade `_safe` on its pass-through
arm alone — the same trap stage 2.3 found when all three original scenarios
turned out to have `begidx == 0`. A fifth scenario therefore constructs the
branches, and the branch-hit counts are **asserted**, not hoped for:

| column | `den == 0` | of those, DISCRIMINATING | step-2 overflow |
|---|---|---|---|
| `sar_dist` | 1 | 1 | 1 |
| `bb_pctb` | 7 | 7 | 0 |
| `bb_width` | 1 | 1 | 1 |
| `macd_norm` | 1 | 1 | 1 |
| `macdhist_norm` | 1 | 1 | 1 |
| `obv_slope` | 6 | **0** | 0 |
| `ad_slope` | 6 | **0** | 0 |
| **total** | **23** | **11** | **4** |

*Discriminating* means omitting step 1 would change the answer — `den == 0` with
`num == 0` gives NaN either way. Two results there were measured, not assumed:

- `bb_pctb` **does** discriminate, against the obvious argument. A flat 20-bar
  window makes TA-Lib's STDDEV clamp to exactly 0, so `bb_upper == bb_lower`;
  the naive expectation is that `close == SMA20 == bb_lower` makes the numerator
  zero too. It does not — TA-Lib's SMA is a running sum divided by 20 and rounds
  **8.00355e-11** away from `close`, so without step 1 those seven cells would
  be `-inf`, not NaN.
- `obv_slope`/`ad_slope` **cannot** discriminate, structurally. Volume is
  non-negative, so a 20-bar sum of zero forces all 20 bars to zero, which leaves
  OBV and AD constant, which makes their `.diff(14)` — a 14-bar window nested
  *inside* the 20-bar one — exactly zero. 0/0 either way. Reported rather than
  counted as coverage it does not provide.

## Negative controls

`tests/run_feature_parity.sh --negative` builds five mutants, each a plausible
wrong answer applied by rewriting **one line**, and requires each to turn the
gate **red**. Each mutation is guarded twice (the new line must appear, the old
must be gone), so a drifted `sed` cannot report a green control over an
unmutated binary.

| mutant | caught by |
|---|---|
| BBANDS `timeperiod` 20 -> 5 (mjolnir's value) | `bb_upper`/`bb_lower` first-valid-index (ref 19, cpp 4) and NaN classification |
| skew: SAMPLE G1 -> POPULATION g1 | `f083` value diff, max rel **1.105e-01** on 625-685 cells/scenario — the exact `(n-2)/sqrt(n(n-1))` factor at n=14 |
| kurt: SAMPLE EXCESS G2 -> POPULATION `D/B^2 - 3` | `f040` value diff |
| obv/ad: `obv_is_cumulative` False -> True (a second `diff`) | `f101`/`f106` first-valid-index (ref 19, cpp 20) **and** 656 cells at max rel **1.5e+03** |
| `_safe`: drop step 2 | `f103`/`f104`/`f105`/`f107` NaN classification at row 360, `ref=nan cpp=+/-inf` |

The last one is caught **only** because the fifth scenario carries a
`close = 1e-305` bar. The four older scenarios report `0/0/0` branch hits on
every column, so the mutant is bit-identical to the real engine there; without
that bar the control would pass and step 2 would be dead code as far as the gate
is concerned.

## A TA-Lib 0.6.4 defect — reported, not fixed

`ta_NATR.c:334-338`, in the pinned 0.6.4 source:

```c
      tempValue = inClose[today];
      if( !TA_IS_ZERO(tempValue) )
         outReal[outIdx] = (prevATR/tempValue)*100.0;
      else
         outReal[0] = 0.0;          /* <-- outReal[0], NOT outReal[outIdx] */
      outIdx++;
```

The `else` writes index **0** — a copy-paste of the initialisation twelve lines
above, where index 0 *is* the right target. Two separate consequences:

1. `outReal[outIdx]` is **never written**. The caller reads back whatever its
   buffer held. The Python wrapper allocates a fresh uninitialised array, so the
   reference returns heap garbage that **changes between identical calls** —
   measured `1.63e+69`, `61686.73`, `2.24e-314` from three calls on the same
   inputs. The C++ reuses one scratch buffer across indicator calls and reads
   back the previous indicator's value. Neither is right; the reference is
   simply not a function of its inputs there.
2. `outReal[0]` **is** clobbered to `0.0` — the first emitted NATR value, an
   unrelated and previously-correct cell, is destroyed. That part is
   deterministic and both sides reproduce it (`natr[14] == 0.0`), so it is
   compared, not waived.

`TA_IS_ZERO(v)` is `|v| < TA_EPSILON` with `TA_EPSILON = 1e-14`
(`ta_utility.h:257,259`) — an **absolute** threshold. No Binance perp trades
below 1e-14, so this is not reachable on today's feed; it is reachable for any
instrument quoted below it, and the failure mode is a `natr` column of
uninitialised memory with nothing logged.

The harness waives exactly those cells, and the waiver cannot grow on its own:
the rows are **derived** from the defect (`|close| < TA_EPSILON`), each must
independently be **proven** non-deterministic by re-running the reference's own
`talib.NATR` 16 times and observing disagreement, any non-determinism *outside*
the derived set **fails the gate**, and a row that turns out stable is compared
rather than waived. If TA-Lib fixes this, the proof stops succeeding and the
cells return under the diff with no edit here.

## The 65-column panel, reconciled against the 82-column reference frame

`engineer_features` produces **82** columns per symbol. The engine emits **65**.
The other 17 are accounted for exhaustively:

| group | n | why absent |
|---|---|---|
| raw passthroughs: `open`, `high`, `low`, `volume`, `quote_volume`, `taker_buy_quote_volume`, `number_of_trades` | 7 | research.py:336 seeds `engineered_frames = [df]`, so the input frame rides through the concat. They are **inputs**, carried in `RawBars`. `close` is the one the engineered panel re-emits, because the regime predicates compare against it. |
| lookahead targets: `return`, `return_long`, `return_short`, `return_long_raw`, `return_short_raw`, `return_dip`, `return_rip` | 7 | every one reads `close`/`high`/`low` at `shift(-1)`. Target construction for the trainer; a live engine computing one would be reading a bar that has not closed. Declared in `EXPECTED_ABSENT_TARGETS` and **proven** absent from the engine's output. |
| index metadata: `year`, `month`, `close_timestamp` | 3 | research.py:667-672. Calendar/bookkeeping, not features. |

The Phase 2 design put the feature panel at **67** = 60 (rename-map) + 7
(scale-free). The engine emits 65. The difference is exactly **`year` and
`month`**: the design's 60 counts them among the rename-map columns, and they
are index metadata rather than engineered features, so they are not emitted
here. 65 + 2 = 67, and 67 + 7 targets + 7 raw passthroughs + `close_timestamp`
= 82. The count reconciles with no unexplained residue.

## Why not mjolnir's bar_builder with bar_sec=900

The two disagree on both things the bar layer does:

- **Bucketing.** Binance assigns a trade to a kline by exchange event time, on
  buckets tiling the UTC day. mjolnir closes on the *first trade of a new
  bucket* against its own stream clock — deliberate and parity-tested there.
  Agamotto trained on Binance's klines, so bar membership must be theirs.
- **Fields.** mjolnir's bar carries book depth, OI, funding and liquidations;
  agamotto needs Binance's nine kline columns. The overlap is OHLC + volume.

## What the live feed actually populates

Measured on hydra 2026-08-18 against the deployed (Aug-12) `tsBinanceFeedPublisher`.
This table is the reason several design decisions look paranoid:

| field | observed | consequence |
|---|---|---|
| `header.exchange_timestamp` | valid epoch ms | **the bucketing clock** |
| `message.last_trade_id` | valid, monotonic | dedupes replays |
| `message.last_trade_ts` | `18359008543379257232` (uninitialised) | unusable; range-checked and counted |
| `message.is_buyer_maker` | `0` on all 613 trades of a 45s capture | `taker_buy_*` cannot be exact → quote rule |
| `mark` / `index` / `funding` | `0` on every event | unusable (no `MARK_PRICE_UPDATE` seen) |

The maker-flag gap is the same one `sentinel/README.md` documents for mjolnir's
`f091`. The field now *exists* in the 20260804 `Quote`; it is simply never set.
The fix belongs in the Feed Publisher, which has Binance's own flag at parse
time — no amount of porting recovers it.

## Measured parity, 2026-08-18

Live run on hydra, BTCUSDT, `bar_sec=60` (identical code path to 900, but a
Binance kline to diff against every minute). 15 minutes: 66,565 quotes, 5,636
`TRADE_UPDATE`, 0 `AGG_TRADE_UPDATE`, 14 bars, `conv_err=0`. 12 bars compared:

| column | match | worst rel err |
|---|---|---|
| open, high, low, close | **12/12** | 0.0 |
| volume, quote_volume | **12/12** | 0.0 |
| number_of_trades | **12/12** | 0.0 |
| `taker_buy_base` | 10/12 | **1.688e-01** |
| `taker_buy_quote` | 10/12 | **1.689e-01** |

Seven of the nine columns reproduce Binance **exactly**. The two that do not are
precisely the pair derived from the maker flag. `unclassified` was 0, so these
are not unsidable trades: the quote rule assigns a side to every trade and is
simply wrong on about two bars in twelve, by up to ~17%.

That matters downstream because `taker_buy_quote_volume` feeds the
`buy_pressure` feature, which the model learned on exact values. Closing it
needs the Feed Publisher to populate `is_buyer_maker` — it has Binance's own
flag at parse time. No further porting recovers it.

Latency, same run: **recv -> kline built** n=14, min 0.6 / p50 0.8 / p99 1.6 /
max 1.6 us. **kline -> signal** is not measurable in Phase 1 (`decide()` is a
no-op); `tests/talib_bench.cpp` bounds it at 268 us p50 for one symbol.

## The five bar rules

Each exists because getting it wrong yields a plausible-but-wrong bar:

1. **Only `TRADE_UPDATE` (6) contributes.** `AGG_TRADE_UPDATE` (7) is a distinct
   kind carrying the same fills re-aggregated; counting both doubles volume,
   quote_volume and number_of_trades while OHLC still looks perfect.
2. **Trade ids dedupe** replays and repeated SHM slots.
3. **The first bucket is discarded.** Attaching mid-bucket misses the trades
   before we attached; that bar would read as real with implausibly low volume.
4. **Empty buckets emit flat bars** (`o=h=l=c=prev close`, `v=0`), as Binance
   does. Skipping them shifts every rolling window downstream.
5. **A trade stamped in a closed bucket is dropped and counted**, never applied.

Trades strictly inside the spread are counted `unclassified` rather than
guessed, and every bar carries `aggressor_source` so no consumer can mistake a
quote-rule approximation for an exact figure.

## Build

**Use `./build_linux.sh`.** It builds in `mjolnir-core-build:latest`
(rockylinux:8, glibc 2.28 / gcc 8.5 — the oldest deploy target), exactly as
`../sentinel_core/build_linux.sh` does, and refuses to ship a `.so` that needs a
newer libstdc++.

```bash
./build_linux.sh                    # build + self-tests
./build_linux.sh --deploy dev105    # and copy the .so to a host
```

**Do NOT build the core in `devbox-v5.1`.** That is the vendor *SDK* image and
its toolchain is newer: measured 2026-08-18, a core built there requires
`GLIBCXX_3.4.29`, which hydra happens to have and **dev105 does not**. The
failure surfaces at `dlopen` on the host that lacks it and reads as a missing
plugin rather than as a toolchain mismatch. The rockylinux:8 build requires no
such symbol and resolves on every target.

devbox-v5.1 remains correct for the *plugin* (`libtsAgamotto.so`), which is
built against the vendor SDK — same split as mjolnir.

A plain cmake invocation still works for local development:

```bash
cmake -S . -B build -DSENTINEL_REPO=$HOME/sandbox/sentinel -DAGAMOTTO_CORE_GITSHA=$(git rev-parse --short HEAD)
cmake --build build -j
```

Then point the public strategy at it:

```bash
cmake -S <sentinel>/Strategy/ltp_release/ltp_strat_sdk -B <...>/build \
      -DLTP_SDK_LIB_TYPE=Release -DAGAMOTTO_CORE_DIR=$PWD/build
```

**The SDK must be 20260804 or newer.** The deployed feed publisher emits the
979-byte `Quote`; a binary built against the 954-byte 20260728 layout dies at
startup with `Mismatched shared memory segment, sizeof(T) = 954`.

## Tests

```bash
./build/kline_parity_driver --selftest                       # 102 assertions (84 + rule 7)
./build/kline_parity_driver --ticks T.csv --klines K.csv      # offline replay diff
./build/core_integration_driver                               # 153 assertions: engine, gate, models AND the decision are WIRED
./build/core_integration_driver --weights <export dir>        # ... against a REAL export (closed-form checks skipped)
./build/regime_parity_driver --selftest                       # atomIsKnown vs atomMask, all 4096 codes
./build/model_parity_driver --selftest                        # 12 refuse-to-load cases + regimeDirName
tests/run_pdops_parity.sh                                     # pdops vs pandas 2.3.3
tests/run_feature_parity.sh --both                            # 65 feature columns vs research.py
tests/run_feature_parity.sh --negative                        # 5 mutants, each must go red
tests/run_regime_parity.sh --both                             # 62 deployed regimes vs research_filters, EXACT
tests/run_regime_parity.sh --negative                         # 5 mutants, each must go red
tests/run_model_parity.sh --both                              # 62 regimes' y_pred vs utils.weights_io.load_regime().predict
tests/run_model_parity.sh --negative                          # 6 controls, each must go red
./build/decision_parity_driver --selftest                     # 12 gate refuse-to-load + 4 ballot cases
tests/run_decision_parity.sh --both                           # fired/side EXACT vs dual_gate_filter + signed_threshold
tests/run_decision_parity.sh --negative                       # 8 controls (3 gate overrides, 1 refuse-at-load, 4 mutants)
./build/talib_bench 2000                                      # ta-lib link proof + budget
python tests/live_reconcile.py --ssh-host hydra --bars 3      # live C++ chain vs the live bot

python tests/fetch_binance_klines.py --symbol BTCUSDT --interval 15m --limit 700 \
    --out <bundle>/config/backfill_BTCUSDT_15m.csv
python tests/compare_agbar_vs_binance.py --log <strategy.log> --symbol BTCUSDT --interval 1m
```

Run the live parity at `bar_sec=60` first: the code path is identical to 900 and
a Binance kline to diff against arrives every minute instead of every fifteen.

Parity tolerance is **relative (1e-9), never exact** — both sides accumulate the
same trades into a double in a different order, so identical values still differ
in the last ULP.

## Warmup

Agamotto needs **700 bars** (`VOL_Q_WINDOW=700`, `min_periods=700` fails closed;
the live bot fetches `limit=700`). At 15m that is **7.3 days** of live bars, so
backfill is not optional in practice. Network I/O stays out of the core — as it
does for mjolnir, whose weights also come from an external tool — so the seam is
a CSV written by `fetch_binance_klines.py` and loaded by the strategy before it
subscribes.

### The boot seam — the CSV MUST be refreshed after startup

There is a **one-bucket hole at every single start**, and it is structural, not a
fault:

1. `fetch_binance_klines.py` writes only CLOSED bars, so the file ends at the
   last closed bucket — call the next one **B**, which is still open.
2. The strategy loads that file, subscribes, and the first live trade lands
   part-way through **B**.
3. **B** is discarded as a partial (rule 3): we missed the trades before we
   attached, and a bar with implausibly low volume is worse than no bar.
4. So the first bar the core builds is **B+1**, and **B** is in neither half.

The core does **not** fabricate B — it never invents a bucket it half observed.
Instead it **quarantines** the pre-seam backfill (`pending_bars`), keeps warmth
honest (`contiguous_bars` counts only the live run), and publishes the exact
outstanding range as `missing_from_ms .. missing_to_ms`. The strategy re-reads
`backfill_csv` on each built bar while that range is open, slices exactly it,
and ingests it; the core splices the quarantine back and the window is
contiguous across the seam.

**That only works if something refreshes the file after B closes.** Run the
fetcher alongside the strategy:

```bash
python tests/fetch_binance_klines.py --symbol BTCUSDT --interval 15m --limit 700 \
    --out <bundle>/config/backfill_BTCUSDT_15m.csv --repeat-sec 60 &
```

Without it the run does not lie — it is cold and says so on every bar
(`[AGSEAM] ... is outstanding`, `[AGBAR] warm=0 bars=1/700`) — but it also never
becomes warm, while holding 699 quarantined bars that would fix it.

Before this, a discontinuity **cleared** history, so every start threw the whole
backfill away and fell back to 700 live bars. Observed in production:

```
[AGDIAG] bars_seen=713 contiguous=14/700 backfilled=699 seam_gaps=1
```

Read `seam_gaps`, `pending_bars`, `seam_repairs` and `missing_from_ms` together:
`pending_bars > 0` with `missing_from_ms != 0` means the fetcher is not
refreshing.
