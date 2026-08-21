// Real implementation of the agamotto ICore contract. PRIVATE.
//
// Phase 1 implemented the bar layer: ticks -> 15m Binance-shaped klines, plus
// REST backfill ingestion and the warmup counter.
//
// STAGE 2.6 wires the feature engine in behind it. On every completed bar, when
// warm, the newest PANEL_BARS retained bars are sliced out of the builder and
// handed to engineerFeatures(); the resulting 699x65 panel is retained, and its
// SHAPE and COST cross the ABI as diagnostics so both are observable before
// anything scores it.
//
// STAGE 3 adds the REGIME GATE behind that. A stack of coded conjunctions is
// installed once via setRegimeStack(); on every panel the gate classifies the
// panel's NEWEST row — the bar that just closed — and the per-regime fire flags
// and run counts cross the ABI.
//
// PHASE 4 adds the MODEL RUNNER behind THAT. Each stack regime's linear weights
// are loaded from `weights_dir` when the stack is installed, and on every panel
// each FIRING regime's model is evaluated on the same NEWEST row the gate
// classified. y_pred, the count of firing-and-finite regimes, and the winning
// regime cross the ABI on the Decision.
//
// PHASE 5 adds the DECISION on top. The per-leg centred gate arrives through
// createCore from algo_params, is VALIDATED there (2 bps floor included, and
// refused rather than clamped), and each firing regime's y_pred is compared
// against ITS leg's edge — `center_long + threshold_long` for a long,
// `center_short - threshold_short` for a short. The votes are counted, netted,
// multiplied by REVERSE, and that sign is the decision. src/decision_rule.cpp
// carries the line-by-line transcription of the three reference files.
//
// *** THERE IS STILL NO ORDER PATH. *** Not in this core, not in the public
// AgamottoStrategy, and not behind a flag. `Decision` crosses the ABI and gets
// LOGGED. Arming is an operator-gated decision that this port does not make.
//
// EXPECT A SMALL NUMBER. 53 of the 62 regimes in the deployed
// pred_agamotto.base.15m_1 stack CANNOT fire at all — their vol-quantile cutoff
// columns are all-NaN on a 699-row panel (marvel PR #532,
// docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md). A run reporting
// all 62 configured and a healthy spread of fires across them would mean this
// port had "fixed" the finding, which it must not. Those same 53 are exactly
// the regimes whose models carry 5 features, while all 9 firable ones carry 16
// — see model_runner.hpp and CoreDiagnostics::model_feature_count_variants.
#include "agamotto_core.hpp"
#include "decision_rule.hpp"
#include "feature_engine.hpp"
#include "kline_builder.hpp"
#include "model_runner.hpp"
#include "regime_gate.hpp"

#include <sys/stat.h>

#include <chrono>
#include <condition_variable>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef AGAMOTTO_CORE_GITSHA
#error "AGAMOTTO_CORE_GITSHA must be defined so a run traces back to its core"
#endif

namespace agamotto {
namespace {

class RealCore final : public ICore {
  public:
    // The worker holds a `this` pointer, so it MUST be stopped before any
    // member it touches is destroyed. Joining in the destructor is the only
    // place that ordering can be guaranteed.
    ~RealCore() override { stopWorker(); }

    RealCore(int64_t product_id, int bar_sec, int warmup_bars, const char* weights_dir,
             const DecisionGate& gate)
      : mProductId(product_id), mWarmupBars(warmup_bars), mBuilder(bar_sec, warmup_bars),
        mGateAbi(gate)
    {
        // PHASE 5. The gate is validated BEFORE anything else in this ctor,
        // because it is the cheapest check and the one an operator is most
        // likely to have got wrong by hand. Field-for-field, name-for-name:
        // the ABI struct and GateParams carry identical member names precisely
        // so this copy can be read as five lines rather than trusted.
        mGate.threshold_long = gate.threshold_long;
        mGate.threshold_short = gate.threshold_short;
        mGate.threshold_center_long = gate.threshold_center_long;
        mGate.threshold_center_short = gate.threshold_center_short;
        mGate.reverse = gate.reverse;
        mGate.validate();   // throws: non-finite, negative, sub-floor, bad reverse

        // PHASE 4. weights_dir is a REQUIRED config key with no default, and it
        // is validated HERE — before a 7.3-day warmup — rather than at the first
        // scored bar. A null or empty path is a MISSING key, not a request for
        // an unscored run: a core that gated bars and predicted nothing would
        // produce a silent no-signal day indistinguishable from a working one.
        if (weights_dir == nullptr || *weights_dir == '\0') {
            throw std::invalid_argument(
                "agamotto::createCore: weights_dir is required and must not be "
                "empty; there is no default. A core with no models produces a "
                "silent no-signal run that looks exactly like a working one");
        }
        mWeightsDir = weights_dir;
        // Existence is checked now, not when the stack arrives, so a typo'd or
        // unmounted path halts the boot instead of surviving to setRegimeStack
        // and being reported as 62 individually missing regimes.
        struct stat st;
        if (::stat(mWeightsDir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            throw std::invalid_argument(
                "agamotto::createCore: weights_dir '" + mWeightsDir +
                "' is not a readable directory");
        }

        // WARMTH IS ONE BAR MORE CONSERVATIVE THAN THE PANEL, DELIBERATELY.
        //
        // The reference engineers EXACTLY 699 rows (trading.py:443 fetches 700,
        // :485 drops the incomplete one), and engineerFeatures REFUSES any
        // other width — 700 rows would move every price_range_pct_q50 cell,
        // because that column is rolling(700, min_periods=1) and is therefore
        // EXPANDING on a frame shorter than 700. The panel width is a
        // CORRECTNESS parameter, not a buffer size, so it is pinned at
        // PANEL_BARS and the engine throws on anything else.
        //
        // Warmth stays at the contract's 700 rather than dropping to 699. The
        // cost is one extra bar of waiting, once, at boot; the benefit is that
        // the retained ring is always strictly longer than the slice, so the
        // slice never IS the whole ring and there is no width to get wrong at
        // the boundary.
        //
        // A warmup at or below the slice width could never produce a panel, and
        // a core that quietly never computed one would look exactly like a
        // quiet market — so refuse at construction rather than at every bar
        // forever.
        if (warmup_bars <= static_cast<int>(PANEL_BARS)) {
            throw std::invalid_argument(
                "agamotto::createCore: warmup_bars=" + std::to_string(warmup_bars) +
                " cannot slice a " + std::to_string(PANEL_BARS) +
                "-bar panel with a bar to spare; pass at least " +
                std::to_string(PANEL_BARS + 1) + " (the contract's 700)");
        }
    }

    void onTick(const TickEvent& ev) override
    {
        // One core instance serves exactly one product. Silently folding
        // another symbol's ticks into these bars would be invisible downstream.
        if (ev.product_id != mProductId) {
            ++mForeignProductTicks;
            return;
        }
        mBuilder.onTick(ev);
    }

    bool ingestBackfill(const KlineBar* bars, int n) override
    {
        const bool ok = mBuilder.ingestBackfill(bars, n);
        if (ok) {
            mBackfilled += n;
        }
        return ok;
    }

    // The panel is computed HERE, on the pop, not on the emit. The builder
    // emits into a queue and the caller drains it, so "a bar completed" and "a
    // bar was handed over" are different moments; the panel must be the one the
    // caller is about to look at.
    // Wall-clock bar close. See KlineBuilder::flushDue and the ICore contract:
    // the bar was previously produced only when the next tick rolled the
    // bucket, which made latency a function of how quiet the symbol was
    // (3.3-11.6 s past the boundary on 2026-08-20). The bars land in the same
    // queue barReady() already drains, so downstream sees no new path.
    // Correct retained bars against the venue's own klines. The tick path loses
    // trades before this core sees them -- the publisher cannot always write to
    // the ring -- and that publisher is not ours to change, so the bar is fixed
    // from the source rather than defended.
    int reconcileAgainst(const KlineBar* bars, int n,
                         KlineBar* out, int max_out) override
    {
        return mBuilder.reconcileAgainst(bars, n, out, max_out);
    }

    int flushDueBuckets(int64_t cutoff_ms) override
    {
        return mBuilder.flushDue(cutoff_ms);
    }

    bool barReady(KlineBar* out) override
    {
        if (!mBuilder.pop(out)) {
            return false;
        }
        // PHASE 5. The near end of the kline-built -> signal-generated span.
        // Taken from the bar itself rather than from "now": the bar may have
        // been emitted several ticks before the caller drained it, and timing
        // from the drain would silently understate the span.
        mBarEmitNs = out->bar_emit_ns;

        // ROUTE A. The panel is ~47 ms per symbol and it used to run RIGHT
        // HERE, on the thread whose other job is draining the SHM ring. With
        // 28 symbols in one process that is ~1.3 s per bar close of not
        // draining, against a 512-slot ring holding 0.47-1.67 s depending on
        // market rate -- which is why trades were lost, measured 1.2-8.5 pct
        // across 28 symbols on 2026-08-20.
        //
        // Async mode hands the work to a worker and returns. The drain thread's
        // per-bar cost becomes the SNAPSHOT below and nothing else, so it stops
        // being a function of how much arithmetic follows it.
        if (!mAsync) {
            snapshotHistory();
            computePanelFor(*out, mSyncHist.data(),
                            static_cast<int>(mSyncHist.size()));
            return true;
        }
        enqueueScoring(*out);
        return true;
    }

    // The worker must not read mBuilder: the drain thread appends to it on
    // every tick. It gets a private COPY of the contiguous run instead, taken
    // here on the drain thread. ~700 bars is the drain thread's entire new
    // per-bar cost, and it is constant -- it does not grow with the panel, the
    // regime count or the model count, which is the whole point.
    void snapshotInto(std::vector<KlineBar>& dst) const
    {
        const int n = mBuilder.contiguousBars();
        dst.resize(static_cast<size_t>(n < 0 ? 0 : n));
        for (int i = 0; i < n; ++i) {
            const KlineBar* b = mBuilder.barAt(i);
            if (b == nullptr) {          // out of range only; keep it bounded
                dst.resize(static_cast<size_t>(i));
                return;
            }
            dst[static_cast<size_t>(i)] = *b;
        }
    }

    void snapshotHistory() { snapshotInto(mSyncHist); }

    void enqueueScoring(const KlineBar& popped)
    {
        std::unique_lock<std::mutex> lk(mMx);
        // A NEWER BAR REPLACES AN UNSTARTED ONE, and it replaces an unread
        // RESULT too. Both match the behaviour this core already had: only the
        // newest bar gets a panel, older ones are skipped and COUNTED
        // (mPanelsSkippedStaleBar). Queueing them instead would grow without bound
        // whenever the worker cannot keep up, and would score bars whose
        // decisions are already too late to act on.
        if (mJobPending) ++mPanelsSkippedStaleBar;
        if (mResultPending) {
            ++mPanelsSkippedStaleBar;
            mResultPending = false;
        }
        snapshotInto(mJobHist);
        mJobPopped  = popped;
        mJobPending = true;
        lk.unlock();
        mCv.notify_one();
    }

    // Poll for a scored bar. Returns true exactly once per scored bar, and the
    // "latest" accessors (panelLatest, regimeFiredLatest, regimePrediction,
    // decide) describe THAT bar until the next one is enqueued.
    //
    // WHY THIS IS SAFE WITHOUT LOCKING THE ACCESSORS. The worker only ever runs
    // between an enqueue and a publish, and BOTH enqueue (barReady) and read
    // (after this returns true) happen on the SDK's single core thread -- so
    // they cannot overlap. The worker goes idle after publishing and cannot
    // start again until that thread enqueues, which it does not do while
    // reading. The mutex here supplies the happens-before, so everything the
    // worker wrote is visible to the reader.
    bool takeScoredBar(int64_t* bar_ts_ms) override
    {
        std::lock_guard<std::mutex> lk(mMx);
        if (!mResultPending) {
            return false;
        }
        mResultPending = false;
        if (bar_ts_ms != nullptr) *bar_ts_ms = mResultTsMs;
        return true;
    }

    void setAsyncScoring(bool on) override
    {
        if (on == mAsync) return;
        if (on) {
            mStop = false;
            mAsync = true;
            mWorker = std::thread([this] { workerLoop(); });
        } else {
            stopWorker();
            mAsync = false;
        }
    }

    bool asyncScoring() const override { return mAsync; }

    void workerLoop()
    {
        for (;;) {
            std::unique_lock<std::mutex> lk(mMx);
            mCv.wait(lk, [this] { return mStop || mJobPending; });
            if (mStop) return;
            // Move the job out under the lock, then compute WITHOUT it: the
            // panel is ~47 ms and holding the mutex for it would block the
            // drain thread's enqueue, reintroducing the stall this exists to
            // remove.
            mWorkHist.swap(mJobHist);
            const KlineBar popped = mJobPopped;
            mJobPending = false;
            lk.unlock();

            computePanelFor(popped, mWorkHist.data(),
                            static_cast<int>(mWorkHist.size()));

            lk.lock();
            mResultTsMs   = popped.bucket_open_ms;
            mResultPending = true;
            lk.unlock();
        }
    }

    void stopWorker()
    {
        if (!mWorker.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mMx);
            mStop = true;
        }
        mCv.notify_all();
        mWorker.join();
    }

    // Warmth is the length of the CONTIGUOUS retained run, not a count of
    // bars that have ever gone past. The old form returned barsSeen(), a
    // monotone counter over bars nothing kept, so the core reported warm on
    // history it did not hold AND stayed warm across a discontinuity — the two
    // ways a 700-bar window can be wrong while looking right.
    bool isWarm() const override { return barsBuffered() >= mWarmupBars; }
    int  barsBuffered() const override { return mBuilder.contiguousBars(); }
    int  warmupRequirement() const override { return mWarmupBars; }

    // Everything the run counted. Previously several of these were
    // incremented and exposed nowhere — a misrouted product_id swallowed every
    // tick with the only evidence in a counter no caller could read.
    CoreDiagnostics diagnostics() const override
    {
        CoreDiagnostics d{};
        d.bars_seen = mBuilder.barsSeen();
        d.contiguous_bars = mBuilder.contiguousBars();
        d.backfilled_bars = mBackfilled;
        d.foreign_product_ticks = mForeignProductTicks;
        d.seam_gaps = mBuilder.seamGaps();
        d.last_gap_from_ms = mBuilder.lastGapFromMs();
        d.last_gap_to_ms = mBuilder.lastGapToMs();
        d.pending_bars = mBuilder.pendingBars();
        d.missing_from_ms = mBuilder.missingFromMs();
        d.missing_to_ms = mBuilder.missingToMs();
        d.seam_repairs = mBuilder.seamRepairs();
        d.quarantines_discarded = mBuilder.quarantinesDiscarded();
        d.rewound_bars_dropped = mBuilder.rewoundBarsDropped();
        d.trade_updates = mBuilder.tradeUpdates();
        d.aggtrade_updates = mBuilder.aggTradeUpdates();
        d.duplicates_dropped = mBuilder.duplicatesDropped();
        d.bars_reconciled = mBuilder.barsReconciled();
        d.late_trades_dropped = mBuilder.lateTradesDropped();
        d.partial_buckets_dropped = mBuilder.partialBucketsDropped();
        d.bad_timestamp_dropped = mBuilder.badTimestampDropped();
        // Rule 7 (the dropped-trade detector). Run totals of what each bar
        // carries individually. These are the ONLY counters in this struct that
        // can move when the SHM ring loses a slot — everything above tallies
        // what arrived, and a slot that was overwritten never does.
        d.trade_id_gaps = mBuilder.tradeIdGaps();
        d.trades_missing = mBuilder.tradesMissing();

        // Stage 2.6. panel_rows/panel_cols are read off the PANEL ITSELF, never
        // from counters kept alongside it: a shape reported from a variable the
        // failure path forgot to reset is exactly how a discarded panel goes on
        // reporting 699x65.
        d.panel_cols = static_cast<int64_t>(mPanel.size());
        d.panel_rows = mPanel.cols.empty()
                         ? 0 : static_cast<int64_t>(mPanel.cols.front().size());
        d.panel_bar_ts_ms = mPanelBarTsMs;
        d.feature_compute_us = mFeatureComputeUs;
        d.feature_compute_us_max = mFeatureComputeUsMax;
        d.feature_compute_us_total = mFeatureComputeUsTotal;
        d.panels_computed = mPanelsComputed;
        d.panels_skipped_not_warm = mPanelsSkippedNotWarm;
        d.panels_skipped_stale_bar = mPanelsSkippedStaleBar;
        d.panel_errors = mPanelErrors;

        // Stage 3. regimes_configured comes off the stack itself for the same
        // reason panel_rows comes off the panel itself: a count kept in a
        // separate variable is one failure path away from describing a stack
        // that is no longer installed.
        d.regimes_configured = static_cast<int64_t>(mStack.size());
        d.regimes_fired_last_bar = mRegimesFiredLastBar;
        d.regime_evals = mRegimeEvals;
        d.regime_gate_us = mRegimeGateUs;
        d.regime_gate_us_max = mRegimeGateUsMax;
        d.regime_errors = mRegimeErrors;

        // Phase 4. Read off the ModelBook itself, for the same reason
        // panel_rows comes off the panel: a count kept in a parallel variable
        // is one failure path away from describing models that are no longer
        // loaded.
        d.models_loaded = static_cast<int64_t>(mModels.size());
        d.model_feature_count_variants =
            mModels.empty() ? 0 : static_cast<int64_t>(mModels.featureCountVariants());
        d.model_features_min =
            mModels.empty() ? 0 : static_cast<int64_t>(mModels.minFeatureCount());
        d.model_features_max =
            mModels.empty() ? 0 : static_cast<int64_t>(mModels.maxFeatureCount());
        d.model_unit_scale_features =
            mModels.empty() ? 0 : static_cast<int64_t>(mModels.unitScaleFeatures());
        d.predictions_computed = mPredictionsComputed;
        d.predict_us = mPredictUs;
        d.predict_us_max = mPredictUsMax;
        d.nan_features_filled = mNanFeaturesFilled;
        d.nonfinite_predictions = mNonfinitePredictions;
        d.model_errors = mModelErrors;

        // Phase 5. Read off the decision itself for the same reason panel_rows
        // comes off the panel: a count kept in a parallel variable is one
        // failure path away from describing a decision that was discarded.
        d.votes_long = mDecision.n_long;
        d.votes_short = mDecision.n_short;
        d.decisions_evaluated = mDecisionsEvaluated;
        d.decisions_fired = mDecisionsFired;
        d.decisions_long = mDecisionsLong;
        d.decisions_short = mDecisionsShort;
        d.bar_to_signal_us = mBarToSignalUs;
        d.bar_to_signal_us_max = mBarToSignalUsMax;
        d.decision_errors = mDecisionErrors;
        return d;
    }

    const char* panelColumnCode(int j) const override
    {
        if (j < 0 || static_cast<size_t>(j) >= mPanel.names.size()) return nullptr;
        return mPanel.names[static_cast<size_t>(j)].c_str();
    }

    double panelLatest(int j) const override
    {
        if (j < 0 || static_cast<size_t>(j) >= mPanel.cols.size())
            return std::numeric_limits<double>::quiet_NaN();
        const std::vector<double>& col = mPanel.cols[static_cast<size_t>(j)];
        if (col.empty()) return std::numeric_limits<double>::quiet_NaN();
        // The panel's LAST row is the bar that just closed. NaN and inf ride
        // out untouched: reproducing pandas' masks IS the contract, and a fill
        // here would turn "this gate cannot fire" into "this gate compares
        // against zero" (see feature_engine.hpp).
        return col.back();
    }

    std::string lastPanelError() const override { return mLastPanelError; }

    // --- PHASE 3: the regime gate ----------------------------------------
    // Validated in FULL before anything is stored. A half-installed stack
    // silently gates the regimes that did not make it to "never fires", which
    // is a strategy that is partly switched off with nothing in the log — the
    // same discipline ingestBackfill() follows for a partly-valid bar run.
    void setRegimeStack(const RegimeSpec* specs, int n) override
    {
        if (specs == nullptr || n <= 0) {
            throw std::invalid_argument(
                "agamotto::setRegimeStack: an EMPTY stack gates every regime "
                "to never-fires; refusing it rather than running switched off");
        }
        std::vector<Spec> parsed;
        parsed.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const RegimeSpec& s = specs[i];
            if (s.n_atoms == 0 || s.n_atoms > MAX_REGIME_ATOMS) {
                throw std::invalid_argument(
                    "agamotto::setRegimeStack: regime " + std::to_string(i) +
                    " has " + std::to_string(static_cast<unsigned>(s.n_atoms)) +
                    " atoms; 0 is the unconditional always-fire gate and more "
                    "than " + std::to_string(MAX_REGIME_ATOMS) + " cannot be "
                    "carried without truncating, which LOOSENS the gate");
            }
            if (s.position != 1 && s.position != -1) {
                throw std::invalid_argument(
                    "agamotto::setRegimeStack: regime " + std::to_string(i) +
                    " has position " + std::to_string(static_cast<int>(s.position)) +
                    "; must be +1 (long) or -1 (short)");
            }
            Spec sp;
            sp.pos = (s.position == 1) ? Position::LONG : Position::SHORT;
            for (uint8_t k = 0; k < s.n_atoms; ++k) sp.atoms.push_back(s.atom_codes[k]);
            // PROVE each atom is one this core can evaluate, HERE, at
            // configuration time. Discovering it at the first warm bar would
            // mean a run that boots clean, waits 7.3 days for warmup and only
            // then reports that its stack is unusable.
            for (const uint16_t c : sp.atoms) {
                if (!atomIsKnown(c)) {
                    throw std::invalid_argument(
                        "agamotto::setRegimeStack: regime " + std::to_string(i) +
                        " names atom code " + std::to_string(static_cast<unsigned>(c)) +
                        ", which this core has no predicate for");
                }
            }
            parsed.push_back(std::move(sp));
        }

        // PHASE 4. The weights are loaded BEFORE the stack is committed, for
        // exactly the reason the atom validation above happens before it: a
        // half-installed stack silently gates the regimes that did not make it
        // to "never trades". Every regime's directory name is RECONSTRUCTED
        // from its atom codes (no name crosses the ABI), and a regime with no
        // directory throws — which is what the Python bot does
        // (FileNotFoundError: Regime folder <name> not found) and what caught a
        // real stack/weights mismatch. Loading here rather than in the ctor is
        // simply when the core first learns WHICH regimes it needs.
        std::vector<std::string> dirs;
        dirs.reserve(parsed.size());
        for (const Spec& sp : parsed) dirs.push_back(regimeDirName(sp.atoms, sp.pos));

        ModelBook book;
        book.load(mWeightsDir, dirs);   // throws, naming the regime and the path

        mStack = std::move(parsed);
        mRegimeDirs = std::move(dirs);
        mModels = std::move(book);
        // Counts belong to the stack they were accumulated against; carrying
        // them across a replacement would attribute one regime's fires to
        // whatever now sits at that index.
        mFireCounts.assign(mStack.size(), 0);
        mFiredLatest.assign(mStack.size(), 0);
        mPredLatest.assign(mStack.size(), std::numeric_limits<double>::quiet_NaN());
        // PHASE 5. The leg of each stack entry, lifted out ONCE so the per-bar
        // vote reads a flat vector instead of reaching into mStack. Same size
        // as everything else here or evaluateDecision() throws, which is the
        // point: a positions vector that fell out of step with the predictions
        // would flip votes silently.
        mStackPositions.clear();
        mStackPositions.reserve(mStack.size());
        for (const Spec& sp : mStack) mStackPositions.push_back(sp.pos);
        mTriggeredLatest.assign(mStack.size(), 0);
        mDecision = DecisionOutcome{};
        mDecisionValid = false;
    }

    int regimeStackSize() const override { return static_cast<int>(mStack.size()); }

    bool regimeFiredLatest(int i) const override
    {
        if (i < 0 || static_cast<size_t>(i) >= mFiredLatest.size()) return false;
        return mFiredLatest[static_cast<size_t>(i)] != 0;
    }

    // PHASE 5. Strictly narrower than regimeFiredLatest: the gate held AND the
    // prediction cleared this regime's leg edge. Parallel to mStack, cleared by
    // exactly the same paths as the fire flags, so a stale vote can never be
    // read as this bar's.
    bool regimeTriggeredLatest(int i) const override
    {
        if (i < 0 || static_cast<size_t>(i) >= mTriggeredLatest.size()) return false;
        return mTriggeredLatest[static_cast<size_t>(i)] != 0;
    }

    int64_t regimeFireCount(int i) const override
    {
        if (i < 0 || static_cast<size_t>(i) >= mFireCounts.size()) return 0;
        return mFireCounts[static_cast<size_t>(i)];
    }

    std::string lastRegimeError() const override { return mLastRegimeError; }

    // --- PHASE 4: the model runner ---------------------------------------
    double regimePrediction(int i) const override
    {
        if (i < 0 || static_cast<size_t>(i) >= mPredLatest.size())
            return std::numeric_limits<double>::quiet_NaN();
        return mPredLatest[static_cast<size_t>(i)];
    }

    int winningRegimeIndex() const override { return mWinningIndex; }

    std::string modelInventory() const override
    {
        return mModels.empty() ? std::string("no models loaded") : mModels.inventory();
    }

    std::string lastModelError() const override { return mLastModelError; }

    // PHASE 5. REPORTS the decision the panel produced; it does not recompute
    // it. The vote was taken in runModels(), on the same row and the same
    // predictions the gate saw, so nothing here can disagree with what was
    // logged per regime.
    //
    // IDEMPOTENT. The bar is counted once (in runModels) and signal_emit_ns is
    // stamped once, on the FIRST call for a bar; a second call returns the same
    // decision with the same timestamp rather than a fresh one, so a caller
    // that logs and then re-reads cannot manufacture a second signal or a
    // shorter latency.
    Decision decide() override
    {
        Decision d;
        d.bar_ts_ms = mPanelBarTsMs;
        d.fired = mDecision.fired;
        d.side = mDecision.side;
        d.y_pred = mDecision.y_pred;
        d.threshold = mDecision.threshold;
        d.threshold_center = mDecision.threshold_center;
        d.n_triggered = mDecision.n_triggered;
        d.winning_regime_code = mWinningCode;

        if (mDecisionValid && mSignalEmitNs == 0) {
            mSignalEmitNs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            // KLINE-BUILT -> SIGNAL-GENERATED. Both ends are stamped on the
            // same clock (system_clock: the builder sets bar_emit_ns from it
            // too), so the difference is a real span rather than two clocks
            // subtracted. Skipped when the bar carries no emit stamp — a
            // BACKFILLED bar has none, and a difference against 0 would read as
            // fifty-odd years of latency.
            if (mBarEmitNs != 0 && mSignalEmitNs > mBarEmitNs) {
                mBarToSignalUs =
                    static_cast<int64_t>((mSignalEmitNs - mBarEmitNs) / 1000ULL);
                if (mBarToSignalUs > mBarToSignalUsMax) {
                    mBarToSignalUsMax = mBarToSignalUs;
                }
            }
        }
        d.signal_emit_ns = mSignalEmitNs;
        return d;
    }

    std::string lastDecisionError() const override { return mLastDecisionError; }

    DecisionGate decisionGate() const override { return mGateAbi; }

    bool coreIsRealImplementation() const override { return true; }

    std::string coreBuildTag() const override
    {
        return std::string("agamotto-core-") + AGAMOTTO_CORE_GITSHA + "-phase5-decision";
    }

  private:
    // Runs the feature engine for the bar that was just popped, or records
    // exactly why it did not. Every early return increments a counter that
    // crosses the ABI — "the panel never ran" and "the panel ran and produced
    // nothing" are the same silence otherwise.
    // Takes an explicit SNAPSHOT rather than reading mBuilder, so this can run
    // on a worker thread while the drain thread keeps appending to the builder.
    // `hist` is a private copy owned by the caller; nothing here touches shared
    // state that the tick path can mutate. `n_hist` is the CONTIGUOUS run, the
    // same quantity isWarm() is defined on.
    void computePanelFor(const KlineBar& popped, const KlineBar* hist, int n_hist)
    {
        if (n_hist < mWarmupBars) {
            ++mPanelsSkippedNotWarm;
            return;
        }

        const KlineBar* newest = (n_hist > 0) ? &hist[n_hist - 1] : nullptr;
        if (newest == nullptr) {
            // Unreachable while warm (warmth IS a nonzero contiguous run), but
            // dereferencing on the strength of that reasoning is how a null
            // deref reaches a live path. Counted, not argued away.
            ++mPanelsSkippedNotWarm;
            return;
        }

        // A BURST POPS BARS THAT ALREADY HAVE SUCCESSORS. A quiet stretch
        // drains several flat bars out of a single tick (builder rule 4), and
        // every one of them is in mHistory before the first pop returns. The
        // panel this function builds ENDS AT THE NEWEST retained bar, so
        // pairing it with an older popped bar would hand Phase 3/4 a window
        // containing bars that had not closed when `popped` did — lookahead,
        // and of the kind that produces a better-than-real backtest rather than
        // an error.
        //
        // Re-slicing a window that ENDED at `popped` is not the fix either:
        // that is not the window live scores, and on a burst it charges a full
        // engineerFeatures pass per bar for panels nothing acts on. Skipped and
        // COUNTED, so an illiquid symbol's missing panels are a number rather
        // than a silence.
        if (popped.bucket_open_ms != newest->bucket_open_ms) {
            ++mPanelsSkippedStaleBar;
            return;
        }

        const int n = n_hist;
        const int first = n - static_cast<int>(PANEL_BARS);
        // Guaranteed by the ctor check (warmup > PANEL_BARS) together with
        // isWarm(), but an underflow here indexes the deque out of bounds, so
        // it is checked rather than reasoned about.
        if (first < 0) {
            ++mPanelsSkippedNotWarm;
            return;
        }

        RawBars rb;
        rb.open.reserve(PANEL_BARS);
        rb.high.reserve(PANEL_BARS);
        rb.low.reserve(PANEL_BARS);
        rb.close.reserve(PANEL_BARS);
        rb.volume.reserve(PANEL_BARS);
        rb.quote_volume.reserve(PANEL_BARS);
        rb.taker_buy_quote_volume.reserve(PANEL_BARS);
        rb.number_of_trades.reserve(PANEL_BARS);
        for (int i = first; i < n; ++i) {
            const KlineBar* b = &hist[i];
            if (b == nullptr) {
                // barAt returns nullptr only out of range, which the bounds
                // above exclude. Refusing to fabricate a bar is the same rule
                // the builder follows at a seam.
                ++mPanelsSkippedNotWarm;
                return;
            }
            rb.open.push_back(b->open);
            rb.high.push_back(b->high);
            rb.low.push_back(b->low);
            rb.close.push_back(b->close);
            rb.volume.push_back(b->volume);
            // ALL THREE OPTIONALS ARE ALWAYS PRESENT ON THIS PATH. The builder
            // emits Binance's nine columns for every bar, backfilled or built,
            // so the "the feed does not carry it" branch (an EMPTY vector) is
            // unreachable here — which is what makes the live panel 65 columns
            // wide and directly comparable to the reference. Omitting one would
            // silently drop quote_vol_ratio / buy_pressure / trade_intensity
            // and the panel would still look valid.
            rb.quote_volume.push_back(b->quote_volume);
            rb.taker_buy_quote_volume.push_back(b->taker_buy_quote_volume);
            rb.number_of_trades.push_back(static_cast<double>(b->number_of_trades));
        }

        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        try {
            Table t = engineerFeatures(rb);
            const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            mPanel = std::move(t);
            mPanelBarTsMs = popped.bucket_open_ms;
            mFeatureComputeUs = static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            if (mFeatureComputeUs > mFeatureComputeUsMax) {
                mFeatureComputeUsMax = mFeatureComputeUs;
            }
            mFeatureComputeUsTotal += mFeatureComputeUs;
            ++mPanelsComputed;
            evaluateGate();
            // SEPARATE from evaluateGate() and with its own catch, so a model
            // failure is reported as a MODEL failure. Folding it into the
            // gate's try would attribute a bad weight file to the predicates
            // and clear the fire flags for a reason that has nothing to do with
            // them — a wrong diagnosis in the one log line an operator reads.
            runModels();
        } catch (const std::exception& e) {
            // WHY this is caught rather than propagated: barReady() runs inside
            // the vendor SDK's quote handler, and an exception escaping into
            // that event loop takes the process down. It is NOT swallowed —
            // panel_errors and lastPanelError() cross the ABI and the strategy
            // LOG_ERRORs the instant the counter moves.
            //
            // The PREVIOUS panel is DESTROYED, not kept. A retained stale panel
            // would go on reporting 699x65 and would be scored, on this bar, as
            // though it described it.
            ++mPanelErrors;
            mLastPanelError = e.what();
            mPanel = Table{};
            mPanelBarTsMs = 0;
            mFeatureComputeUs = 0;
            // The gate reads the panel. With no panel there is nothing to
            // classify, so the previous bar's fire flags are CLEARED rather
            // than left standing — a stale set read as this bar's is the same
            // class of bug as a stale panel being scored.
            clearFiredLatest();
        }
    }

    // The gate, on the panel that was just computed. Runs on the SAME row
    // anything downstream would score (the panel's last row, i.e. the bar that
    // just closed), and over the WHOLE panel, because several predicates
    // compare against columns that are only meaningful panel-wide
    // (price_range_pct_q50 is a 700-bar rolling median — see regime_gate.hpp).
    void evaluateGate()
    {
        if (mStack.empty()) {
            // No stack installed. Counted as "not evaluated" rather than as
            // "nothing fired": those are different states and only the second
            // is a market observation.
            return;
        }
        const size_t rows = mPanel.cols.empty() ? 0 : mPanel.cols.front().size();
        if (rows == 0) {
            clearFiredLatest();
            return;
        }
        const std::chrono::steady_clock::time_point g0 = std::chrono::steady_clock::now();
        try {
            int64_t fired = 0;
            for (size_t i = 0; i < mStack.size(); ++i) {
                const std::vector<char> m =
                    regimeMask(mPanel, mStack[i].atoms, mStack[i].pos);
                // The LAST row is the bar that just closed. NaN compared false
                // all the way through, which is how the 53 vol-quantile-gated
                // regimes stay at 0 for the whole run.
                const bool hit = m.back() != 0;
                mFiredLatest[i] = hit ? 1 : 0;
                if (hit) {
                    ++mFireCounts[i];
                    ++fired;
                }
            }
            const std::chrono::steady_clock::time_point g1 = std::chrono::steady_clock::now();
            mRegimeGateUs = static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(g1 - g0).count());
            if (mRegimeGateUs > mRegimeGateUsMax) mRegimeGateUsMax = mRegimeGateUs;
            mRegimesFiredLastBar = fired;
            ++mRegimeEvals;
        } catch (const std::exception& e) {
            // Caught for the same reason the panel's throw is (barReady runs
            // inside the SDK's quote handler), and swallowed no more than that
            // one is: the flags are cleared so no stale classification can be
            // read as this bar's, the counter crosses the ABI, and the
            // strategy LOG_ERRORs with lastRegimeError().
            ++mRegimeErrors;
            mLastRegimeError = e.what();
            clearFiredLatest();
        }
    }

    // PHASE 4. One linear model per FIRING regime, on the panel's NEWEST row —
    // the same row the gate classified, which is the only row anything scores.
    //
    // Non-firing regimes are not evaluated at all: the reference predicts only
    // on rows its filter let through (trading.py `predict(filtered_signals)`),
    // and scoring a gated-out bar would spend the time to produce a number
    // nothing may act on. Their slot stays NaN rather than 0.0, because 0.0 is
    // a legitimate prediction and would read as a confident flat call.
    void runModels()
    {
        clearPredLatest();
        if (mStack.empty() || mModels.empty()) return;
        const size_t rows = mPanel.cols.empty() ? 0 : mPanel.cols.front().size();
        if (rows == 0) return;

        const std::chrono::steady_clock::time_point p0 = std::chrono::steady_clock::now();
        try {
            // Proved ONCE per panel, not once per model: the column indices
            // were resolved at load, so a panel whose layout moved would feed
            // every model its neighbour's numbers — a plausible value for the
            // wrong feature, which nothing downstream can detect.
            ModelBook::assertPanelLayout(mPanel);

            const size_t row = rows - 1;
            int triggered = 0;
            for (size_t i = 0; i < mStack.size(); ++i) {
                if (mFiredLatest[i] == 0) continue;
                const double y = mModels.at(mRegimeDirs[i])
                                     .predictRow(mPanel, row, &mNanFeaturesFilled);
                ++mPredictionsComputed;
                mPredLatest[i] = y;
                if (!std::isfinite(y)) {
                    // inf is not filled anywhere in the reference (only NaN is),
                    // so an infinite feature reaches here. Counted and EXCLUDED
                    // rather than compared: `inf > threshold` is true, and a
                    // regime that fires on a poisoned column is worse than one
                    // that does not fire.
                    ++mNonfinitePredictions;
                    continue;
                }
                ++triggered;
            }
            const std::chrono::steady_clock::time_point p1 = std::chrono::steady_clock::now();
            mPredictUs = static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count());
            if (mPredictUs > mPredictUsMax) mPredictUsMax = mPredictUs;

            mTriggeredCount = triggered;
        } catch (const std::exception& e) {
            // Caught for the same reason the panel's and the gate's throws are
            // (barReady runs inside the SDK's quote handler), and swallowed no
            // more than those: this bar's predictions are DISCARDED so a stale
            // set cannot be read as this bar's, the counter crosses the ABI, and
            // the strategy LOG_ERRORs with lastModelError().
            ++mModelErrors;
            mLastModelError = e.what();
            clearPredLatest();
            return;
        }
        takeDecision();
    }

    // PHASE 5. The vote, on the predictions runModels() just took.
    //
    // A SEPARATE try FROM THE MODELS', for the same reason the models have one
    // separate from the gate's: a failure here is a DECISION failure, and
    // folding it into mModelErrors would blame a bad weight file for a bad
    // threshold. Phase 4's "largest |y_pred| over everything that predicted"
    // ordering is REPLACED here, not extended: it could name a SHORT regime as
    // the winner of a LONG decision.
    void takeDecision()
    {
        try {
            mDecision = evaluateDecision(mGate, mStackPositions, mPredLatest);
            for (size_t i = 0; i < mStack.size(); ++i) {
                const Position pos = mStackPositions[i];
                const LegGate g = mGate.leg(pos);
                mTriggeredLatest[i] = legFires(mPredLatest[i], g, pos) ? 1 : 0;
            }
            mWinningIndex = mDecision.winning_index;
            mWinningY = (mWinningIndex >= 0) ? mDecision.y_pred : 0.0;
            // The conjunction's LEADING atom. A 3-atom regime has no single
            // code and this ABI field is one uint16; winningRegimeIndex() is
            // the unambiguous identity and the caller holds the stack.
            mWinningCode = (mWinningIndex >= 0)
                             ? mStack[static_cast<size_t>(mWinningIndex)].atoms.front() : 0;
            mDecisionValid = true;
            ++mDecisionsEvaluated;
            if (mDecision.fired) {
                ++mDecisionsFired;
                if (mDecision.side > 0) ++mDecisionsLong; else ++mDecisionsShort;
            }
        } catch (const std::exception& e) {
            // Same discipline as everywhere else on this path: the decision is
            // DISCARDED so a stale `fired` cannot be read as this bar's — the
            // worst version of that bug, because `fired` is the one field a
            // caller could act on.
            ++mDecisionErrors;
            mLastDecisionError = e.what();
            mDecision = DecisionOutcome{};
            mDecisionValid = false;
            mTriggeredLatest.assign(mTriggeredLatest.size(), 0);
        }
    }

    void clearFiredLatest()
    {
        mFiredLatest.assign(mFiredLatest.size(), 0);
        mRegimesFiredLastBar = 0;
        // A prediction outlives its fire flag by nothing: if the gate could not
        // say this bar held, there is no bar for a prediction to be about.
        clearPredLatest();
    }

    void clearPredLatest()
    {
        mPredLatest.assign(mPredLatest.size(), std::numeric_limits<double>::quiet_NaN());
        mTriggeredCount = 0;
        mWinningIndex = -1;
        mWinningY = 0.0;
        mWinningCode = 0;
        // PHASE 5. The decision dies with the predictions it was taken on. A
        // retained `fired` would be a SIGNAL attributed to a bar whose panel,
        // gate or models had just been discarded — the worst version of the
        // stale-read bug this file guards against everywhere else, because it
        // is the one field a caller could act on.
        mTriggeredLatest.assign(mTriggeredLatest.size(), 0);
        mDecision = DecisionOutcome{};
        mDecisionValid = false;
        mSignalEmitNs = 0;
    }

    const int64_t mProductId;
    const int     mWarmupBars;
    KlineBuilder  mBuilder;
    int64_t       mBackfilled{0};
    int64_t       mForeignProductTicks{0};

    Table       mPanel;
    int64_t     mPanelBarTsMs{0};
    int64_t     mFeatureComputeUs{0};
    int64_t     mFeatureComputeUsMax{0};
    int64_t     mFeatureComputeUsTotal{0};
    int64_t     mPanelsComputed{0};
    int64_t     mPanelsSkippedNotWarm{0};
    int64_t     mPanelsSkippedStaleBar{0};

    // ---- ROUTE A: scoring on a worker thread ------------------------------
    // The drain thread must never run the ~47 ms panel: with 28 symbols in one
    // process that was ~1.3 s per bar of not reading the SHM ring, against a
    // ring holding 0.47-1.67 s. Default OFF so behaviour is byte-identical to
    // the synchronous core unless a caller opts in.
    bool                    mAsync{false};
    std::thread             mWorker;
    mutable std::mutex      mMx;
    std::condition_variable mCv;
    bool                    mStop{false};
    // ONE job and ONE result, never a queue: a backlog would score bars whose
    // decisions are already too late, and would grow without bound whenever
    // the worker falls behind. Superseded bars are counted, not silently lost.
    std::vector<KlineBar>   mJobHist;      // guarded by mMx
    KlineBar                mJobPopped{};  // guarded by mMx
    bool                    mJobPending{false};
    std::vector<KlineBar>   mWorkHist;     // worker-only, never touched by the drain thread
    int64_t                 mResultTsMs{0};
    bool                    mResultPending{false};
    std::vector<KlineBar>   mSyncHist;     // drain-thread-only, synchronous path
    int64_t     mPanelErrors{0};
    std::string mLastPanelError;

    // Stage 3. The stack is held DECODED into the gate's own types, so the
    // ABI struct is a transport shape and nothing downstream re-validates it
    // per bar.
    struct Spec {
        std::vector<uint16_t> atoms;
        Position pos{Position::LONG};
    };
    std::vector<Spec>    mStack;
    std::vector<char>    mFiredLatest;    // parallel to mStack; the NEWEST row
    std::vector<int64_t> mFireCounts;     // parallel to mStack; the whole run
    int64_t     mRegimesFiredLastBar{0};
    int64_t     mRegimeEvals{0};
    int64_t     mRegimeGateUs{0};
    int64_t     mRegimeGateUsMax{0};
    int64_t     mRegimeErrors{0};
    std::string mLastRegimeError;

    // Phase 4. mRegimeDirs is parallel to mStack and holds the CODED directory
    // name each spec reconstructs to, computed once when the stack is
    // installed — rebuilding it per bar would put a string format on the hot
    // path for a value that cannot change.
    std::string              mWeightsDir;
    std::vector<std::string> mRegimeDirs;
    ModelBook                mModels;
    std::vector<double>      mPredLatest;   // parallel to mStack; NaN = not scored
    int         mTriggeredCount{0};
    int         mWinningIndex{-1};
    double      mWinningY{0.0};
    uint16_t    mWinningCode{0};
    int64_t     mPredictionsComputed{0};
    int64_t     mPredictUs{0};
    int64_t     mPredictUsMax{0};
    int64_t     mNanFeaturesFilled{0};
    int64_t     mNonfinitePredictions{0};
    int64_t     mModelErrors{0};
    std::string mLastModelError;

    // Phase 5. mGate is the validated internal form; mGateAbi is the struct as
    // handed in, echoed back by decisionGate() so a boot line reports what the
    // core will COMPARE AGAINST rather than what the caller meant to install.
    GateParams   mGate;
    DecisionGate mGateAbi;
    // Parallel to mStack: the gate held AND the leg edge was cleared.
    std::vector<char> mTriggeredLatest;
    std::vector<Position> mStackPositions;   // parallel to mStack, fixed at install
    DecisionOutcome mDecision;
    bool        mDecisionValid{false};
    uint64_t    mBarEmitNs{0};
    uint64_t    mSignalEmitNs{0};
    int64_t     mBarToSignalUs{0};
    int64_t     mBarToSignalUsMax{0};
    int64_t     mDecisionsEvaluated{0};
    int64_t     mDecisionsFired{0};
    int64_t     mDecisionsLong{0};
    int64_t     mDecisionsShort{0};
    int64_t     mDecisionErrors{0};
    std::string mLastDecisionError;
};

} // namespace

std::unique_ptr<ICore> createCore(int64_t product_id, int bar_sec, int warmup_bars,
                                  const char* weights_dir, const DecisionGate& gate)
{
    return std::unique_ptr<ICore>(
        new RealCore(product_id, bar_sec, warmup_bars, weights_dir, gate));
}

} // namespace agamotto
