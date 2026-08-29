#pragma once
// Tick -> 15m Binance-shaped kline accumulator. PRIVATE.
//
// This is NOT mjolnir's bar_builder with bar_sec=900. The two differ on both
// load-bearing decisions:
//
//   * BUCKETING. Binance assigns a trade to a kline by the TRADE's own exchange
//     timestamp, and buckets tile the UTC day. mjolnir buckets by its own
//     stream clock. Agamotto's model trained on Binance's klines, so we must
//     reproduce Binance's assignment or every bar is subtly the wrong set of
//     trades.
//   * FIELDS. Binance's nine columns, including quote_volume and the two
//     taker_buy_* columns, none of which mjolnir's bar carries.
//
// Rules, each of which exists because getting it wrong produces a plausible
// but wrong bar:
//
//   1. ONLY UpdateKind::TRADE_UPDATE (6) contributes volume/trades. The
//      publisher subscribes to btcusdt@trade AND btcusdt@aggTrade, and
//      AGG_TRADE_UPDATE (7) is a DISTINCT kind carrying the SAME fills
//      re-aggregated. Counting both would double volume, quote_volume and
//      number_of_trades while OHLC still looked perfect. Kind 7 is counted for
//      reporting and otherwise ignored. (Measured 2026-08-18 on hydra: 613
//      kind-6 and 0 kind-7 events in 45s — but the subscription is live, so
//      the guard stays.)
//   2. TRADE IDS DEDUPE. Binance trade ids are monotonic per symbol; a replay
//      or a repeated SHM slot is dropped on id <= last_id.
//   3. THE FIRST BUCKET IS DISCARDED. Attaching mid-bucket means we missed the
//      trades before we attached, so that bar is a partial and would read as a
//      real bar with implausibly low volume. It is never emitted.
//   4. EMPTY BUCKETS ARE EMITTED FLAT. Binance publishes a kline for a bucket
//      with no trades: o=h=l=c=previous close, volume 0. Skipping them would
//      shift every rolling window downstream.
//   5. A FUTURE-BUCKET EVENT NEVER TOUCHES THE OPEN BAR. That is lookahead.
//   6. A DISCONTINUITY QUARANTINES THE PRE-GAP RUN, IT DOES NOT DESTROY IT.
//      Rule 3 GUARANTEES a one-bucket hole at every start: the backfill can
//      only end at the last closed bucket, and the bucket we attach on is
//      discarded. Clearing history there threw the whole backfill away on
//      every start (observed in production: bars_seen=713 contiguous=14/700
//      backfilled=699 seam_gaps=1). The pre-gap run is held aside instead,
//      uncounted, until the missing buckets are ingested and it is spliced
//      back. The hole is never filled by fabricating a bar.
//   7. A JUMP IN THE TRADE-ID SEQUENCE IS A DROPPED TRADE, AND IT IS COUNTED.
//      The SHM ring (container_size: 1024) overflows in a burst and the
//      consumer sees NOTHING: a lost slot never arrives, so every counter that
//      tallies what DID arrive stays clean. Measured on hydra 2026-08-19 —
//      the 15:15 15m bar lost 5.04% of its trades and MISSED THE HIGH BY 562
//      POINTS (69888 vs Binance 70450) with unclassified=0, aggressor=exact
//      and conv_err=0. Ids are monotonic per symbol (rule 2 already relies on
//      that for dedupe), so `id > expected` is the only in-band evidence the
//      loss leaves behind. Each bar therefore carries how many gap EVENTS fell
//      in it and how many IDS they skipped, and the run carries the totals.
//
//      THREE THINGS ARE DELIBERATELY *NOT* GAPS, because each would
//      manufacture a loss the ring did not cause:
//        (a) the FIRST id of a run — there is no predecessor for it to be
//            missing from, and treating id 4109... as "4109... trades lost"
//            would put a nonsense number on every first bar;
//        (b) an id we dropped OURSELVES as a duplicate (rule 2) — it arrived,
//            we chose not to apply it, and the sequence never advanced past it;
//        (c) an id equal to the expected one — the ordinary case.
//
//      DECLARED LIMITATION, pinned by a self-test rather than left to
//      surprise someone. An id that arrives OUT OF ORDER (lower than one
//      already accepted) is first counted missing by the jump that overtook
//      it, and is then dropped by rule 2 when it turns up. That is the RIGHT
//      answer for the question this counter asks — "how many trades are not
//      in this bar" — since rule 2 means the late one never contributes
//      either; it is the wrong answer for "how many did the ring lose". The
//      two coincide on Binance's trade stream, which is ordered.
//
//      NOTHING IS RECOVERED. A slot the ring overwrote is gone; the fix
//      belongs in the feed publisher. This makes the loss LOUD, nothing more.

#include <cstdint>
#include <deque>
#include <vector>

#include "agamotto_core.hpp"

namespace agamotto {

class KlineBuilder {
  public:
    // period_sec must divide 86400 — checked by the caller; buckets tile the
    // UTC day exactly as Binance's do. max_history caps the retained bar ring;
    // pass the warmup requirement, since that is the longest window anything
    // downstream reads.
    KlineBuilder(int period_sec, int max_history);

    // Feed one adapted tick. Completed bars are appended to the ready queue.
    void onTick(const TickEvent& ev);

    // Pop one completed bar. False when none pending.
    bool pop(KlineBar* out);

    // Seed closed bars (REST backfill). Oldest-first, on-grid, contiguous.
    // Returns false and ingests NOTHING on any violation — a silently accepted
    // hole would train the 700-bar rolling windows on a lie.
    //
    // Accepts a run that either APPENDS to the retained history (ends before it
    // begins is the prepend case below) or PREPENDS to it. Prepending is what
    // closes the backfill->live seam: the process attaches mid-bucket, discards
    // that partial bucket, and its first built bar is therefore one bucket
    // later than the newest bar any boot-time backfill could contain. Filling
    // that hole afterwards is the only way to reach a contiguous window without
    // fabricating a bar for a bucket we only partly observed.
    //
    // After any accepted join the quarantined prefix (see quarantine below) is
    // re-tested: a run that lands exactly on the hole SPLICES the two halves
    // back into one contiguous deque, which is how the boot seam is repaired
    // without a restart and without inventing the bucket we only half saw.
    bool ingestBackfill(const KlineBar* bars, int n);

    /// Replace retained bars that DISAGREE with the venue's own klines, and
    /// report how many were corrected.
    ///
    /// WHY THIS EXISTS. The tick path loses trades before this builder ever
    /// sees them: the feed publisher reports `Failed to publish market data`
    /// when it cannot write to the SHM ring, so the message never reaches a
    /// consumer at all. That publisher is vendor code and is not ours to fix.
    /// Measured 2026-08-21 on a QUIET bar (168 msg/s, 3.05 s of ring headroom,
    /// with scoring already moved off the drain thread): AAVE -12.21 pct,
    /// 1000PEPE -11.34 pct, LINK -3.91 pct, and even BNB one trade short.
    ///
    /// So the bar is CORRECTED from the authoritative source rather than
    /// defended. `bars` are Binance's own klines, already refreshed every 60 s
    /// for the seam repair, so this costs one comparison per retained bar and
    /// no new I/O.
    ///
    /// SCOPE, deliberately narrow: it replaces only bars this builder ALREADY
    /// HOLDS at a matching bucket_open_ms. It never inserts, never extends and
    /// never reorders -- filling a hole is ingestBackfill's job and conflating
    /// the two would let a stale CSV silently rewrite history it should only
    /// have patched.
    ///
    /// A bar built from ticks carries diagnostics the kline cannot (aggressor
    /// source, trade-id gaps, recv timing). Those are PRESERVED: only the nine
    /// venue columns are overwritten, so a corrected bar still reports how badly
    /// its tick stream was damaged.
    ///
    /// `out`/`max_out` receive the CORRECTED bars so a caller can LOG them.
    /// Without that the correction is invisible: the only record of a bar is
    /// the line written when it was BUILT, which still carries the short
    /// values, so a reader could not tell a fixed bar from a broken one and
    /// no test could prove the fix worked.
    int reconcileAgainst(const KlineBar* bars, int n,
                         KlineBar* out = nullptr, int max_out = 0);

    // Total bars ever emitted/ingested. A monotone counter — NOT a warmup
    // signal; use contiguousBars() for that.
    int64_t barsSeen() const { return mBarsSeen; }

    // Length of the CONTIGUOUS run of retained bars ending at the newest one.
    // This is what warmth must be judged on: 700 bars with a hole in them are
    // not 700 bars, and every rolling window spanning the hole is wrong.
    int  contiguousBars() const { return static_cast<int>(mHistory.size()); }

    // Newest retained bar, or nullptr when empty.
    const KlineBar* newestBar() const { return mHistory.empty() ? nullptr : &mHistory.back(); }

    // Retained bar by position, 0 = oldest. nullptr when out of range — the
    // caller gets nothing rather than a neighbouring bar. This is the window
    // the model will read in Phase 2, and it is what lets a test assert that a
    // repaired bucket holds the INGESTED kline and not a fabricated one.
    const KlineBar* barAt(int idx) const
    {
        if (idx < 0 || static_cast<size_t>(idx) >= mHistory.size()) return nullptr;
        return &mHistory[static_cast<size_t>(idx)];
    }

    // A discontinuity was seen and the pre-gap history QUARANTINED (not
    // destroyed — see mPending). The range is the buckets that are MISSING, so
    // an operator can fetch exactly them. These two are a record of the LAST
    // discontinuity and do not shrink as it is repaired; use missingFromMs() /
    // missingToMs() for the hole that is still outstanding right now.
    int64_t seamGaps() const { return mSeamGaps; }
    int64_t lastGapFromMs() const { return mLastGapFromMs; }
    int64_t lastGapToMs() const { return mLastGapToMs; }

    // --- the quarantine ---------------------------------------------------
    // Bars that were contiguous until a discontinuity cut them off from the
    // live run. They are RETAINED but deliberately NOT counted by
    // contiguousBars(), so warmth stays honest while the hole is open; feeding
    // the exact missing buckets through ingestBackfill() reunites them.
    //
    // WHY this exists: the boot seam is STRUCTURAL, not a fault. The backfill
    // CSV can only ever end at the last CLOSED bucket, the process then
    // attaches part-way through the next bucket B and must discard B as a
    // partial (we missed its early trades), so the first built bar is B+1 and
    // B is missing by construction. The old code cleared history at that point,
    // throwing away every backfill bar on every single start and falling back
    // to a 700-bar live warmup. Bucket B is fetchable the moment it closes, so
    // the hole is repairable — but only if the other 699 bars still exist.
    int  pendingBars() const { return static_cast<int>(mPending.size()); }

    // The buckets still missing between the quarantine and the live run,
    // inclusive. Both 0 when there is nothing outstanding. This is exactly the
    // range a caller must fetch and ingest to become contiguous again.
    int64_t missingFromMs() const
    {
        return mPending.empty() ? 0 : mPending.back().bucket_open_ms + mPeriodMs;
    }
    int64_t missingToMs() const
    {
        return (mPending.empty() || mHistory.empty())
                 ? 0 : mHistory.front().bucket_open_ms - mPeriodMs;
    }

    // Times a quarantine was successfully reunited with the live run.
    int64_t seamRepairs() const { return mSeamRepairs; }
    // Times an OUTSTANDING quarantine was dropped because a second
    // discontinuity opened; the older segment is then two holes away from live
    // and repairing it would need both closed, so only the newer is kept.
    int64_t quarantinesDiscarded() const { return mQuarantinesDiscarded; }
    // Bars whose bucket was not strictly newer than the retained history —
    // a rewind, not a gap. Never written into history (that would rewrite the
    // past); counted so a wrong-clock or stale feed is visible.
    int64_t rewoundBarsDropped() const { return mRewoundBarsDropped; }

    // Diagnostics for the parity report.
    int64_t tradeUpdates() const { return mTradeUpdates; }
    int64_t aggTradeUpdates() const { return mAggTradeUpdates; }
    // Close the open bucket if its END is at or before `cutoff_ms`, without
    // waiting for a tick to roll it. Returns 1 if it closed one, else 0.
    //
    // WHY. Bucket membership is by EXCHANGE timestamp, but closing was driven
    // by the arrival of the first tick of the NEXT bucket -- so a bar's latency
    // tracked how quiet the symbol was. Measured 2026-08-20 across 28 symbols:
    // 3.3-11.6 s past the boundary, p50 3.7 s, while recv->bar was 0.9 us.
    //
    // The caller passes now MINUS a grace. The grace covers trades ALREADY IN
    // FLIGHT -- a trade received after the boundary can still belong to the
    // bucket that just ended -- and nothing else. It is not a wait on the venue.
    //
    // IDEMPOTENT: once the bucket is closed there is no open bucket to close,
    // so polling this costs nothing and cannot double-emit.
    int     flushDue(int64_t cutoff_ms);

    // Bars whose venue columns were CORRECTED from the authoritative klines.
    // Nonzero means the tick stream lost data -- see reconcileAgainst.
    int64_t barsReconciled() const { return mBarsReconciled; }

    int64_t duplicatesDropped() const { return mDuplicatesDropped; }
    int64_t lateTradesDropped() const { return mLateTradesDropped; }
    int64_t partialBucketsDropped() const { return mPartialBucketsDropped; }
    int64_t badTimestampDropped() const { return mBadTimestampDropped; }

    // --- the DROPPED-TRADE detector (rule 7) ------------------------------
    // Run totals of the per-bar KlineBar::trade_id_gaps / n_trades_missing.
    // See the rule-7 banner at the top of this file for what they mean and for
    // the measurement that motivated them.
    int64_t tradeIdGaps() const { return mTradeIdGaps; }
    int64_t tradesMissing() const { return mTradesMissing; }

  private:
    struct Accum {
        int64_t bucket_open_ms{0};
        double  open{0.0}, high{0.0}, low{0.0}, close{0.0};
        double  volume{0.0}, quote_volume{0.0};
        int64_t n_trades{0};
        double  taker_buy_base{0.0}, taker_buy_quote{0.0};
        int64_t n_unclassified{0};
        int64_t trade_id_gaps{0};
        int64_t n_trades_missing{0};
        bool    any_exact{false};
        bool    any_quote_rule{false};
        bool    has_any_trade{false};
    };

    int64_t bucketOf(int64_t ts_ms) const;
    void    pushHistory(const KlineBar& b);
    void    startBucket(int64_t bucket_open_ms);
    void    emitCurrent(uint64_t close_trigger_recv_ns);
    void    emitFlat(int64_t bucket_open_ms);
    void    applyTrade(const TickEvent& ev);

    const int64_t mPeriodMs;
    const size_t  mMaxHistory;

    Accum   mCur{};
    bool    mHaveOpenBucket{false};
    bool    mCurIsPartial{false};
    // Rule 3 applies to the ONE bucket this process joined late, and to nothing
    // afterwards. Cleared the moment that bucket is disposed of, so a bucket
    // opened later -- after flushDue() closed its predecessor and left no open
    // bucket -- is correctly treated as fully observed. Conflating the two made
    // every quiet symbol on hydra go dark; see the banner on onTick().
    bool    mAttachPending{true};
    // The newest bucket this builder has CLOSED on the live path (emitted or
    // discarded), so a reopen after a flush knows which buckets it slept
    // through and owes flat bars for. Not derived from mHistory: that is
    // swapped out wholesale when a discontinuity quarantines the run.
    bool    mHaveLastClosed{false};
    int64_t mLastClosedBucketMs{0};
    bool    mHaveClose{false};
    double  mPrevClose{0.0};

    double  mLastBid{0.0}, mLastAsk{0.0};
    uint64_t mLastTradeId{0};
    bool     mHaveLastTradeId{false};

    int64_t mBarsSeen{0};
    int64_t mTradeUpdates{0}, mAggTradeUpdates{0};
    int64_t mDuplicatesDropped{0}, mLateTradesDropped{0}, mPartialBucketsDropped{0};
    int64_t mBarsReconciled{0};
    int64_t mBadTimestampDropped{0};
    int64_t mTradeIdGaps{0}, mTradesMissing{0};

    int64_t mSeamGaps{0};
    int64_t mLastGapFromMs{0}, mLastGapToMs{0};
    int64_t mSeamRepairs{0};
    int64_t mQuarantinesDiscarded{0};
    int64_t mRewoundBarsDropped{0};

    bool trySplice();

    std::deque<KlineBar> mReady;
    // Retained bars, oldest first, always contiguous by construction — a gap
    // moves everything before it into mPending, because only the contiguous
    // tail is usable as a rolling window.
    std::deque<KlineBar> mHistory;
    // The quarantined pre-gap run, oldest first, itself contiguous. Separated
    // from mHistory by exactly [missingFromMs() .. missingToMs()]. Never
    // counted toward warmth; only trySplice() can move it back.
    std::deque<KlineBar> mPending;
};

} // namespace agamotto
