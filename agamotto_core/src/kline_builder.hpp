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

#include <cstdint>
#include <deque>
#include <vector>

#include "agamotto_core.hpp"

namespace agamotto {

class KlineBuilder {
  public:
    // period_sec must divide 86400 — checked by the caller; buckets tile the
    // UTC day exactly as Binance's do.
    explicit KlineBuilder(int period_sec);

    // Feed one adapted tick. Completed bars are appended to the ready queue.
    void onTick(const TickEvent& ev);

    // Pop one completed bar. False when none pending.
    bool pop(KlineBar* out);

    // Seed closed bars (REST backfill). Oldest-first, on-grid, contiguous.
    // Returns false and ingests NOTHING on any violation — a silently accepted
    // hole would train the 700-bar rolling windows on a lie.
    bool ingestBackfill(const KlineBar* bars, int n);

    int64_t barsSeen() const { return mBarsSeen; }

    // Diagnostics for the parity report.
    int64_t tradeUpdates() const { return mTradeUpdates; }
    int64_t aggTradeUpdates() const { return mAggTradeUpdates; }
    int64_t duplicatesDropped() const { return mDuplicatesDropped; }
    int64_t lateTradesDropped() const { return mLateTradesDropped; }
    int64_t partialBucketsDropped() const { return mPartialBucketsDropped; }
    int64_t badTimestampDropped() const { return mBadTimestampDropped; }

  private:
    struct Accum {
        int64_t bucket_open_ms{0};
        double  open{0.0}, high{0.0}, low{0.0}, close{0.0};
        double  volume{0.0}, quote_volume{0.0};
        int64_t n_trades{0};
        double  taker_buy_base{0.0}, taker_buy_quote{0.0};
        int64_t n_unclassified{0};
        bool    any_exact{false};
        bool    any_quote_rule{false};
        bool    has_any_trade{false};
        uint64_t last_recv_ns{0};
    };

    int64_t bucketOf(int64_t ts_ms) const;
    void    startBucket(int64_t bucket_open_ms, uint64_t recv_ns);
    void    emitCurrent(uint64_t close_trigger_recv_ns);
    void    emitFlat(int64_t bucket_open_ms);
    void    applyTrade(const TickEvent& ev);

    const int64_t mPeriodMs;

    Accum   mCur{};
    bool    mHaveOpenBucket{false};
    bool    mCurIsPartial{false};
    bool    mHaveClose{false};
    double  mPrevClose{0.0};

    double  mLastBid{0.0}, mLastAsk{0.0};
    uint64_t mLastTradeId{0};
    bool     mHaveLastTradeId{false};

    int64_t mBarsSeen{0};
    int64_t mTradeUpdates{0}, mAggTradeUpdates{0};
    int64_t mDuplicatesDropped{0}, mLateTradesDropped{0}, mPartialBucketsDropped{0};
    int64_t mBadTimestampDropped{0};

    std::deque<KlineBar> mReady;
};

} // namespace agamotto
