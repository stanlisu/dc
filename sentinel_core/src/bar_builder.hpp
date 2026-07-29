#pragma once
// Tick -> bar accumulator. PRIVATE: the bar schema is IP (this is the C++ port
// of the reconciled reference builder, which is excluded from the public repo).
//
// Semantics are a 1:1 port and must stay that way — parity is measured against
// the live bot's own dumped bars, so any divergence here shows up as a feature
// divergence downstream. The load-bearing rules:
//   * bars close on the FIRST TRADE OF A NEW BUCKET (next-trade), never on
//     wall-clock;
//   * an event stamped in a FUTURE bucket is buffered, never applied to the
//     still-open bar (that is lookahead);
//   * empty buckets are walked one at a time so last-value fields carry and
//     each gap bucket gets its own buffered updates;
//   * reset() carries close -> open and keeps last-value fields.
#include <cstdint>
#include <map>
#include <vector>

namespace mjolnir {

inline constexpr int BOOK_LEVELS = 5;

// One completed bar. Field set and derivations mirror the reference emit()
// exactly; the caller stamps symbol/timestamp around it.
struct Bar {
    int64_t bucket_ms{0};   // bar-open epoch ms

    double open{0.0}, high{0.0}, low{0.0}, close{0.0};
    double volume{0.0}, buy_vol{0.0}, sell_vol{0.0};
    int64_t n_trades{0};
    double vwap{0.0};
    double trade_imbalance{0.0};

    double bid_price{0.0}, bid_amount{0.0}, ask_price{0.0}, ask_amount{0.0};

    double bids_price[BOOK_LEVELS]{};
    double bids_qty[BOOK_LEVELS]{};
    double asks_price[BOOK_LEVELS]{};
    double asks_qty[BOOK_LEVELS]{};

    double depth_bid_L1{0.0}, depth_bid_L3{0.0}, depth_bid_L5{0.0};
    double depth_ask_L1{0.0}, depth_ask_L3{0.0}, depth_ask_L5{0.0};

    double mark_price{0.0}, index_price{0.0};
    double funding_rate{0.0}, predicted_funding_rate{0.0};
    double open_interest{0.0};

    double liq_long_notional{0.0}, liq_short_notional{0.0};
    int64_t liq_long_count{0}, liq_short_count{0}, liq_total_count{0};

    double cycle_progress{0.0};
    int64_t secs_to_boundary{0};
};

class BarBuilder {
  public:
    // target_sec drives cycle_progress / secs_to_boundary. Defaulting it to
    // bar_sec would silently produce the constant-0 boundary columns even when
    // the model was trained on boundary-mode features, so it is REQUIRED.
    BarBuilder(int bar_sec, int target_sec);

    // Returns true and fills `out` when this trade closed the PREVIOUS bucket.
    bool onTrade(double price, double qty, bool is_buyer_maker, int64_t ts_ms,
                 int64_t n_trades, Bar* out);

    void onBookTicker(double bid_p, double bid_q, double ask_p, double ask_q, int64_t ts_ms);
    void onDepth(const double* bid_px, const double* bid_qty,
                 const double* ask_px, const double* ask_qty, int n_levels, int64_t ts_ms);
    void onMarkPrice(double mark, double index, double funding, double predicted_funding,
                     int64_t ts_ms);
    void onLiquidation(bool side_is_buy, double notional, int64_t ts_ms);
    void setOpenInterest(double oi, int64_t ts_ms);

    bool started() const { return mCurrentBucket >= 0; }
    int64_t currentBucket() const { return mCurrentBucket; }

  private:
    struct Buffered {
        bool has_bt{false};
        double bid_p{0}, bid_q{0}, ask_p{0}, ask_q{0};
        bool has_depth{false};
        double depth_bid[BOOK_LEVELS]{}, depth_ask[BOOK_LEVELS]{};
        double bids_price[BOOK_LEVELS]{}, bids_qty[BOOK_LEVELS]{};
        double asks_price[BOOK_LEVELS]{}, asks_qty[BOOK_LEVELS]{};
        bool has_mark{false};
        double mark{0}, index{0}, funding{0}, predicted_funding{0};
        bool has_oi{false};
        double oi{0};
        struct Liq { bool is_buy; double notional; };
        std::vector<Liq> liqs;
    };

    int64_t bucketOf(int64_t ts_ms) const;
    void reset(int64_t bucket);
    void applyBuffered(int64_t bucket);
    Buffered& bufferFor(int64_t bucket) { return mBuffered[bucket]; }
    Bar emit() const;

    int mBarSec;
    int mTargetSec;
    int64_t mCurrentBucket{-1};

    // trade accumulators
    double mOpen{0}, mHigh{0}, mLow{0}, mClose{0};
    double mVolume{0}, mBuyVol{0}, mDollarVolume{0};
    int64_t mNTrades{0};

    // last-value fields (carried across resets)
    double mBidPrice{0}, mBidAmount{0}, mAskPrice{0}, mAskAmount{0};
    double mDepthBid[BOOK_LEVELS]{}, mDepthAsk[BOOK_LEVELS]{};
    double mBidsPrice[BOOK_LEVELS]{}, mBidsQty[BOOK_LEVELS]{};
    double mAsksPrice[BOOK_LEVELS]{}, mAsksQty[BOOK_LEVELS]{};
    double mMarkPrice{0}, mIndexPrice{0}, mFundingRate{0}, mPredictedFunding{0};
    double mOpenInterest{0};

    // per-bar liquidation accumulators
    double mLiqLongNotional{0}, mLiqShortNotional{0};
    int64_t mLiqLongCount{0}, mLiqShortCount{0};

    std::map<int64_t, Buffered> mBuffered;
};

} // namespace mjolnir
