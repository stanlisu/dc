#include "bar_builder.hpp"

#include <algorithm>
#include <stdexcept>

namespace mjolnir {

BarBuilder::BarBuilder(int bar_sec, int target_sec)
    : mBarSec(bar_sec), mTargetSec(target_sec)
{
    if (bar_sec <= 0) throw std::invalid_argument("bar_sec must be > 0");
    if (target_sec <= 0) throw std::invalid_argument("target_sec must be > 0");
}

int64_t BarBuilder::bucketOf(int64_t ts_ms) const
{
    const int64_t width = static_cast<int64_t>(mBarSec) * 1000;
    // Floor division. ts_ms is epoch-positive in practice, but a negative or
    // clock-skewed stamp must still floor DOWN, not truncate toward zero, or a
    // pre-epoch event would land in the wrong bucket.
    int64_t q = ts_ms / width;
    if (ts_ms < 0 && q * width != ts_ms) --q;
    return q * width;
}

void BarBuilder::reset(int64_t bucket)
{
    mCurrentBucket = bucket;
    // Carry last close into the new bar's OHLC, matching the reference: a gap
    // bucket with no trades must still emit a flat bar at the last price
    // rather than zeros.
    mOpen = mClose;
    mHigh = mClose;
    mLow  = mClose;
    mVolume = 0.0;
    mBuyVol = 0.0;
    mNTrades = 0;
    mDollarVolume = 0.0;
    mLiqLongNotional = 0.0;
    mLiqShortNotional = 0.0;
    mLiqLongCount = 0;
    mLiqShortCount = 0;
    // book / mark / OI are last-value fields and deliberately NOT reset.
}

void BarBuilder::applyBuffered(int64_t bucket)
{
    auto it = mBuffered.find(bucket);
    if (it == mBuffered.end()) return;
    Buffered& b = it->second;

    if (b.has_bt) {
        mBidPrice = b.bid_p; mBidAmount = b.bid_q;
        mAskPrice = b.ask_p; mAskAmount = b.ask_q;
    }
    if (b.has_depth) {
        for (int i = 0; i < BOOK_LEVELS; ++i) {
            mDepthBid[i]  = b.depth_bid[i];
            mDepthAsk[i]  = b.depth_ask[i];
            mBidsPrice[i] = b.bids_price[i];
            mBidsQty[i]   = b.bids_qty[i];
            mAsksPrice[i] = b.asks_price[i];
            mAsksQty[i]   = b.asks_qty[i];
        }
    }
    if (b.has_mark) {
        mMarkPrice = b.mark; mIndexPrice = b.index;
        mFundingRate = b.funding; mPredictedFunding = b.predicted_funding;
    }
    if (b.has_oi) {
        mOpenInterest = b.oi;
    }
    for (const auto& l : b.liqs) {
        // side BUY == LONG liquidation (matches the production convention,
        // which is the opposite of the Binance textbook reading).
        if (l.is_buy) { mLiqLongNotional += l.notional; ++mLiqLongCount; }
        else          { mLiqShortNotional += l.notional; ++mLiqShortCount; }
    }
    mBuffered.erase(it);
}

Bar BarBuilder::emit() const
{
    Bar bar;
    bar.bucket_ms = mCurrentBucket < 0 ? 0 : mCurrentBucket;
    bar.open = mOpen; bar.high = mHigh; bar.low = mLow; bar.close = mClose;
    bar.volume = mVolume;
    bar.buy_vol = mBuyVol;
    bar.sell_vol = mVolume - mBuyVol;
    bar.n_trades = mNTrades;
    bar.vwap = (mVolume > 0.0) ? (mDollarVolume / mVolume) : mClose;
    bar.trade_imbalance = (2.0 * mBuyVol - mVolume) / (mVolume + 1e-10);

    bar.bid_price = mBidPrice; bar.bid_amount = mBidAmount;
    bar.ask_price = mAskPrice; bar.ask_amount = mAskAmount;
    for (int i = 0; i < BOOK_LEVELS; ++i) {
        bar.bids_price[i] = mBidsPrice[i];
        bar.bids_qty[i]   = mBidsQty[i];
        bar.asks_price[i] = mAsksPrice[i];
        bar.asks_qty[i]   = mAsksQty[i];
    }
    // L1/L3/L5 are the CUMULATIVE depths at indices 0/2/4.
    bar.depth_bid_L1 = mDepthBid[0];
    bar.depth_bid_L3 = mDepthBid[2];
    bar.depth_bid_L5 = mDepthBid[4];
    bar.depth_ask_L1 = mDepthAsk[0];
    bar.depth_ask_L3 = mDepthAsk[2];
    bar.depth_ask_L5 = mDepthAsk[4];

    bar.mark_price = mMarkPrice;
    bar.index_price = mIndexPrice;
    bar.funding_rate = mFundingRate;
    bar.predicted_funding_rate = mPredictedFunding;
    bar.open_interest = mOpenInterest;

    bar.liq_long_notional = mLiqLongNotional;
    bar.liq_short_notional = mLiqShortNotional;
    bar.liq_long_count = mLiqLongCount;
    bar.liq_short_count = mLiqShortCount;
    bar.liq_total_count = mLiqLongCount + mLiqShortCount;

    const int64_t bucket_sec = bar.bucket_ms / 1000;
    const int64_t offset_sec = bucket_sec % mTargetSec;
    bar.cycle_progress = static_cast<double>(offset_sec) / static_cast<double>(mTargetSec);
    bar.secs_to_boundary = (mTargetSec - offset_sec) % mTargetSec;
    return bar;
}

bool BarBuilder::onTrade(double price, double qty, bool is_buyer_maker, int64_t ts_ms,
                         int64_t n_trades, Bar* out)
{
    const int64_t bucket = bucketOf(ts_ms);
    bool completed = false;

    if (mCurrentBucket < 0) {
        mCurrentBucket = bucket;
        mOpen = mHigh = mLow = mClose = price;
    } else if (bucket > mCurrentBucket) {
        const int64_t width = static_cast<int64_t>(mBarSec) * 1000;
        const int64_t prev_bucket = bucket - width;
        // Walk empty buckets ONE AT A TIME rather than jumping: each gap bucket
        // must get its own reset + buffered updates, otherwise a buffered event
        // stamped inside the gap would be applied to the wrong bar (or dropped).
        if (mCurrentBucket < prev_bucket) {
            int64_t next_bucket = mCurrentBucket + width;
            while (next_bucket <= prev_bucket) {
                reset(next_bucket);
                applyBuffered(next_bucket);
                next_bucket += width;
            }
        }
        if (out) *out = emit();
        completed = true;
        reset(bucket);
        applyBuffered(bucket);
        // Drop buffers stranded in the past; keeping them would let a stale
        // event surface on a later bucket.
        for (auto it = mBuffered.begin(); it != mBuffered.end();) {
            if (it->first < bucket) it = mBuffered.erase(it);
            else ++it;
        }
    }

    if (mNTrades == 0) {
        mOpen = price; mHigh = price; mLow = price;
    } else {
        mHigh = std::max(mHigh, price);
        mLow  = std::min(mLow, price);
    }
    mClose = price;
    mVolume += qty;
    if (!is_buyer_maker) mBuyVol += qty;   // taker BUY lifted the ask
    mNTrades += n_trades;
    mDollarVolume += price * qty;

    return completed;
}

void BarBuilder::onBookTicker(double bid_p, double bid_q, double ask_p, double ask_q,
                              int64_t ts_ms)
{
    const int64_t eb = bucketOf(ts_ms);
    if (mCurrentBucket >= 0 && eb > mCurrentBucket) {
        Buffered& b = bufferFor(eb);
        b.has_bt = true;
        b.bid_p = bid_p; b.bid_q = bid_q; b.ask_p = ask_p; b.ask_q = ask_q;
    } else {
        mBidPrice = bid_p; mBidAmount = bid_q;
        mAskPrice = ask_p; mAskAmount = ask_q;
    }
}

void BarBuilder::onDepth(const double* bid_px, const double* bid_qty,
                         const double* ask_px, const double* ask_qty, int n_levels,
                         int64_t ts_ms)
{
    double depth_bid[BOOK_LEVELS]{}, depth_ask[BOOK_LEVELS]{};
    double bp[BOOK_LEVELS]{}, bq[BOOK_LEVELS]{}, ap[BOOK_LEVELS]{}, aq[BOOK_LEVELS]{};
    double cum_bid = 0.0, cum_ask = 0.0;
    for (int i = 0; i < BOOK_LEVELS; ++i) {
        if (i < n_levels) {
            bp[i] = bid_px[i]; bq[i] = bid_qty[i];
            ap[i] = ask_px[i]; aq[i] = ask_qty[i];
        }
        cum_bid += bq[i];
        cum_ask += aq[i];
        depth_bid[i] = cum_bid;
        depth_ask[i] = cum_ask;
    }

    const int64_t eb = bucketOf(ts_ms);
    if (mCurrentBucket >= 0 && eb > mCurrentBucket) {
        Buffered& b = bufferFor(eb);
        b.has_depth = true;
        for (int i = 0; i < BOOK_LEVELS; ++i) {
            b.depth_bid[i] = depth_bid[i]; b.depth_ask[i] = depth_ask[i];
            b.bids_price[i] = bp[i]; b.bids_qty[i] = bq[i];
            b.asks_price[i] = ap[i]; b.asks_qty[i] = aq[i];
        }
    } else {
        for (int i = 0; i < BOOK_LEVELS; ++i) {
            mDepthBid[i] = depth_bid[i]; mDepthAsk[i] = depth_ask[i];
            mBidsPrice[i] = bp[i]; mBidsQty[i] = bq[i];
            mAsksPrice[i] = ap[i]; mAsksQty[i] = aq[i];
        }
    }
}

void BarBuilder::onMarkPrice(double mark, double index, double funding,
                             double predicted_funding, int64_t ts_ms)
{
    const int64_t eb = bucketOf(ts_ms);
    if (mCurrentBucket >= 0 && eb > mCurrentBucket) {
        Buffered& b = bufferFor(eb);
        b.has_mark = true;
        b.mark = mark; b.index = index;
        b.funding = funding; b.predicted_funding = predicted_funding;
    } else {
        mMarkPrice = mark; mIndexPrice = index;
        mFundingRate = funding; mPredictedFunding = predicted_funding;
    }
}

void BarBuilder::onLiquidation(bool side_is_buy, double notional, int64_t ts_ms)
{
    const int64_t eb = bucketOf(ts_ms);
    if (mCurrentBucket >= 0 && eb > mCurrentBucket) {
        Buffered& b = bufferFor(eb);
        b.liqs.push_back(Buffered::Liq{side_is_buy, notional});
    } else {
        if (side_is_buy) { mLiqLongNotional += notional; ++mLiqLongCount; }
        else             { mLiqShortNotional += notional; ++mLiqShortCount; }
    }
}

void BarBuilder::setOpenInterest(double oi, int64_t ts_ms)
{
    // Same bucket discipline as every other last-value stream: an OI snapshot
    // stamped in a future bucket must not leak into the still-open bar.
    const int64_t eb = bucketOf(ts_ms);
    if (mCurrentBucket >= 0 && eb > mCurrentBucket) {
        Buffered& b = bufferFor(eb);
        b.has_oi = true;
        b.oi = oi;
    } else {
        mOpenInterest = oi;
    }
}

} // namespace mjolnir
