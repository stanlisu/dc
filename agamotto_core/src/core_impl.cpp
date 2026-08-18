// Real implementation of the agamotto ICore contract. PRIVATE.
//
// Phase 1 implements the bar layer only: ticks -> 15m Binance-shaped klines,
// plus REST backfill ingestion and the warmup counter. decide() is deliberately
// inert and says so — the feature engine, regime gate and model land in Phase 2.
#include "agamotto_core.hpp"
#include "kline_builder.hpp"

#include <memory>
#include <string>

#ifndef AGAMOTTO_CORE_GITSHA
#error "AGAMOTTO_CORE_GITSHA must be defined so a run traces back to its core"
#endif

namespace agamotto {
namespace {

class RealCore final : public ICore {
  public:
    RealCore(int64_t product_id, int bar_sec, int warmup_bars)
      : mProductId(product_id), mWarmupBars(warmup_bars), mBuilder(bar_sec)
    {
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

    bool barReady(KlineBar* out) override { return mBuilder.pop(out); }

    bool isWarm() const override { return barsBuffered() >= mWarmupBars; }
    int  barsBuffered() const override { return static_cast<int>(mBuilder.barsSeen()); }
    int  warmupRequirement() const override { return mWarmupBars; }

    // Phase 1: never fires. Returning a default-constructed Decision (fired =
    // false) is the whole contract here; a caller that mistakes this for "no
    // signal today" is told otherwise by the strategy's own log line.
    Decision decide() override { return Decision{}; }

    bool coreIsRealImplementation() const override { return true; }

    std::string coreBuildTag() const override
    {
        return std::string("agamotto-core-") + AGAMOTTO_CORE_GITSHA + "-phase1-bars";
    }

  private:
    const int64_t mProductId;
    const int     mWarmupBars;
    KlineBuilder  mBuilder;
    int64_t       mBackfilled{0};
    int64_t       mForeignProductTicks{0};
};

} // namespace

std::unique_ptr<ICore> createCore(int64_t product_id, int bar_sec, int warmup_bars)
{
    return std::unique_ptr<ICore>(new RealCore(product_id, bar_sec, warmup_bars));
}

} // namespace agamotto
