// Private core — implements the public opaque contract.
//
// M1 SCAFFOLD. The contract, lifecycle, config plumbing and build tagging are
// real; the four IP modules (bar builder, feature engine, regime gate, model
// runner) are not yet implemented and are marked NOT_IMPLEMENTED below.
//
// Deliberate design choice: an unimplemented stage FAILS LOUD rather than
// returning a neutral value. A core that quietly returned "no bar" or "no fire"
// would be indistinguishable from a working core on a quiet day, which is
// exactly how a broken shadow run gets mistaken for a clean one.
#include "mjolnir_core.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MJOLNIR_CORE_GITSHA
#define MJOLNIR_CORE_GITSHA "unknown"
#endif

namespace mjolnir {
namespace {

[[noreturn]] void notImplemented(const char* what)
{
    throw std::logic_error(std::string("mjolnir_core: ") + what
                           + " not implemented yet (M1 scaffold). "
                             "Refusing to return a neutral value that would look like a "
                             "working core on a quiet day.");
}

class Core final : public ICore {
  public:
    explicit Core(const std::string& paramsPath) : mParamsPath(paramsPath)
    {
        std::ifstream ifs_(paramsPath);
        if (!ifs_.is_open()) {
            throw std::runtime_error("mjolnir_core: cannot open algo_params: " + paramsPath);
        }
        // Config parsing lands with the bar builder — it needs the bar period,
        // warmup gate, coded regime stack path and weights dir together.
    }

    void onTick(const TickEvent&) override
    {
        // bar_builder.cpp: accumulate into the current bucket; close on the
        // first trade of a NEW bucket (next-trade close, matching the reference
        // implementation — never wall-clock).
        notImplemented("bar builder");
    }

    bool barReady(int64_t*) override { notImplemented("bar builder"); }

    bool isWarm() const override { return mBars >= mWarmupBars; }
    int  barsBuffered() const override { return mBars; }

    bool isAnchor() const override { return mIsAnchor; }
    int  anchorFrameSize() const override { return static_cast<int>(mAnchorFrame.size()); }
    const double* anchorFrame() const override
    {
        return mAnchorFrame.empty() ? nullptr : mAnchorFrame.data();
    }

    void setAnchorFrame(const double* frame, int n, int64_t barTsNs) override
    {
        if (frame == nullptr || n <= 0) {
            // Absent anchor state must propagate as a hard error at decide()
            // time, mirroring the reference contract where a peer without the
            // anchor frame raises rather than silently skipping cross-features.
            mAnchorFrame.clear();
            mAnchorBarTs = 0;
            return;
        }
        mAnchorFrame.assign(frame, frame + n);
        mAnchorBarTs = barTsNs;
    }

    Decision decide() override
    {
        // feature_engine -> regime_gate -> model_runner -> vote -> hold-TTL.
        notImplemented("feature engine / regime gate / model runner");
    }

    int dumpPredictions(uint16_t*, int*, double*, int) const override
    {
        notImplemented("model runner");
    }

  private:
    std::string mParamsPath;
    int mBars{0};
    int mWarmupBars{0};
    bool mIsAnchor{false};
    std::vector<double> mAnchorFrame;
    int64_t mAnchorBarTs{0};
};

} // namespace

std::unique_ptr<ICore> makeCore(const std::string& algo_params_json)
{
    return std::unique_ptr<ICore>(new Core(algo_params_json));
}

bool coreIsRealImplementation() { return true; }

const char* coreBuildTag() { return "core-" MJOLNIR_CORE_GITSHA; }

} // namespace mjolnir
