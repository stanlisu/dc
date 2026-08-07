#pragma once
// The warmup fire gate: how many bars have EVER been fed, versus the configured
// gate. A separate type so it can be unit-tested on its own — constructing a
// Core needs weights, a regime stack and LightGBM, which is why the defect below
// had no cheap test to catch it.
//
// WHY THIS IS NOT `mBuf.size()`
// -----------------------------
// The reference counts bars cumulatively and never decays the count
// (knull/mjolnir_bridge.py: `_count_bar_fed` increments a plain dict entry,
// `_warmup_gate_ok` compares it against WARMUP_FIRE_GATE_BARS). The bar buffer
// is a different thing: it is trimmed to BUFFER_MAXLEN (1000) because the
// feature panel and the regime quantiles are DEFINED over exactly that window.
//
// So a gate ABOVE BUFFER_MAXLEN is legitimate and deliberate — production runs
// warmup_fire_gate_bars = 1080, i.e. "let 1080 bars flow through before trusting
// the most recent 1000". Comparing that gate against the trimmed buffer's size
// makes it permanently unreachable, and does so silently: the buffer sits at
// 1000, the progress log reports 1000, and nothing ever errors.
//
// That is not hypothetical. On hydra, pid 501412 logged "warming bars=1000" 103
// times over 3d22h and emitted ZERO [MJDEC] lines
// (~/mjolnir_shadow_bundle/deploy.log, 2026-08-07).
//
// Corollary, deliberately NOT enforced: gate > BUFFER_MAXLEN is a valid config
// and must not raise.
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mjolnir {

class WarmupGate {
  public:
    explicit WarmupGate(int gate_bars) : mGate(gate_bars)
    {
        // A negative gate would make the core warm before a single bar closed.
        // No default and no clamp — an out-of-range config raises.
        if (gate_bars < 0)
            throw std::runtime_error("warmup_fire_gate_bars must be >= 0, got "
                                     + std::to_string(gate_bars));
    }

    // One closed bar was fed to the core. Monotonic: never trimmed, never reset.
    void countBarFed() { ++mBarsFed; }

    bool isWarm() const { return mBarsFed >= static_cast<int64_t>(mGate); }

    // Cumulative bars fed — what the warmup progress line must report, so a
    // stalled gate is visible instead of reading as a saturated buffer.
    int64_t barsFed() const { return mBarsFed; }

    int gate() const { return mGate; }

  private:
    int64_t mBarsFed{0};
    int     mGate{0};
};

} // namespace mjolnir
