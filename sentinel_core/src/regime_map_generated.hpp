// GENERATED from dc/obfuscation/map.json — do not edit by hand.
// Regenerate: python sentinel_core/tools/gen_regime_map.py
//
// PRIVATE artifact. Real names live here because this compiles into the private
// core; the PUBLIC repo never sees them (its leak gate enforces that).
#pragma once
#include <string>
#include <unordered_map>

namespace mjolnir {
inline const std::unordered_map<std::string, std::string>& regimeCodeToName()
{
    static const std::unordered_map<std::string, std::string> M = {
    {"r001", "above_all_mas"},
    {"r002", "above_ma20"},
    {"r003", "adx_trend"},
    {"r004", "ask_heavy"},
    {"r005", "backwardation"},
    {"r006", "basis_discount"},
    {"r007", "basis_premium"},
    {"r008", "bb_rebound"},
    {"r009", "below_ma20"},
    {"r010", "bid_heavy"},
    {"r011", "bop_bearish"},
    {"r012", "bop_bullish"},
    {"r013", "btc_high_vol"},
    {"r014", "btc_low_vol"},
    {"r015", "btc_trending_down"},
    {"r016", "btc_trending_up"},
    {"r017", "buy_flow"},
    {"r018", "buy_pressure"},
    {"r019", "cci_reversal"},
    {"r020", "contango"},
    {"r021", "deep_book"},
    {"r022", "funding_negative"},
    {"r023", "funding_positive"},
    {"r024", "high_hv"},
    {"r025", "high_iv_rank"},
    {"r026", "high_liquidation_pressure"},
    {"r027", "high_spread"},
    {"r028", "high_vol"},
    {"r029", "high_volume"},
    {"r030", "iv_discount"},
    {"r031", "iv_premium"},
    {"r032", "liq_spike"},
    {"r033", "long_liq_spike"},
    {"r034", "low_hv"},
    {"r035", "low_iv_rank"},
    {"r036", "low_liquidation_pressure"},
    {"r037", "low_spread"},
    {"r038", "low_vol"},
    {"r039", "low_volume"},
    {"r040", "ma_momentum"},
    {"r041", "macd_bearish"},
    {"r042", "macd_bullish"},
    {"r043", "mfi_overbought"},
    {"r044", "mfi_oversold"},
    {"r045", "mom_positive"},
    {"r046", "momentum_down"},
    {"r047", "momentum_up"},
    {"r048", "near_ma"},
    {"r049", "negative_skew"},
    {"r050", "ofi_negative"},
    {"r051", "ofi_positive"},
    {"r052", "oi_contraction"},
    {"r053", "oi_expanding"},
    {"r054", "oi_expansion"},
    {"r055", "positive_skew"},
    {"r056", "pre_funding_settlement"},
    {"r057", "roc_negative"},
    {"r058", "roc_positive"},
    {"r059", "rsi_overbought"},
    {"r060", "rsi_oversold"},
    {"r061", "sar_aligned"},
    {"r062", "sell_flow"},
    {"r063", "short_liq_spike"},
    {"r064", "stoch_bullish"},
    {"r065", "strong_candle"},
    {"r066", "strong_trend"},
    {"r067", "tight_spread"},
    {"r068", "trade_imbalance"},
    {"r069", "vol_breakout"},
    {"r070", "vol_compression"},
    {"r071", "vol_expansion"},
    {"r072", "wide_spread"},
    };
    return M;
}
} // namespace mjolnir
