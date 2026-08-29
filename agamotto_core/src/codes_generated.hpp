// GENERATED from dc/obfuscation/map.json — do not edit by hand.
//
// CODES ONLY. The identifiers below are C++ symbols (they compile away); the
// VALUES are the obfuscation codes. No real regime or feature name appears as a
// string literal, so none can be recovered from the built .so with `strings`.
//
// This replaces the earlier code->name / name->code tables, which embedded all
// 172 real names in the binary (118 were recoverable) — the artifact was no more
// opaque than the vendored map.json that ships with the Python packages.
//
// Passthrough columns (OHLCV, timestamps, depth_bid_L*, ofi_L*, bids_*/asks_*)
// are deliberately NOT coded: the obfuscation map does not cover them, they are
// universal market-data field names, and they carry no strategy information.
#pragma once
#include <cstdint>

namespace agamotto {
namespace codes {

// ---- features: column keys used by the engine and the model ----------------
inline constexpr const char* F_ACF_LAG1 = "f001";
inline constexpr const char* F_AD = "f002";
inline constexpr const char* F_AD_SLOPE = "f101";
inline constexpr const char* F_ADX = "f003";
inline constexpr const char* F_ASK_IV = "f004";
inline constexpr const char* F_ATM_CALL_PRICE_CLOSE = "f005";
inline constexpr const char* F_ATM_CALL_PRICE_HIGH = "f006";
inline constexpr const char* F_ATM_CALL_PRICE_OPEN = "f007";
inline constexpr const char* F_ATM_PUT_PRICE_CLOSE = "f008";
inline constexpr const char* F_ATM_PUT_PRICE_HIGH = "f009";
inline constexpr const char* F_ATM_PUT_PRICE_OPEN = "f010";
inline constexpr const char* F_ATM_STRADDLE_PRICE_CLOSE = "f011";
inline constexpr const char* F_ATM_STRADDLE_PRICE_HIGH = "f012";
inline constexpr const char* F_ATM_STRADDLE_PRICE_OPEN = "f013";
inline constexpr const char* F_ATR = "f014";
inline constexpr const char* F_BASIS_PCT = "f015";
inline constexpr const char* F_BB_LOWER = "f016";
inline constexpr const char* F_BB_PCTB = "f102";
inline constexpr const char* F_BB_UPPER = "f017";
inline constexpr const char* F_BB_WIDTH = "f103";
inline constexpr const char* F_BID_IV = "f018";
inline constexpr const char* F_BOP = "f019";
inline constexpr const char* F_BTC_BOOK_IMBALANCE_L1 = "f020";
inline constexpr const char* F_BTC_LIQ_DIRECTIONAL = "f021";
inline constexpr const char* F_BTC_MID_RETURN_LAG1 = "f022";
inline constexpr const char* F_BTC_MID_RETURN_LAG4 = "f023";
inline constexpr const char* F_BTC_OFI_L1 = "f024";
inline constexpr const char* F_BTC_SPREAD_RATIO = "f025";
inline constexpr const char* F_BTC_TRADE_IMBALANCE = "f026";
inline constexpr const char* F_BUY_PRESSURE = "f027";
inline constexpr const char* F_CCI = "f028";
inline constexpr const char* F_CMO = "f029";
inline constexpr const char* F_CYCLE_PROGRESS = "f030";
inline constexpr const char* F_DELTA = "f031";
inline constexpr const char* F_DELTA_CALL = "f032";
inline constexpr const char* F_DEPTH_IMBALANCE_L1 = "f033";
inline constexpr const char* F_DEPTH_IMBALANCE_L3 = "f034";
inline constexpr const char* F_DEPTH_IMBALANCE_L5 = "f035";
inline constexpr const char* F_DOLLAR_TSI = "f036";
inline constexpr const char* F_DX = "f037";
inline constexpr const char* F_GAMMA = "f038";
inline constexpr const char* F_HIGH_OPEN_PCT = "f039";
inline constexpr const char* F_KURT = "f040";
inline constexpr const char* F_KYLE_LAMBDA = "f041";
inline constexpr const char* F_LIQ_BURST_RATIO = "f042";
inline constexpr const char* F_LIQ_DIRECTIONAL_IMBALANCE = "f043";
inline constexpr const char* F_LIQ_LONG_NOTIONAL = "f044";
inline constexpr const char* F_LIQ_SHORT_NOTIONAL = "f045";
inline constexpr const char* F_LIQ_TOTAL_NOTIONAL = "f046";
inline constexpr const char* F_LOW_OPEN_PCT = "f047";
inline constexpr const char* F_MACD = "f048";
inline constexpr const char* F_MACD_NORM = "f104";
inline constexpr const char* F_MACDHIST = "f049";
inline constexpr const char* F_MACDHIST_NORM = "f105";
inline constexpr const char* F_MARK_IV = "f050";
inline constexpr const char* F_MFI = "f051";
inline constexpr const char* F_MICROPRICE = "f052";
inline constexpr const char* F_MICROPRICE_VS_MID = "f053";
inline constexpr const char* F_MID_PRICE = "f054";
inline constexpr const char* F_MINUS_DI = "f055";
inline constexpr const char* F_MOM = "f056";
inline constexpr const char* F_NATR = "f057";
inline constexpr const char* F_NOTIONAL = "f058";
inline constexpr const char* F_OBV = "f059";
inline constexpr const char* F_OBV_SLOPE = "f106";
inline constexpr const char* F_OFI_AGG = "f060";
inline constexpr const char* F_OI_ACCELERATION = "f061";
inline constexpr const char* F_OI_VELOCITY = "f062";
inline constexpr const char* F_OPEN_CLOSE_DIFF = "f063";
inline constexpr const char* F_OPEN_CLOSE_PCT = "f064";
inline constexpr const char* F_PARKINSON_VOL = "f065";
inline constexpr const char* F_PLUS_DI = "f066";
inline constexpr const char* F_PRE_FUNDING = "f067";
inline constexpr const char* F_PRICE_RANGE = "f068";
inline constexpr const char* F_PRICE_RANGE_PCT = "f069";
inline constexpr const char* F_PRICE_RANGE_PCT_Q50 = "f070";
inline constexpr const char* F_PRICE_RANGE_PCT_Q80 = "f108";
inline constexpr const char* F_PRICE_RANGE_PCT_Q90 = "f109";
inline constexpr const char* F_PRICE_RANGE_PCT_Q95 = "f110";
inline constexpr const char* F_PV = "f071";
inline constexpr const char* F_QUOTE_VOL_RATIO = "f072";
inline constexpr const char* F_RELATIVE_SPREAD = "f073";
inline constexpr const char* F_RET_LAG1 = "f074";
inline constexpr const char* F_RET_LAG2 = "f075";
inline constexpr const char* F_RET_LAG3 = "f076";
inline constexpr const char* F_ROC = "f077";
inline constexpr const char* F_RSI = "f078";
inline constexpr const char* F_RSI_28 = "f079";
inline constexpr const char* F_RSI_7 = "f080";
inline constexpr const char* F_SAR = "f081";
inline constexpr const char* F_SAR_DIST = "f107";
inline constexpr const char* F_SECS_TO_BOUNDARY = "f082";
inline constexpr const char* F_SKEW = "f083";
inline constexpr const char* F_SPREAD = "f084";
inline constexpr const char* F_STD = "f085";
inline constexpr const char* F_STOCH_D = "f086";
inline constexpr const char* F_STOCH_K = "f087";
inline constexpr const char* F_STOCHRSI_D = "f088";
inline constexpr const char* F_STOCHRSI_K = "f089";
inline constexpr const char* F_THETA = "f090";
inline constexpr const char* F_TRADE_IMBALANCE = "f091";
inline constexpr const char* F_TRADE_INTENSITY = "f092";
inline constexpr const char* F_TRIX = "f093";
inline constexpr const char* F_ULTOSC = "f094";
inline constexpr const char* F_VEGA = "f095";
inline constexpr const char* F_VOL_RATIO = "f096";
inline constexpr const char* F_VOL_RET_LAG1 = "f097";
inline constexpr const char* F_VOL_RET_LAG2 = "f098";
inline constexpr const char* F_VOL_RET_LAG3 = "f099";
inline constexpr const char* F_WILLR = "f100";

// ---- regimes: numeric codes for gate dispatch ------------------------------
inline constexpr uint16_t R_ABOVE_ALL_MAS = 1;
inline constexpr uint16_t R_ABOVE_MA20 = 2;
inline constexpr uint16_t R_ADX_TREND = 3;
inline constexpr uint16_t R_ASK_HEAVY = 4;
inline constexpr uint16_t R_BACKWARDATION = 5;
inline constexpr uint16_t R_BASIS_DISCOUNT = 6;
inline constexpr uint16_t R_BASIS_PREMIUM = 7;
inline constexpr uint16_t R_BB_REBOUND = 8;
inline constexpr uint16_t R_BELOW_MA20 = 9;
inline constexpr uint16_t R_BID_HEAVY = 10;
inline constexpr uint16_t R_BOP_BEARISH = 11;
inline constexpr uint16_t R_BOP_BULLISH = 12;
inline constexpr uint16_t R_BTC_HIGH_VOL = 13;
inline constexpr uint16_t R_BTC_LOW_VOL = 14;
inline constexpr uint16_t R_BTC_TRENDING_DOWN = 15;
inline constexpr uint16_t R_BTC_TRENDING_UP = 16;
inline constexpr uint16_t R_BUY_FLOW = 17;
inline constexpr uint16_t R_BUY_PRESSURE = 18;
inline constexpr uint16_t R_CCI_REVERSAL = 19;
inline constexpr uint16_t R_CONTANGO = 20;
inline constexpr uint16_t R_DEEP_BOOK = 21;
inline constexpr uint16_t R_FUNDING_NEGATIVE = 22;
inline constexpr uint16_t R_FUNDING_POSITIVE = 23;
inline constexpr uint16_t R_HIGH_HV = 24;
inline constexpr uint16_t R_HIGH_IV_RANK = 25;
inline constexpr uint16_t R_HIGH_LIQUIDATION_PRESSURE = 26;
inline constexpr uint16_t R_HIGH_SPREAD = 27;
inline constexpr uint16_t R_HIGH_VOL = 28;
inline constexpr uint16_t R_HIGH_VOL_Q80 = 73;
inline constexpr uint16_t R_HIGH_VOL_Q90 = 74;
inline constexpr uint16_t R_HIGH_VOL_Q95 = 75;
inline constexpr uint16_t R_HIGH_VOLUME = 29;
inline constexpr uint16_t R_IV_DISCOUNT = 30;
inline constexpr uint16_t R_IV_PREMIUM = 31;
inline constexpr uint16_t R_LIQ_SPIKE = 32;
inline constexpr uint16_t R_LONG_LIQ_SPIKE = 33;
inline constexpr uint16_t R_LOW_HV = 34;
inline constexpr uint16_t R_LOW_IV_RANK = 35;
inline constexpr uint16_t R_LOW_LIQUIDATION_PRESSURE = 36;
inline constexpr uint16_t R_LOW_SPREAD = 37;
inline constexpr uint16_t R_LOW_VOL = 38;
inline constexpr uint16_t R_LOW_VOLUME = 39;
inline constexpr uint16_t R_MA_MOMENTUM = 40;
inline constexpr uint16_t R_MACD_BEARISH = 41;
inline constexpr uint16_t R_MACD_BULLISH = 42;
inline constexpr uint16_t R_MFI_OVERBOUGHT = 43;
inline constexpr uint16_t R_MFI_OVERSOLD = 44;
inline constexpr uint16_t R_MOM_POSITIVE = 45;
inline constexpr uint16_t R_MOMENTUM_DOWN = 46;
inline constexpr uint16_t R_MOMENTUM_UP = 47;
inline constexpr uint16_t R_NEAR_MA = 48;
inline constexpr uint16_t R_NEGATIVE_SKEW = 49;
inline constexpr uint16_t R_OFI_NEGATIVE = 50;
inline constexpr uint16_t R_OFI_POSITIVE = 51;
inline constexpr uint16_t R_OI_CONTRACTION = 52;
inline constexpr uint16_t R_OI_EXPANDING = 53;
inline constexpr uint16_t R_OI_EXPANSION = 54;
inline constexpr uint16_t R_POSITIVE_SKEW = 55;
inline constexpr uint16_t R_PRE_FUNDING_SETTLEMENT = 56;
inline constexpr uint16_t R_ROC_NEGATIVE = 57;
inline constexpr uint16_t R_ROC_POSITIVE = 58;
inline constexpr uint16_t R_RSI_OVERBOUGHT = 59;
inline constexpr uint16_t R_RSI_OVERSOLD = 60;
inline constexpr uint16_t R_SAR_ALIGNED = 61;
inline constexpr uint16_t R_SELL_FLOW = 62;
inline constexpr uint16_t R_SHORT_LIQ_SPIKE = 63;
inline constexpr uint16_t R_STOCH_BULLISH = 64;
inline constexpr uint16_t R_STRONG_CANDLE = 65;
inline constexpr uint16_t R_STRONG_TREND = 66;
inline constexpr uint16_t R_TIGHT_SPREAD = 67;
inline constexpr uint16_t R_TRADE_IMBALANCE = 68;
inline constexpr uint16_t R_VOL_BREAKOUT = 69;
inline constexpr uint16_t R_VOL_COMPRESSION = 70;
inline constexpr uint16_t R_VOL_EXPANSION = 71;
inline constexpr uint16_t R_WIDE_SPREAD = 72;

} // namespace codes
} // namespace agamotto
