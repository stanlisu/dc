"""Shared module-level helpers for the agamotto research modules.

Extracted from research.py so research_features.py / research_filters.py can use
them without importing research.py (which imports THEM — that would cycle).
research.py re-exports every name here, so `agamotto.research.<name>` keeps
resolving for external callers and capability probes.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from ._obf.codec import default
    return default()


# Use relative import for internal module
try:
    from .lib_binance import fetch_futures_klines, klines_to_dataframe
except ImportError:
    # Fallback to assume it's in the path or installed
    try:
        from lib_binance import fetch_futures_klines, klines_to_dataframe
    except ImportError:
        pass


# ── Route B: direct-to-S3 filter writes ──────────────────────────────────────
# Mirrors mjolnir.core.streaming's Route B (same contract, same region, same
# fail-loud rules) so kline research can write its bulk filter parquets straight
# to s3:// instead of needing a local/NAS staging tree the size of the whole
# filter set (orb 15m = 195 GB, which does not fit on shield2's local disk).
#
# Before this existed, an s3:// OUTPUT_DIR did NOT raise — os.path.isabs()
# returns False for "s3://...", so create() joined the URI onto home_root and
# os.makedirs() produced a local directory literally named "s3:". Probed
# 2026-08-03 on shield2: 87 MB of parquets landed in marvel/s3:/tardis-stan-data/
# and ZERO objects reached S3, exit 0.
_S3_REGION = "ap-northeast-1"


def _is_s3(path) -> bool:
    """True if ``path`` is an ``s3://`` URI (Route B direct-to-S3 target)."""
    return str(path).startswith("s3://")


def _s3_key(uri) -> str:
    """``s3://bucket/prefix`` -> ``bucket/prefix``.

    pyarrow's ``S3FileSystem`` addresses objects by a bucket-qualified key, not
    a bare key — same convention as mjolnir.core.streaming._s3_key.
    """
    return str(uri)[len("s3://"):].rstrip("/")


def _open_s3fs():
    """Open an ``S3FileSystem`` on the DEFAULT AWS credential chain.

    Fails LOUDLY on any init/credential/region error: per CLAUDE.md there is NO
    fallback to the local filesystem for an s3:// request. A broken S3 config
    must abort the run, not silently write local (which is precisely the bug
    this module previously had).
    """
    import pyarrow.fs as pafs
    try:
        return pafs.S3FileSystem(region=_S3_REGION)
    except Exception as exc:
        raise RuntimeError(
            f"failed to init S3FileSystem(region={_S3_REGION!r}) for s3:// filter "
            f"writes via the default AWS credential chain: {exc!r}. Refusing to "
            "fall back to local (CLAUDE.md: no silent fallbacks)."
        ) from exc


