#!/usr/bin/env python3
"""Classify a host's `ps` listing into live TRADING / OPS / RESEARCH processes.

THE ONE definition of "is this host busy?" for every deploy path. Both callers
source scripts/bot_guard.sh, which drives this:

  * dc/scripts/deploy_host.sh            (manual deploys and the Deploy CI job)
  * marvel/scripts/sync_all_repos.sh     (/sync-all-repo's per-host dc gate)

WHY IT EXISTS. Both gates used to grep the remote `ps` for three PYTHON names,
``run_knull|trade_execution|mjolnir_bridge``. Measured on hydra 2026-08-28,
while that gate reported the host CLEAR::

    /opt/bin/tsLtpShmOms             pid 2909173  17:53:49  <- venue credentials
    /opt/bin/tsBinanceFeedPublisher  pid  598913  13:23:49
    /opt/bin/tsBinanceFeedPublisher  pid  598915  13:23:49
    python3 gauntlet/launch_sentinel_bots.py --order-path-dry-run false  pid 1286381
    /bin/bash ./refresh_fleet_klines.sh                                  pid 1286564

and on shield, ``rolling_predict_returns.py --workers 20`` at load 6.48. A
deploy runs ``rsync --delete`` plus ``pip install -e`` over exactly the tree
those processes have mapped.

HOW IT MATCHES, and why not a substring. `ps | grep <pattern>` counts the
pipeline that asked the question. `tasks/lessons.md` 2026-08-05 and 2026-08-24
record five wait-loops lost to that in one session, including one that spun six
days on hydra: bracketing ONE occurrence of the pattern is not enough, because a
second unbracketed copy later on the same command line re-poisons the guard. So
nothing here reads a raw substring. Following gauntlet/start_oms.py on marvel
`main`, a process is identified by resolving **argv[0]**:

  * a binary        -> argv[0] itself (basename, plus its directory);
  * an interpreter  -> the SCRIPT argv[0] is running, by basename stem;
  * a shell wrapper -> `bash -c '<anything>'` is NOT read. A wrapper's `-c`
    string is text about a process, never a process. When bash -c runs a
    single simple command it execs it, so the real thing always has its own
    ps row (hydra: 1286380 is the wrapper, 1286381 is the launcher) and
    nothing is lost by refusing to parse the string.

WHY THREE CLASSES, not one boolean. A live trading process is a hard STOP: it
holds venue credentials and a mapped import tree. A research job is also a
reason not to deploy -- a `pip install -e` mid-run corrupts it -- but it is the
operator's call whether to wait for it or kill it, and they cannot make that
call if the gate only says "busy".

NEVER READ A SHORT LISTING AS "CLEAR". An ssh that dies mid-stream, a host that
refuses the connection, an OOM'd `ps`: each yields a short or garbled listing,
and the old `| wc -l` gate turned every one of them into the number 0, i.e.
"idle, deploy away". Anything that is not a parseable listing of a plausible
number of processes raises UnusableListing here, and both callers skip the host.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

# Exit codes. The shell callers branch on these, so they are part of the API.
RC_CLEAR = 0
RC_BLOCKED = 1      # TRADING or OPS -- refuse
RC_UNUSABLE = 2     # listing unusable -- refuse, host state UNKNOWN
RC_RESEARCH = 3     # research only -- refuse by default, operator may wait

CLASS_TRADING = "TRADING"
CLASS_OPS = "OPS"
CLASS_RESEARCH = "RESEARCH"

# A real host runs dozens of processes. Fewer than this is a broken capture,
# not an idle machine. Kernel threads alone clear it on every Linux box.
MIN_ROWS = 5

# --- what counts -----------------------------------------------------------
# C++ sentinel fleet, by argv[0] BASENAME. The two directory rules below cover
# binaries not named here, so this list never has to be exhaustive.
SENTINEL_BINARIES = frozenset({
    "tsLtpBaseAlgo", "tsLtpShmOms", "tsBinanceFeedPublisher",
})
# Any executable shipped by the sentinel release process is fleet, so a binary
# nobody listed above is still caught. The release tree nests a build-type
# directory under bin/ -- measured on hydra 2026-08-28:
#   /opt/releases/ltp_release_20260804/bin/Release/tsLtpBaseAlgo
# so this must NOT be anchored one level below bin/.
SENTINEL_DIR_RULES = (
    re.compile(r"^/opt/bin/ts[A-Za-z0-9_]*$"),
    re.compile(r"^/opt/releases/[^/]+/bin/.+$"),
)

# Python/shell entry points, by basename STEM (extension dropped).
# Bot entry points are the authoritative list from marvel's close-all skill
# (.claude/skills/close-all/SKILL.md:42-50) and tesseract/watchdog.py:401.
TRADING_SCRIPTS = frozenset({
    "run_knull", "trade_execution", "mjolnir_bridge", "orb_bridge",
    "agamotto_bridge", "vibranium_bridge", "stormbreaker_bridge",
    "naive_bridge", "run_vibranium", "sumo_executor", "run_mjolnir_bridge",
})
# Every knull bridge is `<algo>_bridge.py`; new algos must not need a code
# change here to be seen.
TRADING_STEM_SUFFIXES = ("_bridge",)

OPS_SCRIPTS = frozenset({
    "launch_sentinel_bots", "launch_bots", "launch_xmen_bots",
    "refresh_fleet_klines", "watchdog", "close_all", "close_all_positions",
    "start_oms", "stop_oms",
})

# Heavy jobs a `pip install -e` mid-run corrupts.
RESEARCH_SCRIPTS = frozenset({
    "rolling_predict_returns", "run_research", "run_research_sharded",
    "run_agamotto_research", "run_orb_research", "run_aether_research",
    "run_scepter_research", "run_kline_research_sharded",
    "build_bars", "mm_ladder_sim", "step9_sim", "generate_daily_pnl",
    "run_pipeline", "run_mjolnir_pipeline", "explore_walkback",
})

_INTERPRETERS = re.compile(r"^(python|python\d(\.\d+)?|pypy\d?)$")
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh"})
# argv[0]s that merely prefix the real command; step over them.
_PREFIX_COMMANDS = frozenset({"env", "nohup", "setsid", "stdbuf", "nice", "ionice", "time"})
# Options that take a value, so the token after them is NOT the script.
_PY_OPTS_WITH_ARG = frozenset({"-W", "-X", "--check-hash-based-pycs"})

_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(.*\S)\s*$")
_PS_HEADER = re.compile(r"^\s*PID\b")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


class UnusableListing(RuntimeError):
    """The ps listing could not be trusted -- treat the host as UNKNOWN, never clear."""


@dataclass(frozen=True)
class Finding:
    pid: int
    etime: str
    args: str
    klass: str
    target: str   # the resolved binary/script name the classification keyed on

    def render(self) -> str:
        return f"{self.klass:<8} pid={self.pid:<9} elapsed={self.etime:<12} {self.args}"


def _shell_target(rest: list[str]) -> str | None:
    """The script a shell is running, or None for `bash -c '<string>'`."""
    while rest and rest[0].startswith("-") and rest[0] != "--":
        # -c, and every combined form of it (-lc, -ec, ...), introduces an
        # inline STRING. Text about a process is not a process; refuse it.
        if "c" in rest[0].lstrip("-"):
            return None
        rest = rest[1:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    return rest[0] if rest else None


def _interpreter_target(rest: list[str]) -> str | None:
    """The script or module an interpreter is running, or None for inline code."""
    while rest:
        tok = rest[0]
        if tok in ("-c", "-"):
            return None                      # inline code / stdin, e.g. loky workers
        if tok == "-m":
            # `python -m pkg.mod` -- the module's last segment is the name.
            return rest[1].rsplit(".", 1)[-1] if len(rest) > 1 else None
        if tok == "--":
            rest = rest[1:]
            break
        if tok.startswith("-"):
            rest = rest[2:] if tok in _PY_OPTS_WITH_ARG else rest[1:]
            continue
        break
    return rest[0] if rest else None


def _resolve_target(args: str) -> str | None:
    """Return the binary path or script path this row is really running.

    None means "this row is not a process we can attribute" -- a shell wrapper
    whose -c string we refuse to read, an interpreter running inline code, or a
    command with no script operand.
    """
    toks = args.split()
    while toks and (_ENV_ASSIGN.match(toks[0]) or PurePosixPath(toks[0]).name in _PREFIX_COMMANDS):
        toks = toks[1:]
    if not toks:
        return None

    argv0 = toks[0]
    base = PurePosixPath(argv0).name
    # `ps` renders a login shell as "-bash"; strip the leading dash.
    base = base[1:] if base.startswith("-") and len(base) > 1 else base

    if base in _SHELLS:
        return _shell_target(toks[1:])
    if _INTERPRETERS.match(base):
        return _interpreter_target(toks[1:])
    return argv0


def _stem(path: str) -> str:
    name = PurePosixPath(path).name
    return name[:-3] if name.endswith(".py") else (name[:-3] if name.endswith(".sh") else name)


def _classify(target: str) -> str | None:
    name = PurePosixPath(target).name
    if name in SENTINEL_BINARIES:
        return CLASS_TRADING
    if any(rule.match(target) for rule in SENTINEL_DIR_RULES):
        return CLASS_TRADING

    stem = _stem(target)
    if stem in TRADING_SCRIPTS or stem.endswith(TRADING_STEM_SUFFIXES):
        return CLASS_TRADING
    if stem in OPS_SCRIPTS:
        return CLASS_OPS
    if stem in RESEARCH_SCRIPTS:
        return CLASS_RESEARCH
    return None


def scan(listing: str) -> list[Finding]:
    """Classify a `ps -eo pid=,etime=,args=` listing. Raises on an unusable one."""
    rows = 0
    found: list[Finding] = []
    for line in listing.splitlines():
        if not line.strip() or _PS_HEADER.match(line):
            continue
        m = _ROW.match(line)
        if not m:
            # Never skip quietly: an ssh error line landing in the listing is
            # exactly how "unreachable" gets misread as "idle".
            raise UnusableListing(f"unparseable ps row (host state UNKNOWN): {line!r}")
        rows += 1
        pid, etime, args = int(m.group(1)), m.group(2), m.group(3)
        target = _resolve_target(args)
        if target is None:
            continue
        klass = _classify(target)
        if klass is not None:
            found.append(Finding(pid=pid, etime=etime, args=args, klass=klass,
                                 target=_stem(target)))
    if rows < MIN_ROWS:
        raise UnusableListing(
            f"only {rows} process row(s) -- a live host always shows dozens. "
            f"The capture is truncated or the ssh failed; host state UNKNOWN, "
            f"NOT clear.")
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ps-file", help="classify this captured listing instead of stdin")
    args = ap.parse_args(argv)

    text = open(args.ps_file).read() if args.ps_file else sys.stdin.read()
    try:
        found = scan(text)
    except UnusableListing as exc:
        print(f"UNUSABLE listing: {exc}", file=sys.stderr)
        return RC_UNUSABLE

    for f in found:
        print(f.render())
    if any(f.klass in (CLASS_TRADING, CLASS_OPS) for f in found):
        return RC_BLOCKED
    if found:
        return RC_RESEARCH
    return RC_CLEAR


if __name__ == "__main__":
    sys.exit(main())
