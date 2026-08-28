"""Regression tests for the pre-deploy live-process gate.

THE DEFECT THIS GUARDS AGAINST (measured on hydra 2026-08-28, while the old
gate reported the host CLEAR):

    tsLtpShmOms             pid 2909173  17:53:49  <- holds venue credentials
    tsBinanceFeedPublisher  pid  598913  13:23:49
    tsBinanceFeedPublisher  pid  598915  13:23:49
    launch_sentinel_bots.py pid 1286381   2-20:01  <- --order-path-dry-run false
    refresh_fleet_klines.sh pid 1286564   2-20:01

The old pattern was ``run_knull|trade_execution|mjolnir_bridge`` — three PYTHON
names — so it saw none of the C++ sentinel fleet and none of the launchers, and
would have let `rsync --delete` + `pip install -e` run over a tree a live
credentialed trading process had mapped.

Every "detected" test below is paired with a mutant in
``test_three_name_mutant_*``: revert the matcher to the old three names and the
test must go red. That pairing is the point — a detection test that also passes
against the buggy matcher proves nothing.

The second axis is SELF-MATCH. `ps | grep <pattern>` counts the pipeline that
asked the question (tasks/lessons.md 2026-08-05 / 2026-08-24: bracketing ONE
occurrence is not enough, a later unbracketed copy re-poisons the guard). The
matcher here never looks at a raw substring: it resolves argv[0], and for a
shell it REFUSES to read the `-c` string at all. `test_self_match_*` feeds it
listings that would fool a substring matcher.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"

sys.path.insert(0, str(_SCRIPTS))
import bot_guard  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures. Enough real kernel/system rows that `--min-rows` is satisfied, so a
# test never accidentally exercises the truncated-listing path.
# ---------------------------------------------------------------------------
FILLER = """\
      1  6-11:26:11 /sbin/init
    282  6-11:26:26 /sbin/multipathd -d -s
    538  6-11:26:22 /usr/sbin/acpid
    547  6-11:26:22 /usr/sbin/irqbalance --foreground
    548  6-11:26:22 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers
    610  6-11:26:22 s3fs tardis-stan-data /mnt/tardis-stan-data -o rw,allow_other
   1883  6-11:18:07 (sd-pam)
1410712  2-19:09:03 /usr/sbin/zabbix_agentd --foreground
3636654       00:25 sleep 60
3641179       00:00 ps -eo pid=,etime=,args=
"""


def listing(*rows: str) -> str:
    return FILLER + "".join(r if r.endswith("\n") else r + "\n" for r in rows)


def classes(text: str) -> list[str]:
    return [f.klass for f in bot_guard.scan(text)]


def argvs(text: str) -> list[str]:
    return [f.args for f in bot_guard.scan(text)]


# The exact matcher the live gate used before this change, as a mutant. Any
# test that passes under BOTH this and the real matcher is not testing the fix.
THREE_NAME_MUTANT = ("run_knull", "trade_execution", "mjolnir_bridge")


def three_name_gate(text: str) -> list[str]:
    """Reproduces `ps -eo args= | grep -E 'run_knull|trade_execution|mjolnir_bridge'`."""
    return [ln for ln in text.splitlines() if any(p in ln for p in THREE_NAME_MUTANT)]


# ---------------------------------------------------------------------------
# 1. C++ sentinel fleet — the exact live defect
# ---------------------------------------------------------------------------
BASE_ALGO = "1286900    04:11:02 /opt/bin/tsLtpBaseAlgo -f /opt/infra_configs/agamotto_BTCUSDT.json"
SHM_OMS = "2909173    17:53:49 /opt/bin/tsLtpShmOms -f /opt/infra_configs/oms_ltp_config.json"
FEED_P1 = " 598913    13:23:49 /opt/bin/tsBinanceFeedPublisher -f /opt/infra_configs/feed_publisher_p1.json"
FEED_P2 = " 598915    13:23:49 /opt/bin/tsBinanceFeedPublisher -f /opt/infra_configs/feed_publisher_p2.json"
RELEASE_BIN = "  77120    01:02:03 /opt/releases/2026-08-20/bin/tsLtpBaseAlgo -f /opt/infra_configs/x.json"


def test_ts_ltp_base_algo_detected():
    assert classes(listing(BASE_ALGO)) == ["TRADING"]


def test_three_name_mutant_misses_ts_ltp_base_algo():
    """THE LIVE DEFECT. The old gate reports this host clear."""
    assert three_name_gate(listing(BASE_ALGO)) == []


def test_shm_oms_and_feed_publishers_detected():
    found = bot_guard.scan(listing(SHM_OMS, FEED_P1, FEED_P2))
    assert [f.klass for f in found] == ["TRADING"] * 3
    assert [f.pid for f in found] == [2909173, 598913, 598915]


def test_three_name_mutant_misses_oms_and_feed():
    assert three_name_gate(listing(SHM_OMS, FEED_P1, FEED_P2)) == []


def test_opt_releases_binary_detected():
    """Sentinel also runs out of /opt/releases/<tag>/bin, not only /opt/bin."""
    assert classes(listing(RELEASE_BIN)) == ["TRADING"]


def test_real_release_path_with_a_build_type_directory_detected():
    """Verbatim from hydra 2026-08-28: bin/ nests a Release/ level under it."""
    row = ("3676214       04:34 /opt/releases/ltp_release_20260804/bin/Release/tsLtpBaseAlgo "
           "-f config/sp_config.agamotto_shard0.json")
    assert classes(listing(row)) == ["TRADING"]


def test_unlisted_binary_under_a_release_tree_detected():
    """The directory rule is the safety net for a binary this file never names."""
    row = "3676999       04:34 /opt/releases/ltp_release_20260804/bin/Release/someNewExecutor -f x.json"
    assert classes(listing(row)) == ["TRADING"]


def test_unknown_opt_bin_ts_binary_detected():
    """Any /opt/bin/ts* is the fleet, including one this list has never seen."""
    row = "  99001    00:10:00 /opt/bin/tsSomeFutureAlgo --config /opt/infra_configs/z.json"
    assert classes(listing(row)) == ["TRADING"]


# ---------------------------------------------------------------------------
# 2. Python bots — the three the old gate did catch must STILL be caught
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row", [
    "  48710    05:00:00 /home/stan/miniconda3/envs/py313/bin/python "
    "/home/stan/sandbox/marvel/knull/run_knull.py -c pred_agamotto.base.15m_1/setting.json --venue ltp",
    "  48711    05:00:00 python3 /home/stan/sandbox/marvel/ltp/trade_execution.py -c ltp/setting.json",
    "  48712    05:00:00 python3 knull/mjolnir_bridge.py -c mjolnir.base.30m_1/setting.json --venue ltp",
])
def test_original_three_python_bots_still_detected(row):
    assert classes(listing(row)) == ["TRADING"]


@pytest.mark.parametrize("row", [
    "  48713    05:00:00 python3 knull/orb_bridge.py -c pred_orb.base.15m_1/setting.json --venue ltp",
    "  48714    05:00:00 python3 knull/agamotto_bridge.py -c x/setting.json --venue sumo",
    "  48715    05:00:00 python3 knull/stormbreaker_bridge.py -c x/setting.json --venue ltp",
    "  48716    05:00:00 python3 sumo/sumo_executor.py -c x/setting.json",
    "  48717    05:00:00 python3 vibranium/run_vibranium.py -c x/setting.json",
])
def test_sibling_bots_the_old_gate_missed_are_detected(row):
    """mjolnir_bridge was hardcoded; every sibling bridge/executor was invisible."""
    assert classes(listing(row)) == ["TRADING"]


def test_three_name_mutant_misses_sibling_bridges():
    row = "  48713    05:00:00 python3 knull/orb_bridge.py -c pred_orb.base.15m_1/setting.json --venue ltp"
    assert three_name_gate(listing(row)) == []


# ---------------------------------------------------------------------------
# 3. Launchers / ops in flight
# ---------------------------------------------------------------------------
LAUNCH_SENTINEL = ("1286381  2-20:01:38 python3 gauntlet/launch_sentinel_bots.py --algo agamotto.base.15m_1 "
                   "--account LTP --shards 2 --order-path-symbols BTCUSDT --order-path-dry-run false")
REFRESH_FLEET = "1286564  2-20:01:20 /bin/bash ./refresh_fleet_klines.sh"
WATCHDOG = "  90210    00:30:00 /home/stan/miniconda3/envs/py313/bin/python tesseract/watchdog.py"


def test_launch_sentinel_bots_detected():
    assert classes(listing(LAUNCH_SENTINEL)) == ["OPS"]


def test_three_name_mutant_misses_launch_sentinel_bots():
    assert three_name_gate(listing(LAUNCH_SENTINEL)) == []


def test_refresh_fleet_klines_shell_script_detected():
    """argv[0] is /bin/bash and the SCRIPT is argv[1] — not a -c string."""
    assert classes(listing(REFRESH_FLEET)) == ["OPS"]


def test_watchdog_detected():
    assert classes(listing(WATCHDOG)) == ["OPS"]


# ---------------------------------------------------------------------------
# 4. Research jobs — detected, and NOT collapsed into "bot"
# ---------------------------------------------------------------------------
ROLLING_PREDICT = ("124216    01:30:27 /opt/miniconda3/envs/py313/bin/python gauntlet/rolling_predict_returns.py "
                   "-c gauntlet/pred_orb.base.15m_1 --window-size 4 --workers 20 --no-weights")


def test_rolling_predict_is_research_not_trading():
    found = bot_guard.scan(listing(ROLLING_PREDICT))
    assert [f.klass for f in found] == ["RESEARCH"]
    assert found[0].klass != "TRADING"


def test_three_name_mutant_misses_rolling_predict():
    assert three_name_gate(listing(ROLLING_PREDICT)) == []


@pytest.mark.parametrize("row", [
    "  10001    01:00:00 python3 mjolnir/gauntlet/run_research.py -c x --window 1",
    "  10002    01:00:00 python3 mjolnir/gauntlet/build_bars.py --tf 5s",
    "  10003    01:00:00 python3 gauntlet/mm_ladder_sim.py -c x --date 2026-08-20",
    "  10004    01:00:00 python3 psylocke/step9_sim.py -c x",
    "  10005    01:00:00 python3 gauntlet/generate_daily_pnl.py -c x",
    "  10006    01:00:00 python3 gauntlet/run_agamotto_research.py -c x",
])
def test_research_jobs_detected(row):
    assert classes(listing(row)) == ["RESEARCH"]


def test_classes_are_reported_separately():
    """The gate must not be one boolean: an operator may wait on RESEARCH."""
    text = listing(SHM_OMS, LAUNCH_SENTINEL, ROLLING_PREDICT)
    assert classes(text) == ["TRADING", "OPS", "RESEARCH"]


# ---------------------------------------------------------------------------
# 5. SELF-MATCH IMMUNITY — the mechanism that produced the original bug
# ---------------------------------------------------------------------------
# Real rows captured from hydra/shield/shield2 on 2026-08-28. Every one of them
# would be a false positive for a substring matcher; none is a real process.
WRAPPER_LAUNCH = ("1286380  2-20:01:38 bash -c cd /home/stan/sandbox/marvel && "
                  "PYTHONPATH=agamotto_pkg/src:. python3 gauntlet/launch_sentinel_bots.py "
                  "--algo agamotto.base.15m_1 --account LTP 2>&1 | grep -E 'ORDER PATH'; date -u")
WRAPPER_GUARD = ("2577971  4-07:05:59 bash -c until ! pgrep -f \"[r]un_knull\"; do sleep 10; done; "
                 "echo run_knull done")
WRAPPER_REFRESH = ("1286562  2-20:01:20 bash -lc cd /home/stan/agamotto_test && setsid nohup "
                   "/bin/bash ./refresh_fleet_klines.sh > fleet_refresher.log 2>&1 < /dev/null &")
THE_GREP_ITSELF = "1286382  2-20:01:38 grep -E run_knull|trade_execution|mjolnir_bridge"
TAIL_OF_A_LOG = ("2375672 10-11:16:34 tail -f -n 0 "
                 "/home/stan/sandbox/marvel/gauntlet/run_pipeline_orb_base_15m_liquid.log")
PYTHON_DASH_C = ("126236    01:23:29 /opt/miniconda3/envs/py313/bin/python -c "
                 "from joblib.externals.loky.backend.resource_tracker import main; main(4, False)")
LOKY_WORKER = ("127323    01:19:17 /opt/miniconda3/envs/py313/bin/python -m "
               "joblib.externals.loky.backend.popen_loky_posix --process-name LokyProcess-35 --pipe 33")


@pytest.mark.parametrize("row,why", [
    (WRAPPER_LAUNCH, "bash -c wrapper whose -c string names a real bot"),
    (WRAPPER_GUARD, "a wait-loop carrying the pattern twice, once unbracketed"),
    (WRAPPER_REFRESH, "bash -lc wrapper naming refresh_fleet_klines.sh"),
    (THE_GREP_ITSELF, "the checking pipeline's own grep"),
    (TAIL_OF_A_LOG, "a tail of a LOG FILE named after a pipeline"),
    (PYTHON_DASH_C, "python -c inline code"),
    (LOKY_WORKER, "a joblib worker, not the job"),
])
def test_self_match_immunity(row, why):
    assert bot_guard.scan(listing(row)) == [], why


def test_self_match_mutant_a_substring_matcher_is_fooled():
    """Proves the fixtures above are real traps, not vacuous."""
    fooled = three_name_gate(listing(WRAPPER_GUARD, THE_GREP_ITSELF))
    assert len(fooled) == 2


def test_wrapper_is_skipped_but_its_real_child_is_found():
    """hydra 2026-08-28: 1286380 is the wrapper, 1286381 is the process."""
    found = bot_guard.scan(listing(WRAPPER_LAUNCH, LAUNCH_SENTINEL))
    assert [f.pid for f in found] == [1286381]


# ---------------------------------------------------------------------------
# 6. Clean host, and the never-deploy-blind guard
# ---------------------------------------------------------------------------
def test_clean_host_is_clear():
    assert bot_guard.scan(FILLER) == []


def test_ps_header_row_is_tolerated():
    assert bot_guard.scan("    PID     ELAPSED \n" + FILLER) == []


def test_truncated_listing_raises_rather_than_reading_as_clear():
    """An ssh that died mid-stream must never be mistaken for an idle host."""
    with pytest.raises(bot_guard.UnusableListing):
        bot_guard.scan("      1  6-11:26:11 /sbin/init\n")


def test_empty_listing_raises():
    with pytest.raises(bot_guard.UnusableListing):
        bot_guard.scan("")


def test_garbage_row_raises_rather_than_being_skipped():
    with pytest.raises(bot_guard.UnusableListing):
        bot_guard.scan(FILLER + "ssh: connect to host hydra port 22: Connection refused\n")


# ---------------------------------------------------------------------------
# 7. CLI contract — the exit codes both shell callers branch on
# ---------------------------------------------------------------------------
def run_cli(text: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_SCRIPTS / "bot_guard.py"), *args],
                          input=text, capture_output=True, text=True)


def test_cli_clear_exits_0():
    assert run_cli(FILLER).returncode == bot_guard.RC_CLEAR


def test_cli_trading_exits_1_and_prints_pid_elapsed_argv():
    r = run_cli(listing(SHM_OMS))
    assert r.returncode == bot_guard.RC_BLOCKED
    assert "2909173" in r.stdout and "17:53:49" in r.stdout and "tsLtpShmOms" in r.stdout


def test_cli_research_only_exits_3_so_the_operator_can_choose_to_wait():
    r = run_cli(listing(ROLLING_PREDICT))
    assert r.returncode == bot_guard.RC_RESEARCH
    assert "RESEARCH" in r.stdout


def test_cli_trading_wins_over_research():
    assert run_cli(listing(SHM_OMS, ROLLING_PREDICT)).returncode == bot_guard.RC_BLOCKED


def test_cli_unusable_listing_exits_2():
    r = run_cli("")
    assert r.returncode == bot_guard.RC_UNUSABLE
    assert "listing" in (r.stdout + r.stderr).lower()


# ---------------------------------------------------------------------------
# 8. The shared shell library both deploy paths source
# ---------------------------------------------------------------------------
GUARD_SH = _SCRIPTS / "bot_guard.sh"


def run_sh(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'set -eo pipefail; . "{GUARD_SH}"; {snippet}'],
                          capture_output=True, text=True)


def test_shell_library_exists_and_parses():
    assert GUARD_SH.is_file()
    assert subprocess.run(["bash", "-n", str(GUARD_SH)]).returncode == 0


def test_shell_ps_command_carries_no_pattern():
    """The REMOTE command must be pattern-free, so it cannot self-match at all."""
    r = run_sh('printf "%s\\n" "$BOT_GUARD_PS_CMD"')
    assert r.returncode == 0
    cmd = r.stdout.strip()
    assert cmd.startswith("ps ")
    for token in ("grep", "run_knull", "trade_execution", "ts", "wc -l"):
        assert token not in cmd.replace("etime", "").replace("args", ""), cmd


def test_shell_classify_reports_and_returns_blocked(tmp_path):
    f = tmp_path / "ps.txt"
    f.write_text(listing(SHM_OMS, BASE_ALGO))
    r = run_sh(f'bot_guard_classify < "{f}" || echo "rc=$?"')
    assert "tsLtpShmOms" in r.stdout and "tsLtpBaseAlgo" in r.stdout
    assert "rc=1" in r.stdout


def test_shell_classify_clear_returns_0(tmp_path):
    f = tmp_path / "ps.txt"
    f.write_text(FILLER)
    r = run_sh(f'bot_guard_classify < "{f}"; echo "rc=$?"')
    assert "rc=0" in r.stdout


def test_shell_classify_research_returns_3(tmp_path):
    f = tmp_path / "ps.txt"
    f.write_text(listing(ROLLING_PREDICT))
    r = run_sh(f'bot_guard_classify < "{f}" || echo "rc=$?"')
    assert "rc=3" in r.stdout


def test_shell_missing_classifier_is_distinct_from_an_unusable_listing(tmp_path):
    """python's own "can't open file" exit code is 2 -- the same as RC_UNUSABLE.

    Found on the first live run against hydra: BOT_GUARD_DIR resolved to the
    caller's cwd and every host came back "ps listing UNUSABLE", which sends the
    operator to look at the HOST for a problem that is in this checkout.
    """
    f = tmp_path / "ps.txt"
    f.write_text(FILLER)
    r = run_sh(f'BOT_GUARD_IMPL={tmp_path}/nope.py bot_guard_classify < "{f}" || echo "rc=$?"')
    assert "rc=4" in r.stdout
    assert "classifier missing" in r.stderr


def test_shell_library_resolves_its_classifier_from_its_own_location(tmp_path):
    """Sourced from any cwd, with or without BASH_SOURCE -- never the caller's."""
    r = subprocess.run(["bash", "-c", f'cd {tmp_path}; . "{GUARD_SH}"; echo "$BOT_GUARD_IMPL"'],
                       capture_output=True, text=True)
    assert r.stdout.strip() == str(_SCRIPTS / "bot_guard.py")


def test_shell_classify_empty_returns_2(tmp_path):
    f = tmp_path / "ps.txt"
    f.write_text("")
    r = run_sh(f'bot_guard_classify < "{f}" || echo "rc=$?"')
    assert "rc=2" in r.stdout


# ---------------------------------------------------------------------------
# 9. deploy_host.sh must actually USE the shared library
# ---------------------------------------------------------------------------
DEPLOY_HOST = _SCRIPTS / "deploy_host.sh"


def test_deploy_host_sources_the_shared_guard():
    body = DEPLOY_HOST.read_text()
    assert "bot_guard.sh" in body
    assert "bot_guard_report" in body
    assert "$BOT_GUARD_PS_CMD" in body


def test_deploy_host_no_longer_carries_its_own_three_name_pattern():
    body = DEPLOY_HOST.read_text()
    assert "[r]un_knull" not in body
    assert "run_knull|trade_execution" not in body


# ---------------------------------------------------------------------------
# 10. End-to-end against the REAL listing captured from hydra
# ---------------------------------------------------------------------------
HYDRA_REAL = _REPO / "tests" / "fixtures" / "ps_hydra_2026_08_28.txt"


def test_real_hydra_listing_finds_what_the_old_gate_missed():
    found = bot_guard.scan(HYDRA_REAL.read_text())
    names = sorted({f.target for f in found})
    assert names == ["launch_sentinel_bots", "refresh_fleet_klines",
                     "tsBinanceFeedPublisher", "tsLtpShmOms"]
    assert sorted({f.klass for f in found}) == ["OPS", "TRADING"]
    assert len(found) == 5  # 2x feed publisher + oms + launcher + refresher


def test_real_hydra_listing_reads_as_CLEAR_under_the_old_gate():
    """The whole reason this change exists."""
    assert three_name_gate(HYDRA_REAL.read_text()) == []
