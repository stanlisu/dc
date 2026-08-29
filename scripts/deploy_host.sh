#!/bin/bash
# deploy_host.sh — deliver the built packages to ONE host, and prove it took.
#
# The single implementation shared by build_distribution.sh (manual, from the
# Mac) and .github/workflows/deploy.yml (CI). Before 2026-08-09 those were two
# different code paths — CI rsynced AND pip-installed, the manual script only
# rsynced — which is half of why the hosts diverged.
#
#   bash scripts/deploy_host.sh --host hydra --python '$HOME/miniconda3/envs/py313/bin/python'
#   bash scripts/deploy_host.sh --host stan@1.2.3.4 --python /opt/miniconda3/envs/py313/bin/python \
#        --algos mjolnir,orb --ssh-opts '-o ConnectTimeout=15'
#
# Four steps, any of which fails the deploy LOUDLY:
#   1. pre-flight  — refuse a host with live bots (rsync --delete tears a
#                    running import tree; a pip install swaps pyarmor_runtime.so
#                    out from under an already-loaded one).
#   2. rsync       — build_dist/<algo>_pkg/ -> <root>/<algo>_pkg/
#   3. install     — pip install -e on EVERY path. `pip install -e A B C` binds
#                    -e to A ONLY; B and C become frozen site-packages copies
#                    that no future rsync reaches. That one missing flag made
#                    every deploy to shield a silent no-op from 2026-08-06.
#   4. verify      — scripts/verify_deploy.py on the host: the module python
#                    imports must BE the tree we just rsynced, byte for byte.
#                    Not "does it import" — a stale package imports perfectly,
#                    which is exactly why this went unnoticed for three days.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$DC_ROOT/build_dist}"
REMOTE_BASE="${REMOTE_BASE:-/home/stan/sandbox/marvel}"
ALL_ALGOS="agamotto,orb,aether,scepter,mjolnir,stormbreaker,vibranium,valkyrie,vomir"

HOST=""; PY=""; ALGOS_CSV="$ALL_ALGOS"; SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=30"
ALLOW_RUNNING_BOTS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --host)     HOST="$2"; shift 2 ;;
        --python)   PY="$2"; shift 2 ;;
        --algos)    ALGOS_CSV="$2"; shift 2 ;;
        --ssh-opts) SSH_OPTS="$SSH_OPTS $2"; shift 2 ;;
        --remote-base) REMOTE_BASE="$2"; shift 2 ;;
        # Deliberately ugly and undocumented in the README: there is no good
        # reason to deploy under a live bot. Exists only so an operator who has
        # genuinely decided to accept a torn import tree can, on purpose.
        --i-accept-a-torn-import-tree) ALLOW_RUNNING_BOTS=1; shift ;;
        *) echo "deploy_host.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$HOST" ] || { echo "deploy_host.sh: --host is required" >&2; exit 2; }
[ -n "$PY" ]   || { echo "deploy_host.sh: --python is required (shield is /opt/miniconda3, hydra+shield2 are \$HOME/miniconda3)" >&2; exit 2; }

IFS=',' read -r -a ALGOS <<< "$ALGOS_CSV"
VERIFIER="$SCRIPT_DIR/verify_deploy.py"
[ -f "$VERIFIER" ] || { echo "deploy_host.sh: $VERIFIER missing — a deploy that cannot be verified is not a deploy" >&2; exit 1; }

# THE shared live-process gate, also sourced by marvel/scripts/sync_all_repos.sh.
# One definition, so the two cannot drift apart again.
GUARD="$SCRIPT_DIR/bot_guard.sh"
[ -f "$GUARD" ] || { echo "deploy_host.sh: $GUARD missing — a deploy that cannot check the host is not a deploy" >&2; exit 1; }
# shellcheck source=scripts/bot_guard.sh
. "$GUARD"

# shellcheck disable=SC2086  # SSH_OPTS is a deliberate word-split option list
ssh_run() { ssh $SSH_OPTS "$HOST" "$@"; }

echo "=== deploy_host $HOST ==="
echo "    python:      $PY"
echo "    remote base: $REMOTE_BASE"
echo "    algos:       ${ALGOS[*]}"

# ---- 1. pre-flight ---------------------------------------------------------
echo "--- pre-flight: live processes on $HOST?"
# The remote command carries NO pattern — a plain `ps`, classified locally by
# bot_guard.py. The previous version grepped on the far side for three python
# names; measured on hydra 2026-08-28 that reported CLEAR while tsLtpShmOms
# (venue credentials), two tsBinanceFeedPublishers, launch_sentinel_bots.py and
# refresh_fleet_klines.sh were all running. See scripts/bot_guard.py.
LISTING="$(ssh_run "$BOT_GUARD_PS_CMD" 2>/dev/null || true)"
GUARD_RC=0
printf '%s\n' "$LISTING" | bot_guard_report "$HOST" || GUARD_RC=$?
if [ "$GUARD_RC" -ne 0 ]; then
    if [ "$ALLOW_RUNNING_BOTS" -eq 0 ]; then
        echo "    REFUSING to deploy. rsync --delete plus a pip install under a"
        echo "    running process can tear its import tree and break pyarmor"
        echo "    runtime binding. Stop it (kill-verify-start), then redeploy."
        exit 1
    fi
    # The override covers a busy host, never an unknown one: there is nothing
    # to accept when we could not read the host's state at all.
    if [ "$GUARD_RC" -eq 2 ]; then
        echo "    --i-accept-a-torn-import-tree does NOT cover an unreadable host." >&2
        exit 1
    fi
    echo "    --i-accept-a-torn-import-tree given; continuing anyway."
fi

# ---- 2. rsync --------------------------------------------------------------
echo "--- rsync"
for algo in "${ALGOS[@]}"; do
    build_dir="$BUILD_ROOT/${algo}_pkg"
    [ -d "$build_dir" ] || { echo "    ERROR: no build at $build_dir" >&2; exit 1; }
    # shellcheck disable=SC2086
    rsync -az --delete -e "ssh $SSH_OPTS" "$build_dir/" "$HOST:$REMOTE_BASE/${algo}_pkg/"
    echo "    ${algo}_pkg -> $HOST:$REMOTE_BASE/${algo}_pkg/"
done

# ---- 3. install (editable, on EVERY path) ----------------------------------
echo "--- pip install (editable)"
PIP_ARGS=""
for algo in "${ALGOS[@]}"; do
    PIP_ARGS="$PIP_ARGS -e $REMOTE_BASE/${algo}_pkg/"
done
if ! ssh_run "$PY -m pip install $PIP_ARGS"; then
    echo "    ERROR: pip install failed on $HOST — the rsynced bytes are NOT live." >&2
    exit 1
fi

# ---- 4. verify -------------------------------------------------------------
# Expected digests are produced by the SAME script that checks them on the host,
# so the two sides cannot drift apart into a false pass.
echo "--- verify"
EXPECTS=""
for algo in "${ALGOS[@]}"; do
    digest="$(python3 "$VERIFIER" --print-digest "$BUILD_ROOT/${algo}_pkg/src")"
    EXPECTS="$EXPECTS --expect ${algo}=${digest}"
done
# shellcheck disable=SC2086
if ! ssh $SSH_OPTS "$HOST" "$PY - $REMOTE_BASE --algos $ALGOS_CSV $EXPECTS" < "$VERIFIER"; then
    echo ""
    echo "========================================================================"
    echo "DEPLOY TO $HOST FAILED VERIFICATION."
    echo "The rsync and the install reported success, but what python imports is"
    echo "not what was deployed. Do NOT log this as delivered, and do not run"
    echo "research or bots on $HOST until it is fixed."
    echo "========================================================================"
    exit 1
fi

echo "=== $HOST: deployed and VERIFIED ==="
