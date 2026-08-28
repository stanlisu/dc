# shellcheck shell=bash
# bot_guard.sh -- THE shared "is this host busy?" gate for every deploy path.
#
# Source it; do not execute it:
#
#     . "$(dirname "$0")/bot_guard.sh"                       # dc/scripts/deploy_host.sh
#     . "${dc_root}/scripts/bot_guard.sh"                    # marvel/scripts/sync_all_repos.sh
#
#     listing="$(ssh "$host" "$BOT_GUARD_PS_CMD" 2>/dev/null)" || listing=""
#     findings="$(printf '%s\n' "$listing" | bot_guard_classify)"; rc=$?
#     case $rc in
#       0) : deploy ;;
#       1) : live trading/ops -- HARD STOP ;;
#       3) : research job in flight -- refuse, but the operator may prefer to wait ;;
#       2) : listing unusable -- host state UNKNOWN, never deploy blind ;;
#     esac
#
# It lives in dc, not marvel, because dc's Deploy workflow checks out ONLY dc
# (.github/workflows/deploy.yml) -- a shared file in marvel would be absent in
# CI. marvel's sync_all_repos.sh already locates dc_root and already aborts when
# ${dc_root}/scripts/deploy_host.sh is missing, so sourcing one more file from
# the same directory adds no new coupling.
#
# ssh stays in the CALLER: deploy_host.sh carries SSH_OPTS as a word-split
# string, sync_all_repos.sh as a bash array. What is shared is the thing that
# was wrong in both -- the pattern and the classification.
#
# THE REMOTE COMMAND CARRIES NO PATTERN. The old gates ran the match on the far
# side (`ps | grep -E 'run_knull|...' | wc -l`), where the grep's own command
# line is in the very listing it is grepping. This ships a plain `ps` back and
# classifies it locally, so there is no pattern on any remote command line to
# self-match. See scripts/bot_guard.py for how argv[0] is resolved.

# ${BASH_SOURCE[0]:-$0}: BASH_SOURCE is unset when this is sourced from a
# non-bash shell, and `dirname ""` is ".", which silently resolves the classifier
# to the CALLER's cwd. That is a path fallback of exactly the kind CLAUDE.md
# bans, and it bit on the first live run of this file.
BOT_GUARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# `pid=,etime=,args=` suppresses the header on every ps we run on (GNU and BSD).
BOT_GUARD_PS_CMD='ps -eo pid=,etime=,args='
BOT_GUARD_PY="${BOT_GUARD_PY:-python3}"
BOT_GUARD_IMPL="${BOT_GUARD_DIR}/bot_guard.py"

# Classify a ps listing on stdin. Prints one line per finding; see exit codes above.
bot_guard_classify() {
    # A missing classifier must not be reported as an unusable LISTING: the two
    # both refuse, but they send an operator looking in different places, and
    # python's own "can't open file" exit code is 2 -- the same as RC_UNUSABLE.
    if [ ! -f "$BOT_GUARD_IMPL" ]; then
        echo "bot_guard: classifier missing at ${BOT_GUARD_IMPL}" >&2
        return 4
    fi
    "$BOT_GUARD_PY" "$BOT_GUARD_IMPL"
}

# Human-readable verdict for ONE host. Reads the listing on stdin, prints an
# indented report, and returns the same code bot_guard_classify does.
#   bot_guard_report <host> <<<"$listing"
bot_guard_report() {
    local host="$1" findings rc
    findings="$(bot_guard_classify)"; rc=$?
    case "$rc" in
        0) echo "    ${host}: clear -- no trading, ops or research process running" ;;
        1) echo "    ${host}: LIVE TRADING/OPS PROCESSES -- refusing:"
           printf '%s\n' "$findings" | sed 's/^/      /' ;;
        3) echo "    ${host}: RESEARCH job in flight -- refusing (a pip install"
           echo "             mid-run corrupts it). Wait for it or stop it:"
           printf '%s\n' "$findings" | sed 's/^/      /' ;;
        2) echo "    ${host}: ps listing UNUSABLE -- host state unknown, never deploy blind" ;;
        4) echo "    ${host}: bot_guard classifier is MISSING -- cannot check the host" ;;
        *) echo "    ${host}: bot_guard_classify returned unexpected ${rc} -- treating as unsafe"
           rc=2 ;;
    esac
    return "$rc"
}
