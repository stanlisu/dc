#!/usr/bin/env bash
# scripts/ci_gitlab_run.sh -- the body of .gitlab-ci.yml's `test` job.
#
# Runs .github/workflows/tests.yml's `test` job inside the GitLab job container on the dev105
# runner, while GitHub Actions is billing-blocked. GitHub stays the repo of record.
#
# Usage (normally only from .gitlab-ci.yml):
#   bash scripts/ci_gitlab_run.sh
#
# WHAT IS EMULATED HERE, and why that is safe:
#   actions/checkout       -> GitLab's own clone (GIT_STRATEGY: clone in .gitlab-ci.yml).
#   actions/setup-python   -> the job image's CPython. The minor version is ASSERTED against
#                             the version tests.yml's setup-python step asks for, never
#                             assumed, so bumping one without the other fails loudly.
#   ubuntu-latest's sudo   -> apt-get installs `sudo`. tests.yml's TA-Lib step runs
#                             `sudo make install` / `sudo ldconfig`; the container is already
#                             root, but the literal command must still resolve, and the step
#                             runs VERBATIM rather than being edited here.
# Everything NOT in that list -- the four `run:` steps -- is read from the tested commit's own
# tests.yml by scripts/ci_gitlab_plan.py and executed verbatim, in order, each with `bash -e`
# (the GitHub runner's default `shell:`), stopping at the first failure. The non-`run:`
# structure of tests.yml is pinned by scripts/ci_gitlab_skeleton.json, so any structural change
# stops this pipeline with a diff instead of running something different.
#
# NO SILENT SKIPS. dc's own tests.yml records what a quietly degraded environment costs: a CI
# without TA-Lib does not skip those tests, it computes DIFFERENT MATH. Nothing here is
# allowed to continue past a missing input.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_PY="${SCRIPT_DIR}/ci_gitlab_plan.py"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
CI_YML=".github/workflows/tests.yml"

say() { echo "::ci_gitlab:: $*"; }
die() { echo "::ci_gitlab:: FAILED: $*" >&2; exit 1; }

[[ -f "$PLAN_PY" ]] || die "${PLAN_PY} not found"
[[ -f "${SCRIPT_DIR}/ci_gitlab_skeleton.json" ]] || die "scripts/ci_gitlab_skeleton.json not found"
cd "$REPO" || die "cannot cd into ${REPO}"
git rev-parse --git-dir >/dev/null 2>&1 \
    || die "${REPO} is not a git checkout (GIT_STRATEGY must be clone, not none)"
[[ -f "$CI_YML" ]] || die "this commit has no ${CI_YML} to read"

HEAD_SHA="$(git rev-parse --verify HEAD)" || die "cannot resolve HEAD"
say "commit ${HEAD_SHA}"

# ubuntu-latest has passwordless sudo; the bookworm python image has no sudo at all.
if ! command -v sudo >/dev/null 2>&1; then
    say "installing sudo (tests.yml's TA-Lib step calls it; ubuntu-latest has it)"
    apt-get update -qq >/dev/null && apt-get install -y -qq sudo >/dev/null \
        || die "cannot install sudo; tests.yml's TA-Lib step would fail on a missing command"
fi

python -m pip install --quiet --disable-pip-version-check pyyaml \
    || die "cannot install pyyaml (needed only to read tests.yml)"
PLAN="$(mktemp -d)/plan" || die "mktemp failed"
python "$PLAN_PY" --ci-yml "$CI_YML" --out "$PLAN" </dev/null \
    || die "plan generation failed (a DRIFT message above means tests.yml changed shape -- see scripts/ci_gitlab_plan.py)"

# shellcheck source=/dev/null
. "${PLAN}/meta.env"
: "${PYTHON_VERSION:?plan meta.env lacks PYTHON_VERSION}"
GOT_PY="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" || die "no python in the image"
[[ "$GOT_PY" == "$PYTHON_VERSION" ]] \
    || die "job image python ${GOT_PY} != tests.yml's setup-python ${PYTHON_VERSION}; change .gitlab-ci.yml's image: to python:${PYTHON_VERSION}-bookworm"
say "python ${GOT_PY} (matches tests.yml)"

FAKE_HOME="$(mktemp -d)" || die "mktemp failed"
git config --file "${FAKE_HOME}/.gitconfig" --add safe.directory '*' || die "cannot seed git config"

# Explicit and closed: GitLab's own variables (CI_JOB_TOKEN above all) are NOT visible to pip
# or pytest. CI/GITHUB_ACTIONS are set because the tested code is tests.yml's code.
BASE_ENV=(
    "HOME=${FAKE_HOME}"
    "PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
    "LANG=C.UTF-8"
    "LC_ALL=C.UTF-8"
    "TZ=UTC"
    "CI=true"
    "GITHUB_ACTIONS=true"
    "PIP_ROOT_USER_ACTION=ignore"
    "PIP_DISABLE_PIP_VERSION_CHECK=1"
)

n_steps=0
while IFS=$'\t' read -r nn name; do
    [[ -n "$nn" ]] || continue
    [[ -f "${PLAN}/${nn}.run" && -f "${PLAN}/${nn}.env" ]] || die "plan step ${nn} is incomplete"
    step_env=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && step_env+=("$line")
    done < "${PLAN}/${nn}.env"
    say "STEP ${nn} START: ${name}"
    t0=$(date +%s)
    ( cd "$REPO" && exec env -i "${BASE_ENV[@]}" ${step_env[@]+"${step_env[@]}"} \
          bash -e "${PLAN}/${nn}.run" ) </dev/null
    rc=$?
    say "STEP ${nn} END rc=${rc} secs=$(( $(date +%s) - t0 )): ${name}"
    [[ $rc -eq 0 ]] || die "step ${nn} (${name}) exited ${rc}"
    n_steps=$((n_steps + 1))
done < "${PLAN}/steps.tsv"

[[ $n_steps -gt 0 ]] || die "steps.tsv listed no steps"
say "ALL STEPS OK (${n_steps} verbatim step(s)) sha=${HEAD_SHA} python=${GOT_PY}"
exit 0
