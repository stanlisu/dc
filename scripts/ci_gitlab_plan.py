#!/usr/bin/env python3
"""ci_gitlab_plan.py -- turn .github/workflows/tests.yml's `test` job into a runnable plan.

TEMPORARY (2026-09-16): GitHub Actions is billing-blocked on the stanlisu org, so dc's `Tests`
workflow never runs. dc now mirrors to a PRIVATE project `mirror/dc` on the self-hosted GitLab
CE on dev105, which runs the same job as a pipeline. GitHub stays the repo of record.

This is the dc twin of marvel's scripts/ci_local_plan.py, and it prevents drift the same two
mechanical ways:

  1. The `run:` steps are NOT copied into the GitLab config. They are read from tests.yml OF
     THE COMMIT UNDER TEST and executed verbatim, each with `bash -e` (the GitHub runner's
     default `shell:`), in step order, stopping at the first failure. Changing the TA-Lib pin,
     the pandas pin or the pytest invocation in tests.yml changes the pipeline with no edit
     here.

  2. Everything ELSE in the job -- runs-on, every `uses:` step and its `with:`, every `env:`,
     any step added later -- cannot be executed on GitLab and is emulated by
     scripts/ci_gitlab_run.sh instead. That part is PINNED: the job with the verbatim `run:`
     bodies blanked out is serialised and compared, byte for byte, against the committed
     scripts/ci_gitlab_skeleton.json. Any difference is a hard error that prints the diff, so
     a new step or a changed checkout option stops the pipeline until a human has taught the
     runner about it and re-pinned with --write-skeleton.

Usage:
  ci_gitlab_plan.py --ci-yml FILE --out DIR   # write a plan
  ci_gitlab_plan.py --ci-yml FILE --check     # drift check only
  ci_gitlab_plan.py --ci-yml FILE --write-skeleton   # re-pin (review the diff first)

Plan layout (DIR):
  meta.env    PYTHON_VERSION=3.13, CI_YML_SHA256=...
  steps.tsv   <NN>\\t<step name>   for each verbatim step, in order
  <NN>.run    the step's run block, verbatim
  <NN>.env    KEY=VALUE per line, the step's env (no ${{ }} expressions are accepted)
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import pathlib
import re
import sys

import yaml

SKELETON_PATH = pathlib.Path(__file__).resolve().with_name("ci_gitlab_skeleton.json")

JOB = "test"
EXPECTED_JOBS = (JOB,)

# Steps executed verbatim, in tests.yml order. Anything not listed is emulated by
# scripts/ci_gitlab_run.sh and therefore pinned by the skeleton.
VERBATIM_STEPS = (
    "Generate vendored obfuscation codec",
    "Install TA-Lib (C library + python bindings)",
    "Install dependencies",
    "Run tests",
)

# The `uses:` step that fixes the interpreter, and the key holding the version. The image in
# .gitlab-ci.yml must match it; ci_gitlab_run.sh asserts that at runtime.
SETUP_PYTHON_PREFIX = "actions/setup-python@"

PLACEHOLDER = "<executed verbatim by scripts/ci_gitlab_run.sh>"


class DriftError(RuntimeError):
    """tests.yml no longer matches what ci_gitlab_run.sh knows how to emulate."""


def load_job(ci_yml_text: str) -> dict:
    data = yaml.safe_load(ci_yml_text)
    if not isinstance(data, dict) or "jobs" not in data:
        raise DriftError("tests.yml has no top-level `jobs:` mapping")
    jobs = data["jobs"]
    if tuple(sorted(jobs)) != EXPECTED_JOBS:
        raise DriftError(
            f"tests.yml jobs are {sorted(jobs)}, ci_gitlab_run.sh emulates exactly {list(EXPECTED_JOBS)}")
    return jobs


def skeleton(jobs: dict) -> dict:
    """The jobs mapping with each verbatim step's `run` body replaced by a placeholder."""
    sk = copy.deepcopy(jobs)
    seen = []
    for step in sk[JOB].get("steps", []):
        name = step.get("name")
        if name in VERBATIM_STEPS:
            if "run" not in step:
                raise DriftError(f"step {name!r} has no `run:` block")
            step["run"] = PLACEHOLDER
            seen.append(name)
    if tuple(seen) != VERBATIM_STEPS:
        raise DriftError(f"verbatim steps found in order {seen}, expected {list(VERBATIM_STEPS)}")
    return sk


def skeleton_text(jobs: dict) -> str:
    return json.dumps(skeleton(jobs), indent=2, sort_keys=True) + "\n"


def check_skeleton(jobs: dict, pinned_text: str) -> None:
    actual = skeleton_text(jobs)
    if actual == pinned_text:
        return
    diff = "".join(difflib.unified_diff(
        pinned_text.splitlines(True), actual.splitlines(True),
        fromfile="scripts/ci_gitlab_skeleton.json (pinned)", tofile="tests.yml at the tested commit"))
    raise DriftError(
        "tests.yml's non-`run:` structure changed; scripts/ci_gitlab_run.sh emulates those "
        "parts and has NOT been taught about the change.\n"
        "Update ci_gitlab_run.sh for it, then re-pin with\n"
        "  python scripts/ci_gitlab_plan.py --ci-yml .github/workflows/tests.yml --write-skeleton\n"
        f"pinned sha256={hashlib.sha256(pinned_text.encode()).hexdigest()} "
        f"actual sha256={hashlib.sha256(actual.encode()).hexdigest()}\n{diff}")


def python_version(jobs: dict) -> str:
    for step in jobs[JOB]["steps"]:
        uses = str(step.get("uses", ""))
        if uses.startswith(SETUP_PYTHON_PREFIX):
            want = step.get("with", {}).get("python-version")
            if not want:
                raise DriftError(f"{uses} has no `with: python-version:`")
            return str(want)
    raise DriftError("tests.yml has no actions/setup-python step, so the interpreter is undefined")


def build_plan(jobs: dict) -> tuple[dict, list[tuple[str, str, dict]]]:
    steps = []
    for step in jobs[JOB]["steps"]:
        name = step.get("name")
        if name not in VERBATIM_STEPS:
            continue
        run = step["run"]
        if "${{" in run:
            raise DriftError(f"step {name!r} run block uses a ${{{{ }}}} expression; it cannot run verbatim")
        env = {}
        for key, val in (step.get("env") or {}).items():
            val = str(val)
            if "${{" in val:
                raise DriftError(f"step {name!r} env {key}={val!r}: expressions are not supported here")
            if "\n" in val or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise DriftError(f"step {name!r} env {key!r} cannot be passed as KEY=VALUE")
            env[key] = val
        steps.append((name, run, env))
    return {"PYTHON_VERSION": python_version(jobs)}, steps


def write_plan(out: pathlib.Path, meta: dict, steps: list, ci_yml_text: str) -> None:
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(meta, CI_YML_SHA256=hashlib.sha256(ci_yml_text.encode()).hexdigest())
    (out / "meta.env").write_text("".join(f"{k}={v}\n" for k, v in meta.items()))
    rows = []
    for i, (name, run, env) in enumerate(steps, start=1):
        nn = f"{i:02d}"
        (out / f"{nn}.run").write_text(run)
        (out / f"{nn}.env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
        rows.append(f"{nn}\t{name}\n")
    (out / "steps.tsv").write_text("".join(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ci-yml", required=True, type=pathlib.Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=pathlib.Path, help="write a plan directory (must not exist)")
    mode.add_argument("--check", action="store_true", help="drift check only")
    mode.add_argument("--write-skeleton", action="store_true", help="re-pin the skeleton")
    args = ap.parse_args(argv)

    text = args.ci_yml.read_text()
    try:
        jobs = load_job(text)
        if args.write_skeleton:
            SKELETON_PATH.write_text(skeleton_text(jobs))
            print(f"ci_gitlab_plan: pinned {SKELETON_PATH}")
            return 0
        if not SKELETON_PATH.is_file():
            raise DriftError(f"{SKELETON_PATH} is missing -- nothing to check tests.yml against")
        check_skeleton(jobs, SKELETON_PATH.read_text())
        if args.check:
            print("ci_gitlab_plan: tests.yml skeleton matches the pin")
            return 0
        meta, steps = build_plan(jobs)
        write_plan(args.out, meta, steps, text)
    except DriftError as exc:
        print(f"ci_gitlab_plan: DRIFT: {exc}", file=sys.stderr)
        return 3
    print(f"ci_gitlab_plan: wrote {len(steps)} verbatim step(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
