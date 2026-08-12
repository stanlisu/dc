"""Regression tests for the post-deploy gate.

The bug this guards against: a deploy rsyncs new bytes into
`<root>/<algo>_pkg/src/<algo>` while python keeps importing a frozen copy in
site-packages, and every check we had still passed. So the load-bearing
assertion is `test_shadowed_install_fails` — a package that imports PERFECTLY
but from the wrong tree must FAIL.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_REPO / "scripts"))
import verify_deploy  # noqa: E402

ALGO = "orb"  # a real algo name so NO_OBF_MAP / DEEP_IMPORT lookups behave


def _make_pkg(base: pathlib.Path, name: str, body: str = "VALUE = 1\n") -> pathlib.Path:
    pkg = base / name
    (pkg / "_obf").mkdir(parents=True)
    (pkg / "__init__.py").write_text(body)
    (pkg / "_obf" / "map.json").write_text(json.dumps({"version": 1}))
    return pkg


def _is_editable_finder(finder) -> bool:
    """True for the MetaPathFinder a PEP 660 `pip install -e` registers."""
    return getattr(type(finder), "__module__", "").startswith("__editable__")


def _is_repo_pkg_src(entry: str) -> bool:
    """True for a sys.path entry pointing at this repo's own <algo>_pkg/src."""
    try:
        p = pathlib.Path(entry).resolve()
    except (OSError, ValueError):
        return False
    return p.name == "src" and p.parent.name.endswith("_pkg") and _REPO in p.parents


@pytest.fixture(autouse=True)
def _clean_import_state():
    saved_path = list(sys.path)
    saved_meta = list(sys.meta_path)
    saved_mods = dict(sys.modules)
    # DEEP_IMPORT would pull in the real orb.research; the fixtures here are
    # stubs, so drop the deep import for the duration of these tests.
    saved_deep = dict(verify_deploy.DEEP_IMPORT)
    verify_deploy.DEEP_IMPORT.pop(ALGO, None)

    # These tests aim `import <algo>` at a temp tree via sys.path. THREE things
    # in a populated environment outrank that, and CI has all three:
    #
    #   1. an already-imported copy in sys.modules — the full suite imports the
    #      repo's own orb long before this file runs, and a cached module wins
    #      over any path;
    #   2. the PEP 660 editable MetaPathFinder that `pip install -e` registers
    #      (tests.yml installs all 9 packages editable). sys.meta_path is
    #      consulted BEFORE sys.path, so sys.path.insert(0, ...) cannot outrank
    #      it — this is the one that actually bit;
    #   3. a legacy .pth-style editable that appends <algo>_pkg/src to sys.path.
    #
    # Running this file ALONE on a machine where the packages are not installed
    # hits none of them, which is why these passed locally and failed in CI
    # (dc#38: "imported: /home/runner/work/dc/dc/orb_pkg/src/orb").
    for name in list(sys.modules):
        if name.split(".")[0] in (ALGO, "vibranium"):
            del sys.modules[name]
    sys.meta_path[:] = [f for f in sys.meta_path if not _is_editable_finder(f)]
    sys.path[:] = [e for e in sys.path if not _is_repo_pkg_src(e)]
    importlib.invalidate_caches()

    yield

    sys.path[:] = saved_path
    sys.meta_path[:] = saved_meta
    sys.modules.clear()
    sys.modules.update(saved_mods)
    verify_deploy.DEEP_IMPORT.clear()
    verify_deploy.DEEP_IMPORT.update(saved_deep)
    importlib.invalidate_caches()


def _root_with_deployed(tmp_path: pathlib.Path, body: str = "VALUE = 1\n"):
    root = tmp_path / "marvel"
    src = root / ("%s_pkg" % ALGO) / "src"
    src.mkdir(parents=True)
    _make_pkg(src, ALGO, body)
    return root, src


def test_shadowed_install_fails(tmp_path):
    """The whole point: imports fine, but from a DIFFERENT tree -> FAIL."""
    root, _ = _root_with_deployed(tmp_path)
    shadow = tmp_path / "site-packages"
    shadow.mkdir()
    _make_pkg(shadow, ALGO, "VALUE = 2\n")

    sys.path.insert(0, str(shadow))
    importlib.invalidate_caches()

    ok, lines = verify_deploy.check(ALGO, str(root), {})
    assert ok is False
    blob = "\n".join(lines)
    assert "SHADOWED" in blob
    # It must name both trees, or the operator cannot act on the failure.
    assert str(shadow / ALGO) in blob
    assert str(root / ("%s_pkg" % ALGO) / "src" / ALGO) in blob


def test_editable_style_resolution_passes(tmp_path):
    """Resolving to the deployed tree itself (what an editable install does)."""
    root, src = _root_with_deployed(tmp_path)
    sys.path.insert(0, str(src))
    importlib.invalidate_caches()

    ok, lines = verify_deploy.check(ALGO, str(root), {})
    assert ok is True, "\n".join(lines)
    assert any(ln.startswith("  OK") for ln in lines)


def test_digest_mismatch_fails(tmp_path):
    """A partial/interrupted rsync leaves the right path with wrong bytes."""
    root, src = _root_with_deployed(tmp_path)
    sys.path.insert(0, str(src))
    importlib.invalidate_caches()

    ok, lines = verify_deploy.check(ALGO, str(root), {ALGO: "0" * 64})
    assert ok is False
    assert "DIGEST MISMATCH" in "\n".join(lines)


def test_missing_install_fails(tmp_path):
    """Deployed but never installed is a failed deploy, not a pass."""
    root, _ = _root_with_deployed(tmp_path)
    importlib.invalidate_caches()
    ok, lines = verify_deploy.check(ALGO, str(root), {})
    assert ok is False
    assert "not importable" in "\n".join(lines)


def test_missing_obf_map_fails(tmp_path):
    """pyarmor drops DATA files and pip drops unlisted package_data."""
    root, src = _root_with_deployed(tmp_path)
    (src / ALGO / "_obf" / "map.json").unlink()
    sys.path.insert(0, str(src))
    importlib.invalidate_caches()

    ok, lines = verify_deploy.check(ALGO, str(root), {})
    assert ok is False
    assert "map.json missing" in "\n".join(lines)


def test_vibranium_needs_no_obf_map(tmp_path):
    """vibranium is excluded from sync_vendor._TARGETS by design."""
    algo = "vibranium"
    root = tmp_path / "marvel"
    src = root / ("%s_pkg" % algo) / "src"
    src.mkdir(parents=True)
    pkg = src / algo
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VALUE = 1\n")

    sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    ok, lines = verify_deploy.check(algo, str(root), {})
    assert ok is True, "\n".join(lines)


def test_digest_ignores_pycache_and_egg_info(tmp_path):
    """Host-side __pycache__ must not make a correct deploy look wrong."""
    src = tmp_path / "src"
    _make_pkg(src, ALGO)
    before = verify_deploy.tree_digest(str(src))

    (src / ALGO / "__pycache__").mkdir()
    (src / ALGO / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"\x00\x01")
    (src / ("%s.egg-info" % ALGO)).mkdir()
    (src / ("%s.egg-info" % ALGO) / "PKG-INFO").write_text("junk")
    assert verify_deploy.tree_digest(str(src)) == before

    # ...but a real shipped byte must change it.
    (src / ALGO / "__init__.py").write_text("VALUE = 999\n")
    assert verify_deploy.tree_digest(str(src)) != before


def test_digest_is_none_for_missing_dir(tmp_path):
    assert verify_deploy.tree_digest(str(tmp_path / "nope")) is None
