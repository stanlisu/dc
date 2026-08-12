# dc — private algo source + PyArmor distribution builder

`dc` ("distribution core") holds the **private source** of every marvel algo
package and builds the **obfuscated artifacts** that ship to the servers.
marvel itself is being made public; the model/feature/regime logic lives here
and reaches production only as PyArmor-obfuscated bytecode.

Two things happen in this repo, and only here:

1. **PyArmor build + deploy** — `<algo>_pkg/src/<algo>/` → obfuscated
   `build_dist/<algo>_pkg/` → rsync to `<host>:/home/stan/sandbox/marvel/<algo>_pkg/`
   → **editable install** → **verify**. The last two steps are not optional —
   see [rsync is not the deploy](#rsync-is-not-the-deploy).
2. **Name obfuscation** — the reversible regime/feature name↔code map
   (`obfuscation/`), vendored into every package so the obfuscated code can
   decode at runtime on the servers.

---

## Layout

| Path | What |
|---|---|
| `<algo>_pkg/` | One package per algo: `src/<algo>/`, `setup.py`, `tests/`, `README.md` |
| `build_dist/` | PyArmor output (gitignored build artifact — never edit by hand) |
| `obfuscation/` | Codec, inventory extractor, map builder, vendor sync |
| `build_distribution.sh` | The single front door for build + deploy |
| `scripts/deploy_host.sh` | One host, end to end: pre-flight → rsync → editable install → verify. Shared by `build_distribution.sh` and CI |
| `scripts/verify_deploy.py` | The post-deploy gate. Runs on the target; proves the deployed bytes are the imported bytes |
| `deploy.log` | **Append one entry per deploy** (see Logging below) |
| `conftest.py` / `pytest.ini` | Regenerate map + vendored codec before every test session |

**Nine algo packages:** `agamotto orb aether scepter mjolnir stormbreaker
vibranium valkyrie vomir`. `mjolnir` and `valkyrie` nest their code under
`src/<algo>/core/`; the rest are flat.

---

## Build & deploy

```bash
cd ~/Documents/sandbox/dc
./build_distribution.sh --build-only        # build all 9, no deploy
./build_distribution.sh --build-only mjolnir # build one algo
./build_distribution.sh                      # build all, then prompt for a deploy target
```

The deploy prompt offers `1) hydra  2) shield  3) shield2  4) all  5) xmen
6) all + xmen  7) skip` — note there is **no "hydra + shield2" combo**. To
deploy to an arbitrary subset, build with `--build-only` and drive one host at
a time with the same script the front door uses:

```bash
bash scripts/deploy_host.sh --host shield \
     --python /opt/miniconda3/envs/py313/bin/python --algos mjolnir,orb
```

Do **not** hand-roll the `rsync` line. rsync alone does not deploy anything —
see below.

`build_distribution.sh` refreshes the obfuscation map and re-vendors the codec
into every package **before** running PyArmor, so a deploy can never ship a
stale map.

### rsync is not the deploy

An rsync into `<host>:/home/stan/sandbox/marvel/<algo>_pkg/` only changes
whether python imports that tree **if the package is installed editable**. A
non-editable install copied the tree into `site-packages` at install time, and
that copy is frozen: rsync never reaches it, the deploy prints success, and the
old code keeps running.

That is not hypothetical. `deploy.yml` used to run
`pip install -e A/ B/ C/ ...`, and **pip binds `-e` only to the path that
immediately follows it** — so `agamotto` was editable and the other **eight**
were frozen copies. Measured on shield 2026-08-09: `mjolnir.core.ladder`
imported a **2026-08-06** copy from `site-packages` while the deployed tree
held the **2026-08-08** build; shield2 held a third version again. Every deploy
to shield from 2026-08-06 was a silent no-op for 8 of 9 packages, and it cost a
research panel rebuild before anyone noticed.

Two rules follow, and both are now enforced in code:

1. **Every package is installed editable, on every host** — `-e` before *each*
   path, so the rsynced tree is the single copy and stays authoritative.
2. **Every deploy is verified on the target** by `scripts/verify_deploy.py`,
   which fails non-zero unless the module python imports **is** the tree we
   rsynced, with a matching content digest.

> The old runbook check here used `sys.path.insert(0, ".../mjolnir_pkg/src")`
> before importing. That could never detect this: forcing the deployed tree
> onto `sys.path` is precisely what hides which tree a normal run would pick.
> And "does it import" is worthless as a staleness test — **a stale package
> imports perfectly**. Check identity, not importability.

Verify by hand any time you doubt what a host is running:

```bash
ssh shield '/opt/miniconda3/envs/py313/bin/python - /home/stan/sandbox/marvel' \
    < scripts/verify_deploy.py
```

### Rules

- **Build on the Mac, target linux.x86_64.** `pyarmor gen --platform
  linux.x86_64` cross-compiles; the servers never run PyArmor.
- **Never deploy onto a host with running bots.** rsync uses `--delete`; a bot
  that imports a package mid-sync gets a torn tree, and a `pip install` swaps
  `pyarmor_runtime.so` out from under an already-loaded one. `deploy_host.sh`
  now refuses such a host instead of trusting you to check. It had been only a
  note: CI rsynced and reinstalled all nine packages on hydra at
  2026-08-09 03:42Z while `run_knull.py` had been live with real money since
  2026-08-08 07:11Z.
- **`shield`'s python is `/opt/miniconda3/envs/py313`**, not `~/miniconda3`
  (hydra and shield2 use `~/miniconda3/envs/py313`).
- **`shield` is the host `deploy.yml` cannot reach** — it is a LAN address
  (192.168.72.135), unreachable from GitHub-hosted runners, so its step is
  `continue-on-error` and CI has never installed on it. shield therefore drifts
  furthest and must be deployed manually from the Mac. (hydra and shield2 both
  have CI steps.)

### PyArmor licensing

The local install is **PyArmor 9.2.3 trial**. The trial has a build quota and
can fail mid-build with `RuntimeError: out of license`, writing
`pyarmor.bug.log` in the repo root (seen 2026-06-29). It is transient — a full
9-package build succeeded 2026-07-28. If it recurs, build algo-by-algo rather
than assuming the source is broken.

### Obfuscated output is per-build, not reproducible

Every `pyarmor gen` run emits different bytes **and a differently-keyed
`pyarmor_runtime.so`**, even from identical source. Consequences:

- Diffing two builds of the same commit shows every file as changed. That is
  **not** evidence of a source difference — compare `_obf/map.json` checksums,
  file counts, and the actual exposed API instead.
- The obfuscated `.py` files are bound to the `.so` built alongside them.
  **Never rsync a package's `.py` files without its matching
  `pyarmor_runtime_000000/`** — always sync the whole `<algo>_pkg/` tree.

---

## Name obfuscation (`obfuscation/`)

Regime atoms and feature columns are replaced by short reversible codes so
public marvel never carries the private vocabulary. Two namespaces, coded
separately: **72 regimes** (`r###`) and **100 features** (`f###`) as of
map version 1.

```bash
cd ~/Documents/sandbox/dc
PYTHONPATH=$(ls -d *_pkg/src | paste -sd: -) python obfuscation/extract_inventory.py > obfuscation/inventory.json
python obfuscation/build_map.py     # inventory.json -> map.json
python obfuscation/sync_vendor.py   # map.json + codec.py -> every <pkg>/_obf/
```

| File | Role |
|---|---|
| `extract_inventory.py` | AST scan of `dc/*_pkg` for regime atoms + feature columns (no runtime/TA-Lib needed) |
| `build_map.py` | `inventory.json` → `map.json`, **stable and append-only** |
| `codec.py` | Encode/decode, preserving structure (`_and_`, `_long`/`_short`, TF prefixes) |
| `sync_vendor.py` | Vendors `codec.py` + `map.json` into each `<pkg>/_obf/` |

### The append-only invariant (production-breaking if violated)

`build_map.py` is **APPEND-ONLY as of 2026-07-24**: codes already in the
committed `map.json` are preserved forever, new names get the next unused
code, and retired names **keep** their entry (never reused). Deployed
artifacts — coded weights meta, filter parquet columns, stack CSVs —
reference codes by value, so renumbering silently invalidates production
data. The earlier sequential enumeration renumbered everything after any
removal (caught when dropping `oi_velocity`/`oi_acceleration` shifted
`r053..r072` and `f062..f100`). Never "clean up" or re-sequence `map.json`.

### Vendoring, and why each package gets its own copy

The codec must be importable *inside* each package at runtime, but
`build_distribution.sh` only ships `<algo>_pkg/` dirs — not `dc/obfuscation/`.
Several packages (mjolnir, stormbreaker, valkyrie) don't depend on agamotto,
so there is no shared location. Hence one vendored copy per package, imported
as `from <pkg>._obf.codec import default`. The `<pkg>/_obf/` copies are
gitignored (derived); `conftest.py` regenerates them before every test session
so a fresh clone can run tests with no manual build step.

**`map.json` is a DATA file.** `pyarmor gen` only processes `.py` and drops it,
and pip drops non-listed data files — which is why `build_distribution.sh`
copies it explicitly into the build output *and* patches `setup.py` with
`package_data={"": [..., "*.json"]}`. If a deploy ever fails with a codec
load error, check that `<pkg>/src/<pkg>/_obf/map.json` survived.

**`vibranium_pkg` is deliberately excluded** from `sync_vendor._TARGETS` — it
carries no `_obf/` and ships no `map.json`. Eight of nine packages are coded;
a missing `map.json` under `vibranium_pkg` is expected, not a build failure.

---

## Tests

```bash
cd ~/Documents/sandbox/dc && pytest            # conftest regenerates map + vendored codec first
pytest mjolnir_pkg/tests -q                    # one package
```

---

## Logging

Every build/deploy **must** append a timestamped entry to `deploy.log`
(marvel's "log every operational command" rule). Record: what was built, from
which dc commit, which hosts received it, rsync success counts, and the
verification actually run on each target. Template:

```
YYYY-MM-DD HH:MM [local] rebuild+redeploy <algos> -> <hosts> (main@<sha>, <what changed>)
  - built via ./build_distribution.sh --build-only (pyarmor <ver>, linux.x86_64)
  - rsync build_dist/<algo>_pkg/ -> <host>:/home/stan/sandbox/marvel/<algo>_pkg/ (N/N OK)
  - verified on <hosts>: <the import/API check you ran and its result>
```

---

## Downstream consumers

Deployed packages land at `/home/stan/sandbox/marvel/<algo>_pkg/` and are
imported from `<pkg>/src` **via the editable install**, not via `PYTHONPATH`.

This distinction matters and used to be stated wrongly here. marvel's entry
points export only `PYTHONPATH="agamotto_pkg/src:."` (e.g.
`mjolnir/gauntlet/run_mjolnir_pipeline.sh:26`, `gauntlet/run_pipeline.sh:54`,
every `stormbreaker/gauntlet/*.sh`) — **one** package. The other eight are
found purely through `site-packages`, which is why their install mode silently
decided whether a deploy took effect. Three resolution orders exist today:

| Entry point | mjolnir/stormbreaker resolve from |
|---|---|
| `run_mjolnir_pipeline.sh`, all `stormbreaker/gauntlet/*.sh`, `gauntlet/run_pipeline.sh` | the install (`site-packages`) |
| `mjolnir/gauntlet/explore_tick.sh`, `run_mjolnir_rolling.sh` | `MJOLNIR_PKG_SRC`, default `/home/stan/sandbox/dc/mjolnir_pkg/src` — the **deobfuscated dc clone**, whatever git state it is in |
| `mjolnir/gauntlet/run_era_variant.sh` | `$REPO/mjolnir_pkg/src` — the deployed pyarmor tree |

Two runs of "the same" algo on one host can therefore execute different code.
`scripts/verify_deploy.py` tells you which tree the *install* resolves to; for
the `MJOLNIR_PKG_SRC` entry points also check `git -C ~/sandbox/dc log -1`.

**xmen** (the private knull mirror, live trading stack) keeps its **own
git-tracked copy** of `mjolnir_pkg` at `~/sandbox/xmen/mjolnir_pkg` — it is
*not* a symlink and is *not* updated by `build_distribution.sh`. When a dc
change must reach the live bridge, refresh xmen's copy and commit it there
(e.g. xmen `275b6c9`, "refresh pyarmor build with BTC compute-once (dc #22)").
Before concluding xmen is stale, **verify it** rather than trusting a prereq
note — per-build byte differences make it look outdated when it is not:

```bash
ssh hydra '~/miniconda3/envs/py313/bin/python -c "
import sys, inspect; sys.path.insert(0, \"/home/stan/sandbox/xmen/mjolnir_pkg/src\")
from mjolnir.trading import MjolnirTrading
print(list(inspect.signature(MjolnirTrading.predict_from_inputs).parameters))"'
```
