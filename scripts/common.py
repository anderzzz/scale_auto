"""Shared utilities: config, paths, logging, retry-with-backoff, seed."""
import os
import sys
import time
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
CHARTS = OUTPUT / "charts"
LOGS = ROOT / "logs"
STATUS_LOG = LOGS / "status.log"

for d in (CACHE, OUTPUT, CHARTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
SEED = CONFIG.get("random_seed", 17)


def set_seed():
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)


def _ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"{_ts()}  {msg}"
    print(line, flush=True)
    with open(STATUS_LOG, "a") as f:
        f.write(line + "\n")


def phase_start(name):
    log(f"[PHASE START] {name}")


def phase_end(name, extra=""):
    log(f"[PHASE END]   {name} {extra}".rstrip())


def warn(msg):
    log(f"[WARN] {msg}")


def exists_nonempty(path):
    """True if path exists and is non-empty (file >0 bytes, dir has entries)."""
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        return any(p.iterdir())
    return p.stat().st_size > 0


def retry(fn, *, tries=5, base_delay=2.0, max_delay=60.0, what="operation"):
    """Call fn() with exponential backoff. Fail loud after `tries`."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt == tries:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            warn(f"{what} failed (attempt {attempt}/{tries}): {e}. Retrying in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(
        f"{what} failed after {tries} attempts. Last error: {last}. "
        f"ACTION: check network/host availability and rerun this phase."
    )


def download(url, dest, *, min_bytes=0, what=None):
    """Stream-download url to dest with retry. Skip if already non-empty & big enough."""
    import requests
    dest = Path(dest)
    what = what or f"download {url}"
    if dest.exists() and dest.stat().st_size > min_bytes:
        log(f"[SKIP] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return dest

    def _do():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        size = tmp.stat().st_size
        if size <= min_bytes:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded only {size} bytes (< {min_bytes} expected)")
        tmp.replace(dest)
        return dest

    retry(_do, what=what)
    log(f"[OK] downloaded {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def host_up(host, timeout=15):
    import requests
    try:
        requests.head(f"https://{host}/", timeout=timeout, allow_redirects=True)
        return True
    except Exception:
        try:
            requests.get(f"https://{host}/", timeout=timeout, stream=True)
            return True
        except Exception as e:  # noqa: BLE001
            warn(f"host {host} unreachable: {e}")
            return False


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    log(f"[OK] wrote {Path(path).name}")


def read_json(path, default=None):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return default
    with open(p) as f:
        return json.load(f)


def update_manifest(entry):
    """Append/replace a file entry in output/manifest.json keyed by 'name'."""
    mpath = OUTPUT / "manifest.json"
    man = read_json(mpath, default={"files": []})
    man["files"] = [e for e in man["files"] if e.get("name") != entry.get("name")]
    man["files"].append(entry)
    man["updated"] = _ts()
    write_json(mpath, man)


def manifest_entry_for(path, source_url):
    p = Path(path)
    return {
        "name": p.name,
        "path": str(p.relative_to(ROOT)) if ROOT in p.parents else str(p),
        "source_url": source_url,
        "size_bytes": p.stat().st_size if p.exists() else None,
        "timestamp": _ts(),
    }
