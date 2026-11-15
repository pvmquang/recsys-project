"""Ghi ket qua KEM nguon goc: seed, bo du lieu, git commit, phien ban thu vien.

Checklist 3.5. Mot file CSV trong reports/ khong noi no duoc chay voi seed nao, tu
ban code nao. Ba thanh vien gui nhau ba file CSV lech nhau 0.01 thi khong ai biet
la do seed, do code, hay do phien ban pandas -- va khong co gi tren bang so noi ra.

Moi file `x.csv` co mot file `x.meta.json` di kem. Tach rieng (khong nhet vao
CSV) de `pd.read_csv` van doc thang duoc nhu cu.

`git_dirty = true` nghia la code co sua CHUA commit luc chay: so do KHONG tai tao
duoc tu commit ghi trong file. Truoc khi dua so len slide, commit roi chay lai.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_LIBS = ("numpy", "pandas", "scipy", "scikit-learn")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def run_info(**extra) -> dict:
    """Thong tin de tai tao mot lan chay. `extra`: seed, dataset, k, ..."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    versions = {}
    for lib in _LIBS:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    return {
        **extra,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "lenh": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "thoi_gian": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "thu_vien": versions,
    }


def save_result(df: pd.DataFrame, path: str | Path, **info) -> Path:
    """Ghi `df` ra CSV va `<ten>.meta.json` ben canh. Tra ve duong dan CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    meta = path.with_suffix(".meta.json")
    meta.write_text(
        json.dumps(run_info(**info), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
