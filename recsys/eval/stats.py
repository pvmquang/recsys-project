"""Khoang tin cay bootstrap THEO USER cho cac do do.

VI SAO CAN
ml-latest-small chi co 610 user kiem tra. Mot chenh lech 0.01 tren bang ket qua
co the chi la nhieu, va bang trung binh khong noi ra dieu do. Do that (seed 42):

    AUC UserKNN - Popularity  = +0.0093, CI 95% [-0.0056, +0.0244]  -> nhieu
    nDCG@10 ItemKNN - UserKNN = +0.0506, CI 95% [+0.0293, +0.0716]  -> dung duoc

Cau "UserKNN thang AUC" vi vay khong co can cu; cau "ItemKNN thang nDCG" thi co.

VI SAO BOOTSTRAP CAP (paired) CHU KHONG SO HAI CI RIENG
Hai mo hinh duoc do tren CUNG 610 user, va user kho thi kho voi moi mo hinh.
Lay mau lai CUNG mot tap user cho ca hai roi lay hieu thi phan "user nay kho"
triet tieu, CI hep hon dang ke. Hai CI rieng chong nhau KHONG co nghia la
chenh lech la nhieu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(trung binh, can duoi, can tren) cua trung binh `values`."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        raise ValueError("Khong co gia tri nao de lay mau lai")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(
    a: pd.Series,
    b: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI cua (a - b), a va b la do do THEO USER, danh chi so bang user_id.

    Ghep theo user_id chu khong theo vi tri: hai lan danh gia co the duyet user
    theo thu tu khac nhau, va ghep theo vi tri se tru diem cua hai NGUOI KHAC NHAU
    ma khong nem loi.
    """
    if not a.index.equals(b.index) and set(a.index) != set(b.index):
        raise ValueError("Hai mo hinh phai duoc do tren CUNG mot tap user")
    d = a - b.reindex(a.index)
    return bootstrap_ci(d.to_numpy(), n_boot=n_boot, alpha=alpha, seed=seed)


def is_significant(ci: tuple[float, float, float]) -> bool:
    """CI khong chua 0."""
    _, lo, hi = ci
    return lo > 0 or hi < 0
