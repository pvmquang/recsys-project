"""Chay danh gia mot hay nhieu mo hinh tren CUNG mot lan chia du lieu.

Day la thu phai xong TRUOC moi mo hinh (tuan 1-2 cua ke hoach). Co no roi
thi moi bac B1-B8 viet xong la do duoc ngay trong ngay. Lam nguoc lai --
viet het mo hinh roi moi danh gia -- thi tuan 11 phai sua lai tat ca.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from recsys.base import Recommender
from recsys.data.split import Split
from recsys.eval import metrics as m

#: Cac gia tri K duoc bao cao. Chot o de xuat de tai.
DEFAULT_KS = (5, 10, 20)

#: K cua bang chinh va cua phan tich theo nhom. Giao dien hien 10 phim.
MAIN_K = 10


def evaluate(
    model: Recommender,
    split: Split,
    ks: Sequence[int] = DEFAULT_KS,
    by_activity: bool = False,
    keep_per_user: bool = False,
) -> dict:
    """Danh gia mot mo hinh DA duoc fit.

    Voi moi user: cham diem 100 ung vien (1 dung + 99 am), xep hang, do.

    Tra ve dict phang, tien do thang vao DataFrame.

    keep_per_user: them khoa "per_user" (DataFrame, chi so user_id) -- can cho
        khoang tin cay, xem recsys/eval/stats.py.
    """
    per_user: list[dict] = []
    all_recommended: list[list[int]] = []
    t0 = time.perf_counter()

    for user_id in split.users:
        candidates = split.candidates(user_id)
        scores = model.score(user_id, candidates)

        # Xao truoc khi xep hang. Khong xao thi moi cap diem BANG NHAU se
        # duoc gioi quyet theo thu tu mang -- ma item dung luon o vi tri 0
        # (xem Split.candidates). Mot mo hinh cham diem bang nhau cho tat ca
        # (vi du: user moi, chua biet gi) se dat nDCG = 1.0 tuyet doi.
        order = np.random.default_rng(user_id).permutation(len(candidates))
        candidates, scores = candidates[order], scores[order]

        ranked = candidates[np.argsort(-scores, kind="stable")].tolist()
        relevant = {int(split.test[user_id])}

        row: dict = {"user_id": user_id, "auc": m.auc(ranked, relevant)}
        for k in ks:
            row[f"precision@{k}"] = m.precision_at_k(ranked, relevant, k)
            row[f"recall@{k}"] = m.recall_at_k(ranked, relevant, k)
            row[f"ndcg@{k}"] = m.ndcg_at_k(ranked, relevant, k)
            row[f"map@{k}"] = m.average_precision_at_k(ranked, relevant, k)
        row["mrr"] = m.reciprocal_rank(ranked, relevant)
        per_user.append(row)

        all_recommended.append(model.recommend(user_id, k=max(ks)))

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(per_user)

    result: dict = {"model": model.name, "n_users": len(df), "seconds": round(elapsed, 2)}
    for col in df.columns:
        if col != "user_id":
            result[col] = float(df[col].mean())

    # Coverage@k cho TUNG k. Mot cot "coverage" khong ghi k thi nguoi doc se
    # ghep no voi cot @10 ben canh, trong khi no la @max(ks) -- da xay ra o ban
    # truoc. Lay tien to cua top-max(ks) la dung: recommend() da xep giam dan.
    n_items = len(model.all_items_)
    for k in ks:
        result[f"coverage@{k}"] = m.coverage((r[:k] for r in all_recommended), n_items)

    if by_activity:
        result["theo_nhom"] = _by_activity(df, split)
    if keep_per_user:
        result["per_user"] = df.set_index("user_id")

    return result


def _by_activity(
    per_user: pd.DataFrame, split: Split, k: int = MAIN_K, n_groups: int = 3
) -> pd.DataFrame:
    """Tach ket qua theo nhom user, chia theo PHAN VI so tuong tac trong train.

    Tra ve DataFrame (mot dong mot nhom), KHONG phai dict long nhau: dict
    khoa tuple cua pandas khong ghep thang vao bao cao duoc.

    VI SAO CHIA THEO PHAN VI, KHONG THEO NGUONG CO DINH (5, 20, 100)
    MovieLens chi nhan user da danh gia IT NHAT 20 phim, nen sau khi tach 1 phim
    lam test, train con it nhat 19. Voi nguong co dinh, nhom "5-20" thuc chat la
    29 user co 19-20 tuong tac -- qua it de ket luan (chenh nDCG UserKNN - ItemKNN
    trong nhom do: +0.028, CI 95% [-0.090, +0.159]) va bi doc nham thanh "user it
    lich su". Chia phan vi thi nhom nao cung ~200 user, va nhan ghi DUNG khoang
    so tuong tac that.

    CANH BAO: bang nay KHONG phai phan tich cold-start. Nhom thua nhat van co
    >= 19 tuong tac. Muon do cold-start thi phai TAO RA no: cat bot lich su huan
    luyen cua mot phan user xuong con 1, 3, 5 tuong tac roi do lai.
    """
    counts = split.train.groupby("user_id").size()
    counts = counts[counts.index.isin(per_user["user_id"])]
    bins = pd.qcut(counts, q=n_groups, duplicates="drop")
    # Nhan bang khoang THAT trong du lieu ("19-45"), khong phai khoang qcut
    # ("(18.999, 45.0]") -- de ghi thang vao bang bao cao.
    span = counts.groupby(bins, observed=True).agg(["min", "max"])
    label = {iv: f"{int(r['min'])}-{int(r['max'])}" for iv, r in span.iterrows()}
    groups = bins.map(label)

    df = per_user.assign(nhom=per_user["user_id"].map(groups))
    cols = [f"recall@{k}", f"ndcg@{k}", "mrr"]
    out = df.groupby("nhom", sort=False)[cols].mean()
    out.insert(0, "so_user", df.groupby("nhom", sort=False).size())
    # Xep theo so tuong tac tang dan, khong theo thu tu chu cai ("102-..." < "19-...").
    order = sorted(out.index, key=lambda s: int(s.split("-")[0]))
    return out.loc[order].round(4)


def compare(
    models: Sequence[Recommender],
    split: Split,
    ks: Sequence[int] = DEFAULT_KS,
) -> pd.DataFrame:
    """Fit va danh gia nhieu mo hinh, tra ve bang so sanh.

    MOI mo hinh nhan CUNG mot `split`. Day la diem mau chot cua ca do an:
    hai mo hinh do tren hai lan chia khac nhau thi khong so sanh duoc, va
    khong co gi tren bang ket qua noi ra dieu do.
    """
    rows = []
    for model in models:
        print(f"  {model.name:<28}", end="", flush=True)
        model.fit(split.train)
        res = evaluate(model, split, ks)
        res.pop("theo_nhom", None)
        rows.append(res)
        print(f" ndcg@10={res['ndcg@10']:.4f}  ({res['seconds']}s)")

    return pd.DataFrame(rows).set_index("model")


def format_table(df: pd.DataFrame, ks: Sequence[int] = DEFAULT_KS) -> str:
    """In bang ket qua gon cho terminal va cho bao cao."""
    cols = ["n_users"]
    for k in ks:
        cols += [f"recall@{k}", f"ndcg@{k}"]
    cols += ["mrr", "auc", f"coverage@{MAIN_K}"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].round(4).to_string()
