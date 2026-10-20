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
from recsys.data.split import Split, user_activity_groups
from recsys.eval import metrics as m

#: Cac gia tri K duoc bao cao. Chot o de xuat de tai.
DEFAULT_KS = (5, 10, 20)


def evaluate(
    model: Recommender,
    split: Split,
    ks: Sequence[int] = DEFAULT_KS,
    by_activity: bool = False,
) -> dict:
    """Danh gia mot mo hinh DA duoc fit.

    Voi moi user: cham diem 100 ung vien (1 dung + 99 am), xep hang, do.

    Tra ve dict phang, tien do thang vao DataFrame.
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

    n_items = len(model.all_items_)
    result["coverage"] = m.coverage(all_recommended, n_items)

    if by_activity:
        result["theo_nhom"] = _by_activity(df, split, ks)

    return result


def _by_activity(
    per_user: pd.DataFrame, split: Split, ks: Sequence[int]
) -> pd.DataFrame:
    """Tach ket qua theo nhom user thua/day -- phuc vu phan tich cold-start.

    Tra ve DataFrame (mot dong mot nhom), KHONG phai dict long nhau: dict
    khoa tuple cua pandas khong ghep thang vao bao cao duoc.

    CANH BAO VE DU LIEU -- doc truoc khi dien so nay vao bao cao:
    MovieLens chi nhan user da danh gia IT NHAT 20 phim. Do tren
    ml-latest-small: min = 20, khong co user nao duoi nguong do. Nghia la
    nhom "<5" LUON RONG, va bang nay KHONG tra loi duoc cau hoi cold-start
    that su ("mo hinh nao tot cho user moi").

    Muon do cold-start thi phai TAO RA no: cat bot lich su huan luyen cua
    mot phan user xuong con 1, 3, 5 tuong tac roi do lai. Bay bang nay ra
    va goi no la phan tich cold-start la ket luan ve mot nhom nguoi dung
    khong ton tai trong du lieu.
    """
    groups = user_activity_groups(split.train)
    df = per_user.assign(nhom=per_user["user_id"].map(groups))
    cols = [f"recall@{ks[1]}", f"ndcg@{ks[1]}", "mrr"]
    out = df.groupby("nhom")[cols].mean()
    out.insert(0, "so_user", df.groupby("nhom").size())
    return out.sort_values("so_user", ascending=False).round(4)


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
    cols += ["mrr", "auc", "coverage"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].round(4).to_string()
