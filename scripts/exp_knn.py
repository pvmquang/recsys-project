"""Hai thi nghiem cua bac B1/B2 -- chay lai duoc, khong phai so go tay.

    python scripts/exp_knn.py

1. THEO NHOM USER  -- kiem gia thuyet cua doc B0: neu loc cong tac thang, no
   phai thang DAM NHAT o nhom user nhieu lich su, dung cho Popularity yeu nhat.
2. HAI CHE DO CHAM DIEM -- "sum" (xep hang) vs "weighted" (du doan rating).
   Bao cao phai do ca hai roi giai thich vi sao chung khac nhau.
3. QUET k -- so lang gieng. Slide L2 tr.22 noi thuc nghiem tren MovieLens goi y
   20-50; day la cho kiem lai con so do bang chinh du lieu cua nhom.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from recsys.data import movielens
from recsys.data.split import leave_one_out
from recsys.eval.runner import evaluate
from recsys.models.baselines import Popularity
from recsys.models.knn import ItemKNN, UserKNN


def main() -> None:
    ratings = movielens.load_ratings()
    split = leave_one_out(ratings)
    print(movielens.describe(ratings))
    print(split.summary())

    print("\n" + "=" * 78)
    print("1. THEO NHOM USER (so tuong tac trong tap huan luyen)")
    print("=" * 78)
    for model in (Popularity(), ItemKNN(k=40), UserKNN(k=40)):
        model.fit(split.train)
        res = evaluate(model, split, by_activity=True)
        print(f"\n--- {model.name} ---")
        print(res["theo_nhom"].to_string())

    print("\n" + "=" * 78)
    print("2. HAI CHE DO CHAM DIEM")
    print("=" * 78)
    rows = []
    for cls in (ItemKNN, UserKNN):
        for mode in ("sum", "weighted"):
            m = cls(k=40, score_mode=mode)
            m.name = f"{cls.name}-{mode}"
            m.fit(split.train)
            r = evaluate(m, split)
            rows.append(
                {
                    "mo_hinh": m.name,
                    "recall@10": r["recall@10"],
                    "ndcg@10": r["ndcg@10"],
                    "mrr": r["mrr"],
                    "auc": r["auc"],
                    "coverage": r["coverage"],
                }
            )
    print(pd.DataFrame(rows).set_index("mo_hinh").round(4).to_string())

    print("\n" + "=" * 78)
    print("3. QUET SO LANG GIENG k")
    print("=" * 78)
    rows = []
    for k in (5, 10, 20, 40, 80, 160):
        for cls in (ItemKNN, UserKNN):
            m = cls(k=k)
            m.fit(split.train)
            r = evaluate(m, split)
            rows.append(
                {
                    "k": k,
                    "mo_hinh": cls.name,
                    "ndcg@10": r["ndcg@10"],
                    "recall@10": r["recall@10"],
                    "coverage": r["coverage"],
                }
            )
    df = pd.DataFrame(rows).pivot(index="k", columns="mo_hinh")
    print(df.round(4).to_string())

    out = Path("reports/exp_knn.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nDa ghi: {out}")


if __name__ == "__main__":
    main()
