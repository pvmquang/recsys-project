"""Ba thi nghiem cua bac B1/B2 -- chay lai duoc, khong phai so go tay.

    python scripts/exp_knn.py

1. THEO NHOM USER  -- kiem gia thuyet cua doc B0: neu loc cong tac thang, no
   phai thang DAM NHAT o nhom user nhieu lich su, dung cho Popularity yeu nhat.
   Nhom chia theo PHAN VI (xem runner._by_activity) -- KHONG phai cold-start.
2. HAI CHE DO CHAM DIEM -- "sum" (xep hang) vs "weighted" (du doan rating).
   Bao cao phai do ca hai roi giai thich vi sao chung khac nhau.
3. QUET k -- so lang gieng. Slide L2 tr.22 noi thuc nghiem tren MovieLens goi y
   20-50; day la cho kiem lai con so do bang chinh du lieu cua nhom.
   ⚠ Hien dang chon k ngay tren TAP KIEM TRA -- chua co tap validation
   (checklist 2.1). Doc ket qua nhu mot duong cong, khong phai mot lua chon.

Moi thi nghiem ghi mot file CSV rieng trong reports/ -- so trong README va slide
phai lay tu do, khong chep tu terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from recsys.data import movielens
from recsys.data.split import leave_one_out
from recsys.eval.runner import MAIN_K, evaluate
from recsys.models.baselines import Popularity
from recsys.models.knn import ItemKNN, UserKNN

OUT_DIR = Path("reports")


def _save(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path)
    print(f"Da ghi: {path}")


def main() -> None:
    ratings = movielens.load_ratings()
    split = leave_one_out(ratings)
    print(movielens.describe(ratings))
    print(split.summary())

    print("\n" + "=" * 78)
    print(f"1. THEO NHOM USER (phan vi so tuong tac trong train) -- ndcg@{MAIN_K}")
    print("=" * 78)
    parts = []
    for model in (Popularity(), ItemKNN(k=40), UserKNN(k=40)):
        model.fit(split.train)
        bang = evaluate(model, split, by_activity=True)["theo_nhom"]
        parts.append(bang.assign(mo_hinh=model.name).set_index("mo_hinh", append=True))
    nhom = pd.concat(parts)
    print(nhom[f"ndcg@{MAIN_K}"].unstack("mo_hinh").join(
        nhom["so_user"].groupby(level="nhom").first()
    ).to_string())
    _save(nhom, "exp_knn_1_theo_nhom.csv")

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
                    f"recall@{MAIN_K}": r[f"recall@{MAIN_K}"],
                    f"ndcg@{MAIN_K}": r[f"ndcg@{MAIN_K}"],
                    "mrr": r["mrr"],
                    "auc": r["auc"],
                    f"coverage@{MAIN_K}": r[f"coverage@{MAIN_K}"],
                }
            )
    che_do = pd.DataFrame(rows).set_index("mo_hinh").round(4)
    print(che_do.to_string())
    _save(che_do, "exp_knn_2_che_do.csv")

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
                    f"ndcg@{MAIN_K}": r[f"ndcg@{MAIN_K}"],
                    f"recall@{MAIN_K}": r[f"recall@{MAIN_K}"],
                    f"coverage@{MAIN_K}": r[f"coverage@{MAIN_K}"],
                }
            )
    quet = pd.DataFrame(rows).set_index(["k", "mo_hinh"])
    print(quet.unstack("mo_hinh").round(4).to_string())
    _save(quet, "exp_knn_3_quet_k.csv")


if __name__ == "__main__":
    main()
