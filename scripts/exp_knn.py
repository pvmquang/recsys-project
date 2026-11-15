"""Ba thi nghiem cua bac B1/B2 -- chay lai duoc, khong phai so go tay.

    python scripts/exp_knn.py

1. THEO NHOM USER  -- kiem gia thuyet cua doc B0: neu loc cong tac thang, no
   phai thang DAM NHAT o nhom user nhieu lich su, dung cho Popularity yeu nhat.
   Nhom chia theo PHAN VI (xem runner._by_activity) -- KHONG phai cold-start.
2. HAI CHE DO CHAM DIEM -- "sum" (xep hang) vs "weighted" (du doan rating).
   Bao cao phai do ca hai roi giai thich vi sao chung khac nhau.
3. QUET k -- so lang gieng. Slide L2 tr.22 noi thuc nghiem tren MovieLens goi y
   20-50; day la cho kiem lai con so do bang chinh du lieu cua nhom.
   CHON k TREN VALIDATION (tuong tac ap chot, checklist 2.1), BAO CAO tren test.
   k chon duoc phai khop knn.K_ITEM / knn.K_USER -- lech thi script canh bao.

Moi thi nghiem ghi mot file CSV rieng trong reports/ -- so trong README va slide
phai lay tu do, khong chep tu terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from recsys.data import movielens
from recsys.data.split import leave_one_out
from recsys.eval.provenance import save_result
from recsys.eval.runner import MAIN_K, evaluate
from recsys.models.baselines import Popularity
from recsys.models.knn import K_ITEM, K_USER, ItemKNN, UserKNN

OUT_DIR = Path("reports")
SEED = 42


def _save(df: pd.DataFrame, name: str) -> None:
    path = save_result(df, OUT_DIR / name, seed=SEED, dataset="ml-latest-small",
                       k_item=K_ITEM, k_user=K_USER)
    print(f"Da ghi: {path}")


def main() -> None:
    ratings = movielens.load_ratings()
    split = leave_one_out(ratings, seed=SEED)
    print(movielens.describe(ratings))
    print(split.summary())

    print("\n" + "=" * 78)
    print(f"1. THEO NHOM USER (phan vi so tuong tac trong train) -- ndcg@{MAIN_K}")
    print("=" * 78)
    parts = []
    for model in (Popularity(), ItemKNN(k=K_ITEM), UserKNN(k=K_USER)):
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
    for cls, k_chon in ((ItemKNN, K_ITEM), (UserKNN, K_USER)):
        for mode in ("sum", "weighted"):
            m = cls(k=k_chon, score_mode=mode)
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
    print("3. QUET SO LANG GIENG k -- chon tren VALIDATION, bao cao tren TEST")
    print("=" * 78)
    val = split.validation()
    rows = []
    for k in (5, 10, 20, 40, 80, 160):
        for cls in (ItemKNN, UserKNN):
            rv = evaluate(cls(k=k).fit(val.train), val)
            rt = evaluate(cls(k=k).fit(split.train), split)
            rows.append(
                {
                    "k": k,
                    "mo_hinh": cls.name,
                    f"ndcg@{MAIN_K}_val": rv[f"ndcg@{MAIN_K}"],
                    f"ndcg@{MAIN_K}_test": rt[f"ndcg@{MAIN_K}"],
                    f"recall@{MAIN_K}_test": rt[f"recall@{MAIN_K}"],
                    f"coverage@{MAIN_K}_test": rt[f"coverage@{MAIN_K}"],
                }
            )
    quet = pd.DataFrame(rows).set_index(["k", "mo_hinh"])
    print(quet.unstack("mo_hinh").round(4).to_string())
    _save(quet, "exp_knn_3_quet_k.csv")

    # Chon k = argmax nDCG tren VALIDATION. Khong nhin cot test khi chon.
    print()
    da_chot = {ItemKNN.name: K_ITEM, UserKNN.name: K_USER}
    for name, g in quet.groupby(level="mo_hinh"):
        k_best = int(g[f"ndcg@{MAIN_K}_val"].idxmax()[0])
        test = g.loc[(k_best, name), f"ndcg@{MAIN_K}_test"]
        print(f"{name}: validation chon k={k_best}, ndcg@{MAIN_K} test = {test:.4f}")
        if k_best != da_chot[name]:
            print(
                f"  CANH BAO: knn.py dang ghi k={da_chot[name]}. Sua K_ITEM/K_USER "
                "trong recsys/models/knn.py roi chay lai moi script.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
