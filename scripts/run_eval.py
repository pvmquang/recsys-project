"""Chay danh gia so sanh cac mo hinh.

    python scripts/run_eval.py                      # ml-latest-small, B0
    python scripts/run_eval.py --dataset ml-1m      # ban cuoi
    python scripts/run_eval.py --seed 7             # doi lan chia de kiem do on dinh
    python scripts/run_eval.py --full-catalog       # xep hang tren toan danh muc (2.4)

Them mot bac mo hinh moi = them mot dong vao MODELS o duoi. Khong sua gi khac.
"""

from __future__ import annotations

import argparse

from recsys.data import movielens
from recsys.data.split import leave_one_out
from recsys.eval.provenance import save_result
from recsys.eval.runner import compare, format_table
from recsys.models.baselines import Popularity, RandomRec
from recsys.models.knn import ItemKNN, UserKNN

# Dang ky mo hinh o day. Tuan 3 tro di, moi bac moi them mot dong.
MODELS = [
    RandomRec(),
    Popularity(),
    ItemKNN(k=40),
    UserKNN(k=40),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-latest-small")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negatives", type=int, default=99)
    p.add_argument("--out", default=None, help="Ghi ket qua ra file CSV")
    p.add_argument(
        "--full-catalog",
        action="store_true",
        help="Xep hang item dung giua MOI item chua xem, khong lay mau 99 item am",
    )
    args = p.parse_args()

    print(f"Bo du lieu: {args.dataset}")
    ratings = movielens.load_ratings(args.dataset)
    print(f"  {movielens.describe(ratings)}")

    split = leave_one_out(ratings, n_negatives=args.negatives, seed=args.seed)
    print(f"  {split.summary()}\n")

    print("Danh gia" + (" (toan danh muc):" if args.full_catalog else ":"))
    df = compare(MODELS, split, full_catalog=args.full_catalog)

    print("\n" + "=" * 78)
    print(format_table(df))
    print("=" * 78)

    if args.out:
        out = save_result(
            df, args.out, seed=args.seed, dataset=args.dataset,
            negatives=args.negatives, full_catalog=args.full_catalog,
            models=[m.name for m in MODELS],
        )
        print(f"\nDa ghi: {out}")


if __name__ == "__main__":
    main()
