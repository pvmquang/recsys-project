"""Moi con so slide can, kem khoang tin cay -- in ra Markdown.

    python scripts/so_lieu_slide.py > reports/so_lieu_slide.md

Slide va README lay so tu day, khong chep tu terminal cua lan chay khac. Moi
chenh lech deu kem CI 95% bootstrap ghep cap theo user (recsys/eval/stats.py):
CI chua 0 thi KHONG duoc viet "mo hinh X thang".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from recsys.data import movielens
from recsys.data.split import leave_one_out
from recsys.eval.runner import MAIN_K, evaluate
from recsys.eval.stats import bootstrap_ci, is_significant, paired_diff_ci
from recsys.models.baselines import Popularity, RandomRec
from recsys.models.knn import ItemKNN, UserKNN

K = MAIN_K
SEED = 42
RANDOM_SEEDS = range(10)


def fmt_ci(ci: tuple[float, float, float]) -> str:
    mean, lo, hi = ci
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def fmt_diff(ci: tuple[float, float, float]) -> str:
    mean, lo, hi = ci
    verdict = "**có ý nghĩa**" if is_significant(ci) else "nhiễu (CI chứa 0)"
    return f"{mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {verdict}"


def main() -> None:
    ratings = movielens.load_ratings()
    movies = movielens.load_movies()
    titles = dict(zip(movies["item_id"], movies["title"]))
    split = leave_one_out(ratings, seed=SEED)

    models = {
        "B0-Random": RandomRec(),
        "B0-Popularity": Popularity(),
        "B1-ItemKNN": ItemKNN(k=40, titles=titles),
        "B2-UserKNN": UserKNN(k=40),
    }
    res, pu = {}, {}
    for name, model in models.items():
        model.fit(split.train)
        r = evaluate(model, split, by_activity=True, keep_per_user=True)
        pu[name] = r.pop("per_user")
        res[name] = r

    print(f"# Số liệu cho slide\n")
    print(f"ml-latest-small · seed {SEED} · k láng giềng = 40 · "
          f"{len(split.users)} user kiểm tra · 1 phim đúng + 99 phim âm\n")

    # ---- Bang chinh -------------------------------------------------------
    print(f"## Bảng chính (CI 95% bootstrap theo user)\n")
    print(f"| Mô hình | recall@{K} | ndcg@{K} | MRR | AUC | coverage@{K} | coverage@20 |")
    print("|---|---|---|---|---|---|---|")
    for name, r in res.items():
        p = pu[name]
        print(
            f"| {name} | {fmt_ci(bootstrap_ci(p[f'recall@{K}']))} "
            f"| {fmt_ci(bootstrap_ci(p[f'ndcg@{K}']))} "
            f"| {r['mrr']:.4f} | {fmt_ci(bootstrap_ci(p['auc']))} "
            f"| {r[f'coverage@{K}']:.4f} | {r['coverage@20']:.4f} |"
        )

    # ---- Random nhieu seed -----------------------------------------------
    print(f"\n## B0-Random qua {len(RANDOM_SEEDS)} seed (kỳ vọng: recall@10 = 0.10, AUC = 0.50)\n")
    rr = []
    for s in RANDOM_SEEDS:
        r = evaluate(RandomRec(seed=s).fit(split.train), split)
        rr.append((r[f"recall@{K}"], r["auc"], r[f"coverage@{K}"]))
    rr = np.array(rr)
    print("| | trung bình | độ lệch chuẩn | nhỏ nhất | lớn nhất |")
    print("|---|---|---|---|---|")
    for i, col in enumerate((f"recall@{K}", "AUC", f"coverage@{K}")):
        x = rr[:, i]
        print(f"| {col} | {x.mean():.4f} | {x.std(ddof=1):.4f} | {x.min():.4f} | {x.max():.4f} |")

    # ---- Chenh lech ghep cap --------------------------------------------
    print("\n## Chênh lệch giữa các mô hình (ghép cặp theo user)\n")
    print("| So sánh | độ đo | chênh lệch | CI 95% | kết luận |")
    print("|---|---|---|---|---|")
    pairs = [
        ("B1-ItemKNN", "B0-Popularity"),
        ("B2-UserKNN", "B0-Popularity"),
        ("B1-ItemKNN", "B2-UserKNN"),
    ]
    for metric in (f"ndcg@{K}", f"recall@{K}", "auc"):
        for a, b in pairs:
            ci = paired_diff_ci(pu[a][metric], pu[b][metric])
            print(f"| {a} − {b} | {metric} | {fmt_diff(ci)} |")

    # ---- Theo nhom --------------------------------------------------------
    print(f"\n## Theo nhóm user (phân vị số tương tác trong train) — ndcg@{K}\n")
    print("⚠ Nhóm thưa nhất vẫn có ≥ 19 tương tác: đây **không** phải phân tích cold-start.\n")
    groups = res["B0-Popularity"]["theo_nhom"]
    names = ["B0-Popularity", "B1-ItemKNN", "B2-UserKNN"]
    print("| Nhóm (số tương tác) | số user | " + " | ".join(names) + " | ItemKNN − UserKNN, CI 95% |")
    print("|---|---|" + "---|" * len(names) + "---|")
    counts = split.train.groupby("user_id").size()
    for g in groups.index:
        lo, hi = (int(x) for x in g.split("-"))
        users = [u for u in split.users if lo <= counts[u] <= hi]
        cells = " | ".join(f"{res[n]['theo_nhom'].loc[g, f'ndcg@{K}']:.4f}" for n in names)
        ci = paired_diff_ci(pu["B1-ItemKNN"].loc[users, f"ndcg@{K}"],
                            pu["B2-UserKNN"].loc[users, f"ndcg@{K}"])
        print(f"| {g} | {groups.loc[g, 'so_user']} | {cells} | {fmt_diff(ci).replace(' | ', ' ')} |")

    # ---- Quet k: 40 vs 80 -------------------------------------------------
    print(f"\n## Số láng giềng k = 40 so với k = 80 (ItemKNN, ndcg@{K})\n")
    k80 = evaluate(ItemKNN(k=80).fit(split.train), split, keep_per_user=True)["per_user"]
    ci = paired_diff_ci(k80[f"ndcg@{K}"], pu["B1-ItemKNN"][f"ndcg@{K}"])
    print("| So sánh | chênh lệch | CI 95% | kết luận |\n|---|---|---|---|")
    print(f"| k=80 − k=40 | {fmt_diff(ci)} |")

    # ---- Vi du explain ----------------------------------------------------
    print("\n## Ví dụ câu giải thích thật (cho slide 10)\n")
    item_knn, user_knn = models["B1-ItemKNN"], models["B2-UserKNN"]
    shown = 0
    for u in split.users:
        rec = item_knn.recommend(u, k=1)
        if not rec:
            continue
        txt = item_knn.explain(u, rec[0])
        if not txt:
            continue
        print(f"- **User {u}** → *{titles.get(rec[0], rec[0])}*")
        print(f"  - ItemKNN: \"{txt}\"")
        txt2 = user_knn.explain(u, rec[0])
        if txt2:
            print(f"  - UserKNN: \"{txt2}\"")
        print(f"  - Popularity: \"{models['B0-Popularity'].explain(u, rec[0])}\"")
        shown += 1
        if shown == 3:
            break

    # ---- Han che giao thuc -----------------------------------------------
    last = ratings.groupby("user_id")["timestamp"].transform("max")
    tie = ratings[ratings["timestamp"] == last].groupby("user_id").size()
    train_items = set(split.train["item_id"])
    unseen = sum(1 for i in split.test.values() if i not in train_items)
    print("\n## Hạn chế của giao thức (cho slide 4 và 8)\n")
    print(f"- {int((tie > 1).sum())}/{len(tie)} user có nhiều phim trùng timestamp cuối "
          f"(nhiều nhất {int(tie.max())} phim cùng một giây) → phim test do phá hòa theo item_id.")
    print(f"- {unseen}/{len(split.test)} phim test chưa từng xuất hiện trong train.")
    print("- Train vẫn chứa tương tác về sau của user KHÁC (ví dụ số đếm của Popularity).")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
