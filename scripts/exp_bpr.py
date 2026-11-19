"""Thi nghiem cua bac B4 (BPR-MF) -- chon sieu tham so tren VALIDATION, bao cao tren TEST.

    python scripts/exp_bpr.py > docs/so-lieu-b4.md     # ~20-30 phut

Cau hoi cua bac B4: hoc theo CAP (BPR) co hon hoc tung diem (MF implicit, B3)
khong, va tra gia bang gi.

1. CHON SIEU THAM SO tren validation (checklist 2.1), tim theo tang:
     f = 64 co dinh, do lr x reg; moi cau hinh danh gia sau 10, 20, 40, 80, 160
     epoch (train_more, khong huan luyen lai); roi do f o cap tot nhat.
   Ket qua phai khop hang so trong recsys/models/bpr.py -- lech thi canh bao.
2. BANG CHINH tren test: BPR canh B0, B1, B3, kem CI 95% va chenh lech ghep cap,
   ca lay mau lan toan danh muc. AUC so duoc giua BPR va MF vi ca hai cham duoc
   gan het ung vien (khac KNN, xem so-lieu-dot-2.md muc 2.5).
3. THIEN LECH PHO BIEN: top-10 cua moi mo hinh phu bao nhieu danh muc, va phim
   trong do pho bien den dau. Bias b_i cua BPR co phai chi la do pho bien khong.
4. COLD-START (nhu checklist 2.8, cung cach chia 4 phan cua exp_giao_thuc.py).
5. VI DU explain() that, BPR canh MF.

In Markdown ra stdout, tien do ra stderr. Moi bang ghi mot CSV kem .meta.json.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from recsys.data import movielens
from recsys.data.split import leave_one_out, truncate_history
from recsys.eval.provenance import save_result
from recsys.eval.runner import MAIN_K, evaluate
from recsys.eval.stats import bootstrap_ci, is_significant, paired_diff_ci
from recsys.models import bpr
from recsys.models.baselines import Popularity
from recsys.models.knn import K_ITEM, ItemKNN
from recsys.models.mf import MF

K = MAIN_K
SEED = 42
OUT_DIR = Path("reports")
N_KEEP = (1, 3, 5, 10)
N_FOLDS = 4
CHECKPOINTS = (10, 20, 40, 80, 160)

POP, IKNN, MFI, BPR = "B0-Popularity", "B1-ItemKNN", "B3-MF-implicit", "B4-BPR"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fmt_ci(ci) -> str:
    mean, lo, hi = ci
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def fmt_diff(ci) -> str:
    mean, lo, hi = ci
    verdict = "**có ý nghĩa**" if is_significant(ci) else "nhiễu"
    return f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict}"


def _save(df: pd.DataFrame, name: str, **info) -> None:
    save_result(df, OUT_DIR / name, seed=SEED, dataset="ml-latest-small", **info)


# ------------------------------------------------------------------ 1. chon --


def tim_sieu_tham_so(val) -> dict:
    print("\n## 1. Chọn siêu tham số trên validation (ndcg@10)\n")
    rows = []

    def thu(f, lr, reg):
        t = time.time()
        m = bpr.BPR(factors=f, lr=lr, reg=reg, n_epochs=0).fit(val.train)
        done = 0
        for ep in CHECKPOINTS:
            m.train_more(ep - done)
            done = ep
            r = evaluate(m, val)
            rows.append({"factors": f, "lr": lr, "reg": reg, "n_epochs": ep,
                         f"ndcg@{K}": r[f"ndcg@{K}"], f"recall@{K}": r[f"recall@{K}"],
                         "loss": m.loss_[-1]})
            log(f"  f={f:<4} lr={lr:<5} reg={reg:<6} ep={ep:<4} "
                f"ndcg={r[f'ndcg@{K}']:.4f}  ({time.time() - t:.0f}s)")

    def tot_nhat(**loc):
        g = pd.DataFrame(rows)
        for k, v in loc.items():
            g = g[g[k] == v]
        return g.loc[g[f"ndcg@{K}"].idxmax()]

    for lr in (0.05, 0.1):
        for reg in (0.001, 0.01, 0.03):
            thu(64, lr, reg)
    b = tot_nhat(factors=64)
    for f in (32, 128):
        thu(f, b["lr"], b["reg"])
    best = tot_nhat()

    grid = pd.DataFrame(rows).set_index(["factors", "lr", "reg", "n_epochs"])
    _save(grid, "exp_bpr_1_validation.csv")
    print(f"Mỗi cấu hình huấn luyện một lần, đánh giá sau {', '.join(map(str, CHECKPOINTS))} "
          "epoch. Ô đậm là tốt nhất của từng cấu hình.\n")
    print("| factors | lr | reg | " + " | ".join(f"{e} epoch" for e in CHECKPOINTS) + " |")
    print("|---|---|---|" + "---|" * len(CHECKPOINTS))
    for (f, lr, reg), g in grid.groupby(level=[0, 1, 2], sort=False):
        nd = g[f"ndcg@{K}"].to_numpy()
        cells = [f"**{v:.4f}**" if v == nd.max() else f"{v:.4f}" for v in nd]
        print(f"| {f} | {lr:g} | {reg:g} | " + " | ".join(cells) + " |")

    chon = {"factors": int(best["factors"]), "lr": float(best["lr"]),
            "reg": float(best["reg"]), "n_epochs": int(best["n_epochs"])}
    print(f"\nChọn: {chon}, ndcg@10 validation {best[f'ndcg@{K}']:.4f}.")

    da_ghi = {"factors": bpr.FACTORS, "lr": bpr.LR, "reg": bpr.REG, "n_epochs": bpr.N_EPOCHS}
    if chon != da_ghi:
        log(f"CANH BAO: bpr.py dang ghi {da_ghi}. Sua hang so trong recsys/models/bpr.py.")
    return chon


def make_models(chon, titles=None) -> dict:
    mi = MF(titles=titles)
    mi.name = MFI
    return {POP: Popularity(), IKNN: ItemKNN(k=K_ITEM, titles=titles), MFI: mi,
            BPR: bpr.BPR(titles=titles, **chon)}


# ------------------------------------------------------------------ 2. test --


def bang_chinh(split, models) -> tuple[dict, dict]:
    print("\n## 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)\n")
    pu, pu_full, rows = {}, {}, []
    for name, m in models.items():
        log(f"  fit {name}")
        m.fit(split.train)
        r = evaluate(m, split, keep_per_user=True)
        pu[name] = r.pop("per_user")
        rf = evaluate(m, split, keep_per_user=True, full_catalog=True)
        pu_full[name] = rf.pop("per_user")
        rows.append({"mo_hinh": name, **{c: r[c] for c in (
            f"recall@{K}", f"ndcg@{K}", "mrr", "auc", "vo_cuc", f"coverage@{K}")},
            f"ndcg@{K}_toan_danh_muc": rf[f"ndcg@{K}"],
            f"recall@{K}_toan_danh_muc": rf[f"recall@{K}"]})
    _save(pd.DataFrame(rows).set_index("mo_hinh").round(4), "exp_bpr_2_test.csv")

    print(f"| Mô hình | recall@{K} | ndcg@{K} | AUC | coverage@{K} | ứng viên −inf |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        p = pu[r["mo_hinh"]]
        print(f"| {r['mo_hinh']} | {fmt_ci(bootstrap_ci(p[f'recall@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(p[f'ndcg@{K}']))} | {fmt_ci(bootstrap_ci(p['auc']))} "
              f"| {r[f'coverage@{K}']:.4f} | {r['vo_cuc']:.1%} |")

    print("\n### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục\n")
    print("| So sánh | độ đo | lấy mẫu | toàn danh mục |")
    print("|---|---|---|---|")
    pairs = [(BPR, MFI), (BPR, IKNN), (BPR, POP)]
    for metric in (f"ndcg@{K}", f"recall@{K}", "auc"):
        for a, b in pairs:
            c1 = fmt_diff(paired_diff_ci(pu[a][metric], pu[b][metric]))
            c2 = fmt_diff(paired_diff_ci(pu_full[a][metric], pu_full[b][metric]))
            print(f"| {a} − {b} | {metric} | {c1} | {c2} |")

    print(f"\nToàn danh mục, ndcg@{K}: " + ", ".join(
        f"{n} {fmt_ci(bootstrap_ci(pu_full[n][f'ndcg@{K}']))}" for n in models))
    return pu, pu_full


# ---------------------------------------------------------------- 3. pho bien --


def pho_bien(split, models) -> None:
    print("\n## 3. Thiên lệch phổ biến của top-10 (toàn danh mục, mọi user)\n")
    dem = split.train["item_id"].value_counts()
    top100 = set(dem.index[:100])
    n_items = split.train["item_id"].nunique()
    rows = []
    for name, m in models.items():
        recs = [m.recommend(u, k=K) for u in split.users]
        phim = [i for r in recs for i in r]
        rows.append({"mo_hinh": name,
                     "so_phim_khac_nhau": len(set(phim)),
                     "phu_danh_muc": len(set(phim)) / n_items,
                     "luot_xem_tb": float(np.mean([dem.get(i, 0) for i in phim])),
                     "ty_le_top100": float(np.mean([i in top100 for i in phim]))})
    df = pd.DataFrame(rows).set_index("mo_hinh")
    _save(df.round(4), "exp_bpr_3_pho_bien.csv")
    print("| Mô hình | số phim khác nhau trong top-10 | phủ danh mục "
          "| lượt xem TB của phim gợi ý | tỉ lệ thuộc top-100 phổ biến |")
    print("|---|---|---|---|---|")
    for name, r in df.iterrows():
        print(f"| {name} | {r['so_phim_khac_nhau']:.0f} | {r['phu_danh_muc']:.1%} "
              f"| {r['luot_xem_tb']:.1f} | {r['ty_le_top100']:.1%} |")

    m = models[BPR]
    counts = dem.reindex(m.items_).to_numpy()
    rho = spearmanr(m.b_, counts).statistic
    print(f"\nSpearman giữa bias b_i của BPR và số lượt xem trong train: {rho:.3f} "
          f"({len(m.items_)} phim).")


# --------------------------------------------------------------- 4. cold --


def cold_start(split, chon, pu_full_hist) -> None:
    print("\n## 4. Cold-start nhân tạo (cắt còn n tương tác gần nhất)\n")
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(np.array(split.users)), N_FOLDS)
    names = [POP, IKNN, MFI, BPR]
    pu: dict[tuple[str, str], pd.Series] = {}
    rows = []
    for n in N_KEEP:
        parts: dict[str, list] = {nm: [] for nm in names}
        for fold in folds:
            cut = truncate_history(split, fold, n)
            cut = replace(cut, test={int(u): split.test[int(u)] for u in fold})
            fold_models = {POP: Popularity(), IKNN: ItemKNN(k=K_ITEM), MFI: MF(),
                           BPR: bpr.BPR(**chon)}
            for nm, m in fold_models.items():
                m.fit(cut.train)
                parts[nm].append(evaluate(m, cut, keep_per_user=True)["per_user"])
        for nm in names:
            df = pd.concat(parts[nm])
            pu[nm, str(n)] = df[f"ndcg@{K}"]
            rows.append({"n_giu": n, "mo_hinh": nm, f"ndcg@{K}": df[f"ndcg@{K}"].mean(),
                         "vo_cuc": df["vo_cuc"].mean()})
        log(f"  cold-start n = {n}: xong")
    for nm in names:
        pu[nm, "day_du"] = pu_full_hist[nm][f"ndcg@{K}"]
        rows.append({"n_giu": "day_du", "mo_hinh": nm,
                     f"ndcg@{K}": pu_full_hist[nm][f"ndcg@{K}"].mean(),
                     "vo_cuc": pu_full_hist[nm]["vo_cuc"].mean()})
    _save(pd.DataFrame(rows).set_index(["n_giu", "mo_hinh"]).round(4),
          "exp_bpr_4_cold_start.csv", n_folds=N_FOLDS)

    print(f"ndcg@{K}, CI 95%, đủ {len(split.users)} user.\n")
    print("| giữ lại | " + " | ".join(names) + " | BPR − MF | BPR − Popularity |")
    print("|---|" + "---|" * (len(names) + 2))
    for lv in [str(n) for n in N_KEEP] + ["day_du"]:
        cells = " | ".join(fmt_ci(bootstrap_ci(pu[nm, lv])) for nm in names)
        diffs = " | ".join(fmt_diff(paired_diff_ci(pu[BPR, lv], pu[b, lv]))
                           for b in (MFI, POP))
        label = "đầy đủ" if lv == "day_du" else f"{lv} tương tác"
        print(f"| {label} | {cells} | {diffs} |")


# ------------------------------------------------------------- 5. explain --


def vi_du(split, models, titles) -> None:
    print("\n## 5. Ví dụ câu giải thích thật\n")
    mb, mi = models[BPR], models[MFI]
    shown = 0
    for u in split.users:
        rec = mb.recommend(u, k=1)
        if not rec or not mb.explain(u, rec[0]):
            continue
        print(f"- **User {u}** → *{titles.get(rec[0], rec[0])}*")
        print(f"  - BPR: \"{mb.explain(u, rec[0])}\"")
        txt = mi.explain(u, rec[0])
        if txt:
            print(f"  - MF: \"{txt}\"")
        shown += 1
        if shown == 3:
            break


def main() -> None:
    ratings = movielens.load_ratings()
    movies = movielens.load_movies()
    titles = dict(zip(movies["item_id"], movies["title"]))
    split = leave_one_out(ratings, seed=SEED)
    print(f"# Thí nghiệm B4: BPR-MF\n\nml-latest-small · seed {SEED} · "
          f"{len(split.users)} user kiểm tra · ItemKNN k = {K_ITEM} · MF theo hằng số mf.py")

    chon = tim_sieu_tham_so(split.validation())
    models = make_models(chon, titles)
    pu, _ = bang_chinh(split, models)
    pho_bien(split, models)
    cold_start(split, chon, pu)
    vi_du(split, models, titles)


if __name__ == "__main__":
    main()
