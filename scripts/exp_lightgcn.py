"""Thi nghiem cua bac B8 (LightGCN) -- chon sieu tham so tren VALIDATION, bao cao tren TEST.

    python scripts/exp_lightgcn.py > docs/so-lieu-b8.md

Cau hoi cua bac B8: lan truyen tren do thi user-phim co them gi so voi BPR-MF
(B4), mo hinh cung ham mat mat va cung dang diem, va tra gia bang gi.

1. CHON SIEU THAM SO tren validation (checklist 2.1), tim theo tang:
     K = 3 lop co dinh, do lr x reg; moi cau hinh danh gia sau nhieu moc epoch
     (train_more, khong huan luyen lai); roi do K o cap tot nhat, ke ca K = 0.
   Ket qua phai khop hang so trong recsys/models/lightgcn.py -- lech thi canh bao.
2. BANG CHINH tren test: LightGCN canh B0, B1, B3, B4, kem CI 95% va chenh lech
   ghep cap, ca lay mau lan toan danh muc.
3. DONG GOP CUA DO THI: cung cau hinh, K = 0 (BPR-MF khong bias, cung Adam) den
   K da chon, tren test. Tach phan hon B4 do do thi hay do cach toi uu.
4. THIEN LECH PHO BIEN: top-10 phu bao nhieu danh muc, phim pho bien den dau,
   do dai vector phim co theo bac cua nut khong.
5. COLD-START (nhu checklist 2.8, cung cach chia 4 phan cua exp_giao_thuc.py).
6. VI DU explain() that.

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
from recsys.models import lightgcn
from recsys.models.baselines import Popularity
from recsys.models.bpr import BPR
from recsys.models.knn import K_ITEM, ItemKNN
from recsys.models.mf import MF

K = MAIN_K
SEED = 42
OUT_DIR = Path("reports")
N_KEEP = (1, 3, 5, 10)
N_FOLDS = 4
CHECKPOINTS = (25, 50, 100, 200, 300)
LAYERS = (0, 1, 2, 3, 4)

POP, IKNN, MFI, BPRN, LGN = "B0-Popularity", "B1-ItemKNN", "B3-MF-implicit", "B4-BPR", "B8-LightGCN"


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


def tim_sieu_tham_so(val) -> tuple[dict, pd.DataFrame]:
    print("\n## 1. Chọn siêu tham số trên validation (ndcg@10)\n")
    rows = []

    def thu(n_layers, lr, reg):
        t = time.time()
        m = lightgcn.LightGCN(n_layers=n_layers, lr=lr, reg=reg, n_epochs=0).fit(val.train)
        done = 0
        for ep in CHECKPOINTS:
            m.train_more(ep - done)
            done = ep
            r = evaluate(m, val)
            rows.append({"n_layers": n_layers, "lr": lr, "reg": reg, "n_epochs": ep,
                         f"ndcg@{K}": r[f"ndcg@{K}"], f"recall@{K}": r[f"recall@{K}"],
                         "loss": m.loss_[-1]})
            log(f"  K={n_layers} lr={lr:<6} reg={reg:<6} ep={ep:<4} "
                f"ndcg={r[f'ndcg@{K}']:.4f}  ({time.time() - t:.0f}s)")

    def tot_nhat(**loc):
        g = pd.DataFrame(rows)
        for k, v in loc.items():
            g = g[g[k] == v]
        return g.loc[g[f"ndcg@{K}"].idxmax()]

    for lr in (0.01, 0.02, 0.05):
        for reg in (1e-4, 1e-2):
            thu(3, lr, reg)
    b = tot_nhat(n_layers=3)
    for n_layers in LAYERS:
        if n_layers != 3:
            thu(n_layers, b["lr"], b["reg"])
    best = tot_nhat()

    grid = pd.DataFrame(rows).set_index(["n_layers", "lr", "reg", "n_epochs"])
    _save(grid, "exp_lightgcn_1_validation.csv")
    print(f"Mỗi cấu hình huấn luyện một lần, đánh giá sau {', '.join(map(str, CHECKPOINTS))} "
          "epoch. Ô đậm là tốt nhất của từng cấu hình. f = 64, lô 8.192 cặp, Adam.\n")
    print("| số lớp K | lr | reg | " + " | ".join(f"{e} epoch" for e in CHECKPOINTS) + " |")
    print("|---|---|---|" + "---|" * len(CHECKPOINTS))
    for (nl, lr, reg), g in grid.groupby(level=[0, 1, 2], sort=False):
        nd = g[f"ndcg@{K}"].to_numpy()
        cells = [f"**{v:.4f}**" if v == nd.max() else f"{v:.4f}" for v in nd]
        print(f"| {nl} | {lr:g} | {reg:g} | " + " | ".join(cells) + " |")

    chon = {"n_layers": int(best["n_layers"]), "lr": float(best["lr"]),
            "reg": float(best["reg"]), "n_epochs": int(best["n_epochs"])}
    print(f"\nChọn: {chon}, ndcg@10 validation {best[f'ndcg@{K}']:.4f}.")

    da_ghi = {"n_layers": lightgcn.N_LAYERS, "lr": lightgcn.LR, "reg": lightgcn.REG,
              "n_epochs": lightgcn.N_EPOCHS}
    if chon != da_ghi:
        log(f"CANH BAO: lightgcn.py dang ghi {da_ghi}. Sua hang so trong "
            "recsys/models/lightgcn.py.")
    return chon, grid


def make_models(chon, titles=None) -> dict:
    mi = MF(titles=titles)
    mi.name = MFI
    return {POP: Popularity(), IKNN: ItemKNN(k=K_ITEM, titles=titles), MFI: mi,
            BPRN: BPR(titles=titles), LGN: lightgcn.LightGCN(titles=titles, **chon)}


# ------------------------------------------------------------------ 2. test --


def bang_chinh(split, models) -> tuple[dict, dict]:
    print("\n## 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)\n")
    pu, pu_full, rows = {}, {}, []
    for name, m in models.items():
        log(f"  fit {name}")
        t = time.time()
        m.fit(split.train)
        giay = time.time() - t
        r = evaluate(m, split, keep_per_user=True)
        pu[name] = r.pop("per_user")
        rf = evaluate(m, split, keep_per_user=True, full_catalog=True)
        pu_full[name] = rf.pop("per_user")
        rows.append({"mo_hinh": name, **{c: r[c] for c in (
            f"recall@{K}", f"ndcg@{K}", "mrr", "auc", "vo_cuc", f"coverage@{K}")},
            f"ndcg@{K}_toan_danh_muc": rf[f"ndcg@{K}"],
            f"recall@{K}_toan_danh_muc": rf[f"recall@{K}"], "giay_fit": round(giay, 1)})
    _save(pd.DataFrame(rows).set_index("mo_hinh").round(4), "exp_lightgcn_2_test.csv")

    print(f"| Mô hình | recall@{K} | ndcg@{K} | AUC | coverage@{K} | ứng viên −inf "
          "| giây fit |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        p = pu[r["mo_hinh"]]
        print(f"| {r['mo_hinh']} | {fmt_ci(bootstrap_ci(p[f'recall@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(p[f'ndcg@{K}']))} | {fmt_ci(bootstrap_ci(p['auc']))} "
              f"| {r[f'coverage@{K}']:.4f} | {r['vo_cuc']:.1%} | {r['giay_fit']:.0f} |")

    print("\n### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục\n")
    print("| So sánh | độ đo | lấy mẫu | toàn danh mục |")
    print("|---|---|---|---|")
    pairs = [(LGN, BPRN), (LGN, MFI), (LGN, IKNN), (LGN, POP)]
    for metric in (f"ndcg@{K}", f"recall@{K}", "auc"):
        for a, b in pairs:
            c1 = fmt_diff(paired_diff_ci(pu[a][metric], pu[b][metric]))
            c2 = fmt_diff(paired_diff_ci(pu_full[a][metric], pu_full[b][metric]))
            print(f"| {a} − {b} | {metric} | {c1} | {c2} |")

    print(f"\nToàn danh mục, ndcg@{K}: " + ", ".join(
        f"{n} {fmt_ci(bootstrap_ci(pu_full[n][f'ndcg@{K}']))}" for n in models))
    return pu, pu_full


# ---------------------------------------------------------------- 3. do thi --


def dong_gop_do_thi(split, grid, chon, pu_lgn, pu_bpr) -> None:
    print("\n## 3. Đóng góp của đồ thị: số lớp lan truyền K, cùng lr, reg, epoch\n")
    print("K = 0 là BPR-MF không bias, học bằng Adam. Mọi dòng dùng cùng lr, reg, số epoch "
          "đã chọn ở mục 1.\n")
    pu, rows = {}, []
    for nl in LAYERS:
        cfg = {**chon, "n_layers": nl}
        if nl == chon["n_layers"]:
            p = pu_lgn
        else:
            m = lightgcn.LightGCN(**cfg).fit(split.train)
            p = evaluate(m, split, keep_per_user=True)["per_user"]
        # Validation lay tu luoi cua muc 1: moi K deu da do o cung lr, reg, moc epoch.
        nd_val = grid.loc[(nl, chon["lr"], chon["reg"], chon["n_epochs"]), f"ndcg@{K}"]
        pu[nl] = p[f"ndcg@{K}"]
        rows.append({"n_layers": nl, f"ndcg@{K}_val": nd_val, f"ndcg@{K}_test": pu[nl].mean(),
                     f"recall@{K}_test": p[f"recall@{K}"].mean(), "auc_test": p["auc"].mean()})
        log(f"  K = {nl}: xong")
    _save(pd.DataFrame(rows).set_index("n_layers").round(4), "exp_lightgcn_3_so_lop.csv")

    print(f"| K | ndcg@{K} validation | ndcg@{K} test | K − (K = 0), test | K − B4-BPR, test |")
    print("|---|---|---|---|---|")
    for r in rows:
        nl = r["n_layers"]
        d0 = "" if nl == 0 else fmt_diff(paired_diff_ci(pu[nl], pu[0]))
        db = fmt_diff(paired_diff_ci(pu[nl], pu_bpr[f"ndcg@{K}"]))
        print(f"| {nl} | {r[f'ndcg@{K}_val']:.4f} | {fmt_ci(bootstrap_ci(pu[nl]))} "
              f"| {d0} | {db} |")


# ---------------------------------------------------------------- 4. pho bien --


def pho_bien(split, models) -> None:
    print("\n## 4. Thiên lệch phổ biến của top-10 (toàn danh mục, mọi user)\n")
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
                     "trung_vi_luot_xem": float(np.median([dem.get(i, 0) for i in phim])),
                     "ty_le_top100": float(np.mean([i in top100 for i in phim]))})
    df = pd.DataFrame(rows).set_index("mo_hinh")
    _save(df.round(4), "exp_lightgcn_4_pho_bien.csv")
    print("| Mô hình | số phim khác nhau trong top-10 | phủ danh mục "
          "| lượt xem TB | trung vị lượt xem | tỉ lệ thuộc top-100 phổ biến |")
    print("|---|---|---|---|---|---|")
    for name, r in df.iterrows():
        print(f"| {name} | {r['so_phim_khac_nhau']:.0f} | {r['phu_danh_muc']:.1%} "
              f"| {r['luot_xem_tb']:.1f} | {r['trung_vi_luot_xem']:.0f} "
              f"| {r['ty_le_top100']:.1%} |")

    print()
    for name in (BPRN, LGN):
        m = models[name]
        counts = dem.reindex(m.items_).to_numpy()
        rho = spearmanr(np.linalg.norm(m.Q_, axis=1), counts).statistic
        print(f"- Spearman giữa độ dài vector phim của {name} và số lượt xem: {rho:.3f}.")
    rho_b = spearmanr(models[BPRN].b_, dem.reindex(models[BPRN].items_).to_numpy()).statistic
    print(f"- Tham chiếu: Spearman giữa bias b_i của {BPRN} và số lượt xem: {rho_b:.3f}.")


# --------------------------------------------------------------- 5. cold --


def cold_start(split, chon, pu_full_hist) -> None:
    print("\n## 5. Cold-start nhân tạo (cắt còn n tương tác gần nhất)\n")
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(np.array(split.users)), N_FOLDS)
    names = [POP, MFI, BPRN, LGN]
    pu: dict[tuple[str, str], pd.Series] = {}
    rows = []
    for n in N_KEEP:
        parts: dict[str, list] = {nm: [] for nm in names}
        for fold in folds:
            cut = truncate_history(split, fold, n)
            cut = replace(cut, test={int(u): split.test[int(u)] for u in fold})
            fold_models = {POP: Popularity(), MFI: MF(), BPRN: BPR(),
                           LGN: lightgcn.LightGCN(**chon)}
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
          "exp_lightgcn_5_cold_start.csv", n_folds=N_FOLDS)

    print(f"ndcg@{K}, CI 95%, đủ {len(split.users)} user.\n")
    print("| giữ lại | " + " | ".join(names) + " | LightGCN − BPR | LightGCN − Popularity |")
    print("|---|" + "---|" * (len(names) + 2))
    for lv in [str(n) for n in N_KEEP] + ["day_du"]:
        cells = " | ".join(fmt_ci(bootstrap_ci(pu[nm, lv])) for nm in names)
        diffs = " | ".join(fmt_diff(paired_diff_ci(pu[LGN, lv], pu[b, lv]))
                           for b in (BPRN, POP))
        label = "đầy đủ" if lv == "day_du" else f"{lv} tương tác"
        print(f"| {label} | {cells} | {diffs} |")


# ------------------------------------------------------------- 6. explain --


def vi_du(split, models, titles) -> None:
    print("\n## 6. Ví dụ câu giải thích thật\n")
    ml, mb = models[LGN], models[BPRN]
    shown = 0
    for u in split.users:
        rec = ml.recommend(u, k=1)
        if not rec or not ml.explain(u, rec[0]):
            continue
        print(f"- **User {u}** → *{titles.get(rec[0], rec[0])}*")
        print(f"  - LightGCN: \"{ml.explain(u, rec[0])}\"")
        txt = mb.explain(u, rec[0])
        if txt:
            print(f"  - BPR: \"{txt}\"")
        shown += 1
        if shown == 3:
            break


def main() -> None:
    ratings = movielens.load_ratings()
    movies = movielens.load_movies()
    titles = dict(zip(movies["item_id"], movies["title"]))
    split = leave_one_out(ratings, seed=SEED)
    print(f"# Thí nghiệm B8: LightGCN\n\nml-latest-small · seed {SEED} · "
          f"{len(split.users)} user kiểm tra · ItemKNN k = {K_ITEM} · MF, BPR theo hằng số "
          "mf.py, bpr.py")

    chon, grid = tim_sieu_tham_so(split.validation())
    models = make_models(chon, titles)
    pu, _ = bang_chinh(split, models)
    dong_gop_do_thi(split, grid, chon, pu[LGN], pu[BPRN])
    pho_bien(split, models)
    cold_start(split, chon, pu)
    vi_du(split, models, titles)


if __name__ == "__main__":
    main()
