"""Thi nghiem cua bac B3 (MF) -- chon sieu tham so tren VALIDATION, bao cao tren TEST.

    python scripts/exp_mf.py > docs/so-lieu-b3.md      # ~15-20 phut

1. CHON SIEU THAM SO tren validation (checklist 2.1), tim theo tang de do ton:
     implicit: f = 32 co dinh, do reg x alpha; roi lay cap tot nhat, do f.
     explicit: f = 32 co dinh, do reg; roi do f.
   Ket qua phai khop hang so trong recsys/models/mf.py -- lech thi canh bao.
2. BANG CHINH tren test: B3 implicit + explicit canh B0-B2, kem CI 95%, kem
   chenh lech ghep cap tren ca TOAN DANH MUC (nhu checklist 2.4).
3. COLD-START (nhu checklist 2.8, cung cach chia 4 phan cua exp_giao_thuc.py).
4. VI DU explain() that.

In Markdown ra stdout, tien do ra stderr. Moi bang ghi mot CSV kem .meta.json.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import movielens
from recsys.data.split import leave_one_out, truncate_history
from recsys.eval.provenance import save_result
from recsys.eval.runner import MAIN_K, evaluate
from recsys.eval.stats import bootstrap_ci, is_significant, paired_diff_ci
from recsys.models import mf
from recsys.models.baselines import Popularity
from recsys.models.knn import K_ITEM, K_USER, ItemKNN, UserKNN

K = MAIN_K
SEED = 42
OUT_DIR = Path("reports")
N_KEEP = (1, 3, 5, 10)
N_FOLDS = 4

POP, IKNN, UKNN = "B0-Popularity", "B1-ItemKNN", "B2-UserKNN"
MFI, MFE = "B3-MF-implicit", "B3-MF-explicit"


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


def tim_sieu_tham_so(val) -> tuple[dict, dict]:
    print("\n## 1. Chọn siêu tham số trên validation (ndcg@10)\n")
    rows = []

    def thu(kind, f, reg, alpha):
        t = time.time()
        m = mf.MF(factors=f, reg=reg, alpha=alpha, kind=kind).fit(val.train)
        r = evaluate(m, val)
        rows.append({"kind": kind, "factors": f, "reg": reg, "alpha": alpha,
                     f"ndcg@{K}": r[f"ndcg@{K}"], f"recall@{K}": r[f"recall@{K}"],
                     "giay_fit": round(time.time() - t, 1)})
        log(f"  {kind:<8} f={f:<4} reg={reg:<6} alpha={alpha:<5} "
            f"ndcg={r[f'ndcg@{K}']:.4f}  ({time.time() - t:.0f}s)")
        return r[f"ndcg@{K}"]

    def tot_nhat(kind, **loc):
        g = pd.DataFrame(rows)
        g = g[g["kind"] == kind]
        for k, v in loc.items():
            g = g[g[k] == v]
        return g.loc[g[f"ndcg@{K}"].idxmax()]

    # implicit: tang 1 -- reg x alpha o f = 32
    for reg in (1.0, 10.0, 100.0):
        for alpha in (1.0, 10.0, 40.0):
            thu("implicit", 32, reg, alpha)
    b = tot_nhat("implicit", factors=32)
    # tang 2 -- so nhan to
    for f in (16, 64, 128):
        thu("implicit", f, b["reg"], b["alpha"])
    best_i = tot_nhat("implicit")

    # explicit: alpha khong dung -- de 0 cho ro
    for reg in (1.0, 10.0, 30.0):
        thu("explicit", 32, reg, 0.0)
    b = tot_nhat("explicit", factors=32)
    for f in (16, 64):
        thu("explicit", f, b["reg"], 0.0)
    best_e = tot_nhat("explicit")

    grid = pd.DataFrame(rows).set_index(["kind", "factors", "reg", "alpha"])
    _save(grid, "exp_mf_1_validation.csv")
    print("| kind | factors | reg | alpha | ndcg@10 val | recall@10 val | giây fit |")
    print("|---|---|---|---|---|---|---|")
    for (kind, f, reg, alpha), r in grid.iterrows():
        print(f"| {kind} | {f} | {reg:g} | {alpha:g} | {r[f'ndcg@{K}']:.4f} "
              f"| {r[f'recall@{K}']:.4f} | {r['giay_fit']:g} |")

    chon_i = {"factors": int(best_i["factors"]), "reg": float(best_i["reg"]),
              "alpha": float(best_i["alpha"])}
    chon_e = {"factors": int(best_e["factors"]), "reg": float(best_e["reg"])}
    print(f"\nChọn: implicit {chon_i}, explicit {chon_e}.")

    da_ghi_i = {"factors": mf.FACTORS, "reg": mf.REG, "alpha": mf.ALPHA}
    da_ghi_e = {"factors": mf.FACTORS_EXPLICIT, "reg": mf.REG_EXPLICIT}
    if chon_i != da_ghi_i or chon_e != da_ghi_e:
        log(f"CANH BAO: mf.py dang ghi implicit {da_ghi_i}, explicit {da_ghi_e}. "
            "Sua hang so trong recsys/models/mf.py.")
    return chon_i, chon_e


def make_models(chon_i, chon_e, titles=None) -> dict:
    mi = mf.MF(kind="implicit", titles=titles, **chon_i)
    me = mf.MF(kind="explicit", **chon_e)
    mi.name, me.name = MFI, MFE
    return {POP: Popularity(), IKNN: ItemKNN(k=K_ITEM, titles=titles),
            UKNN: UserKNN(k=K_USER), MFI: mi, MFE: me}


# ------------------------------------------------------------ 2 + 3. test --


def bang_chinh(split, models) -> dict:
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
    _save(pd.DataFrame(rows).set_index("mo_hinh").round(4), "exp_mf_2_test.csv")

    print(f"| Mô hình | recall@{K} | ndcg@{K} | AUC | coverage@{K} | ứng viên −inf |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        p = pu[r["mo_hinh"]]
        print(f"| {r['mo_hinh']} | {fmt_ci(bootstrap_ci(p[f'recall@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(p[f'ndcg@{K}']))} | {fmt_ci(bootstrap_ci(p['auc']))} "
              f"| {r[f'coverage@{K}']:.4f} | {r['vo_cuc']:.1%} |")

    print(f"\n### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục\n")
    print("| So sánh | độ đo | lấy mẫu | toàn danh mục |")
    print("|---|---|---|---|")
    pairs = [(MFI, IKNN), (MFI, UKNN), (MFI, POP), (MFE, POP), (MFI, MFE)]
    for metric in (f"ndcg@{K}", f"recall@{K}"):
        for a, b in pairs:
            c1 = fmt_diff(paired_diff_ci(pu[a][metric], pu[b][metric]))
            c2 = fmt_diff(paired_diff_ci(pu_full[a][metric], pu_full[b][metric]))
            print(f"| {a} − {b} | {metric} | {c1} | {c2} |")

    print(f"\nToàn danh mục, ndcg@{K}: " + ", ".join(
        f"{n} {fmt_ci(bootstrap_ci(pu_full[n][f'ndcg@{K}']))}" for n in models))
    return pu


# --------------------------------------------------------------- 4. cold --


def cold_start(split, chon_i, pu_full_hist) -> None:
    print("\n## 3. Cold-start nhân tạo (cắt còn n tương tác gần nhất)\n")
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(np.array(split.users)), N_FOLDS)
    names = [POP, IKNN, UKNN, MFI]
    pu: dict[tuple[str, str], pd.Series] = {}
    rows = []
    for n in N_KEEP:
        parts: dict[str, list] = {nm: [] for nm in names}
        for fold in folds:
            cut = truncate_history(split, fold, n)
            cut = replace(cut, test={int(u): split.test[int(u)] for u in fold})
            fold_models = {POP: Popularity(), IKNN: ItemKNN(k=K_ITEM),
                           UKNN: UserKNN(k=K_USER), MFI: mf.MF(kind="implicit", **chon_i)}
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
          "exp_mf_3_cold_start.csv", n_folds=N_FOLDS)

    print(f"ndcg@{K}, CI 95%, đủ {len(split.users)} user.\n")
    print("| giữ lại | " + " | ".join(names) + " | MF − ItemKNN | MF − UserKNN | MF − Popularity |")
    print("|---|" + "---|" * (len(names) + 3))
    for lv in [str(n) for n in N_KEEP] + ["day_du"]:
        cells = " | ".join(fmt_ci(bootstrap_ci(pu[nm, lv])) for nm in names)
        diffs = " | ".join(fmt_diff(paired_diff_ci(pu[MFI, lv], pu[b, lv]))
                           for b in (IKNN, UKNN, POP))
        label = "đầy đủ" if lv == "day_du" else f"{lv} tương tác"
        print(f"| {label} | {cells} | {diffs} |")


# ------------------------------------------------------------- 5. explain --


def vi_du(split, models, titles) -> None:
    print("\n## 4. Ví dụ câu giải thích thật\n")
    mi, ik = models[MFI], models[IKNN]
    shown = 0
    for u in split.users:
        rec = mi.recommend(u, k=1)
        if not rec or not mi.explain(u, rec[0]):
            continue
        print(f"- **User {u}** → *{titles.get(rec[0], rec[0])}*")
        print(f"  - MF: \"{mi.explain(u, rec[0])}\"")
        txt = ik.explain(u, rec[0])
        if txt:
            print(f"  - ItemKNN: \"{txt}\"")
        shown += 1
        if shown == 3:
            break


def main() -> None:
    ratings = movielens.load_ratings()
    movies = movielens.load_movies()
    titles = dict(zip(movies["item_id"], movies["title"]))
    split = leave_one_out(ratings, seed=SEED)
    print(f"# Thí nghiệm B3: phân rã ma trận (ALS)\n\nml-latest-small · seed {SEED} · "
          f"{len(split.users)} user kiểm tra · ItemKNN k = {K_ITEM}, UserKNN k = {K_USER}")

    chon_i, chon_e = tim_sieu_tham_so(split.validation())
    models = make_models(chon_i, chon_e, titles)
    pu = bang_chinh(split, models)
    cold_start(split, chon_i, pu)
    vi_du(split, models, titles)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
