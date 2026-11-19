"""Thi nghiem cua bac B5 (noi dung) -- chon sieu tham so tren VALIDATION, bao cao tren TEST.

    python scripts/exp_content.py > docs/so-lieu-b5.md     # ~15-20 phut

Cau hoi cua bac B5: metadata phim thay duoc bao nhieu phan hanh vi xem (B0-B4), va
no lam duoc gi ma loc cong tac khong lam duoc.

1. CHON SIEU THAM SO tren validation (checklist 2.1): nhom dac trung x cach danh
   trong so x half_life; roi trong so the loai / dao dien o cau hinh tot nhat.
   Ket qua phai khop hang so trong recsys/models/content.py -- lech thi canh bao.
2. BANG CHINH tren test: B5 canh B0, B1, B4, kem CI 95%, ca lay mau lan toan danh muc.
   Them: 23 user co phim test chua tung xuat hien trong train (checklist 2.6).
3. NOI TEN NGUOI hay TACH nhu ma nguon tham khao (xem tmdb._token).
4. THIEN LECH PHO BIEN cua top-10.
5. PHIM MOI (item cold-start): xoa moi tuong tac train voi phim test cua 1/4 so user
   mot luc, do lai. Loc cong tac khong cham duoc phim do; B5 thi co.
6. USER MOI (cat lich su, nhu checklist 2.8).
7. VI DU explain() that.

In Markdown ra stdout, tien do ra stderr. Moi bang ghi mot CSV kem .meta.json.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import movielens, tmdb
from recsys.data.split import leave_one_out, truncate_history
from recsys.eval.provenance import save_result
from recsys.eval.runner import MAIN_K, evaluate
from recsys.eval.stats import bootstrap_ci, is_significant, paired_diff_ci
from recsys.models import content
from recsys.models.baselines import Popularity
from recsys.models.bpr import BPR
from recsys.models.knn import K_ITEM, ItemKNN

K = MAIN_K
SEED = 42
OUT_DIR = Path("reports")
N_KEEP = (1, 3, 5, 10)
N_FOLDS = 4

FIELD_SETS = {
    "the_loai": ("genres",),
    "the_loai+nam": ("genres", "year"),
    "tmdb": ("genres", "year", "directors", "cast", "keywords"),
    "tmdb+tom_tat": ("genres", "year", "directors", "cast", "keywords", "overview"),
}
HALF_LIVES = (None, 50, 20, 10, 5, 3, 1)
WEIGHTS = (1, 2, 4)

POP, IKNN, BPR4, B5 = "B0-Popularity", "B1-ItemKNN", "B4-BPR", "B5-Content"


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


def _hl(h) -> str:
    return "không" if h is None else f"{h:g}"


# ------------------------------------------------------------------ 1. chon --


def tim_sieu_tham_so(val, meta) -> dict:
    print("\n## 1. Chọn siêu tham số trên validation (ndcg@10)\n")
    rows = []

    def thu(fs, weighting, hl, gw=content.GENRE_WEIGHT, dw=content.DIRECTOR_WEIGHT):
        m = content.ContentBased(meta, fields=FIELD_SETS[fs], weighting=weighting,
                                 half_life=hl, genre_weight=gw, director_weight=dw)
        r = evaluate(m.fit(val.train), val)
        rows.append({"fields": fs, "weighting": weighting, "half_life": _hl(hl),
                     "genre_weight": gw, "director_weight": dw,
                     f"ndcg@{K}": r[f"ndcg@{K}"], f"recall@{K}": r[f"recall@{K}"],
                     "vo_cuc": r["vo_cuc"]})
        log(f"  {fs:<13} {weighting:<6} hl={_hl(hl):<6} gw={gw} dw={dw} "
            f"ndcg={r[f'ndcg@{K}']:.4f}")
        return rows[-1]

    t = time.time()
    for fs in FIELD_SETS:
        for w in ("binary", "rating"):
            for hl in HALF_LIVES:
                thu(fs, w, hl)
    g = pd.DataFrame(rows)
    b = g.loc[g[f"ndcg@{K}"].idxmax()]
    hl_b = None if b["half_life"] == "không" else float(b["half_life"])
    for gw in WEIGHTS:
        for dw in (WEIGHTS if "directors" in FIELD_SETS[b["fields"]] else (content.DIRECTOR_WEIGHT,)):
            if (gw, dw) != (content.GENRE_WEIGHT, content.DIRECTOR_WEIGHT):
                thu(b["fields"], b["weighting"], hl_b, gw, dw)
    log(f"  chon xong ({time.time() - t:.0f}s)")

    grid = pd.DataFrame(rows)
    _save(grid.set_index(["fields", "weighting", "half_life", "genre_weight",
                          "director_weight"]).round(4), "exp_content_1_validation.csv")

    print(f"Trọng số thể loại = {content.GENRE_WEIGHT}, đạo diễn = {content.DIRECTOR_WEIGHT}. "
          "Ô đậm là tốt nhất của từng hàng.\n")
    print("| đặc trưng | trọng số | " + " | ".join(f"half_life {_hl(h)}" for h in HALF_LIVES) + " |")
    print("|---|---|" + "---|" * len(HALF_LIVES))
    base = grid[(grid["genre_weight"] == content.GENRE_WEIGHT)
                & (grid["director_weight"] == content.DIRECTOR_WEIGHT)]
    for (fs, w), gg in base.groupby(["fields", "weighting"], sort=False):
        nd = gg[f"ndcg@{K}"].to_numpy()
        cells = [f"**{v:.4f}**" if v == nd.max() else f"{v:.4f}" for v in nd]
        print(f"| {fs} | {w} | " + " | ".join(cells) + " |")

    extra = grid[(grid["fields"] == b["fields"]) & (grid["weighting"] == b["weighting"])
                 & (grid["half_life"] == b["half_life"])]
    print(f"\nTrọng số ở cấu hình tốt nhất ({b['fields']}, {b['weighting']}, "
          f"half_life {b['half_life']}):\n")
    print("| trọng số thể loại | trọng số đạo diễn | ndcg@10 |")
    print("|---|---|---|")
    for _, r in extra.sort_values(["genre_weight", "director_weight"]).iterrows():
        print(f"| {r['genre_weight']} | {r['director_weight']} | {r[f'ndcg@{K}']:.4f} |")

    best = grid.loc[grid[f"ndcg@{K}"].idxmax()]
    chon = {"fields": FIELD_SETS[best["fields"]], "weighting": best["weighting"],
            "half_life": None if best["half_life"] == "không" else float(best["half_life"]),
            "genre_weight": int(best["genre_weight"]),
            "director_weight": int(best["director_weight"])}
    print(f"\nChọn: {best['fields']} {chon}, ndcg@10 validation {best[f'ndcg@{K}']:.4f}.")

    da_ghi = {"fields": tuple(content.FIELDS), "weighting": content.WEIGHTING,
              "half_life": content.HALF_LIFE, "genre_weight": content.GENRE_WEIGHT,
              "director_weight": content.DIRECTOR_WEIGHT}
    if chon != da_ghi:
        log(f"CANH BAO: content.py dang ghi {da_ghi}. Sua hang so trong "
            "recsys/models/content.py.")
    return chon


# ------------------------------------------------------------------ 2. test --


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
    _save(pd.DataFrame(rows).set_index("mo_hinh").round(4), "exp_content_2_test.csv")

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
    for metric in (f"ndcg@{K}", f"recall@{K}", "auc"):
        for b in (POP, IKNN, BPR4):
            c1 = fmt_diff(paired_diff_ci(pu[B5][metric], pu[b][metric]))
            c2 = fmt_diff(paired_diff_ci(pu_full[B5][metric], pu_full[b][metric]))
            print(f"| {B5} − {b} | {metric} | {c1} | {c2} |")

    print(f"\nToàn danh mục, ndcg@{K}: " + ", ".join(
        f"{n} {fmt_ci(bootstrap_ci(pu_full[n][f'ndcg@{K}']))}" for n in models))

    # Checklist 2.6: phim test khong co trong train.
    trong_train = set(split.train["item_id"])
    la = [u for u in split.users if split.test[u] not in trong_train]
    print(f"\n### {len(la)} user có phim test chưa từng xuất hiện trong train (checklist 2.6)\n")
    print("Số user trúng top-10 trong nhóm này:\n")
    print("| Mô hình | lấy mẫu | toàn danh mục |")
    print("|---|---|---|")
    for n in models:
        h1 = int((pu[n].loc[la, f"recall@{K}"] > 0).sum())
        h2 = int((pu_full[n].loc[la, f"recall@{K}"] > 0).sum())
        print(f"| {n} | {h1}/{len(la)} | {h2}/{len(la)} |")
    return pu


# --------------------------------------------------------------- 3. ten nguoi --


def ten_nguoi(split, meta, chon, pu) -> None:
    print("\n## 3. Nối tên người thành một token hay tách như mã nguồn tham khảo\n")
    if not ({"directors", "cast"} & set(chon["fields"])):
        print("Cấu hình đã chọn không dùng đạo diễn hay diễn viên, nên không đo được.")
        return
    tach = content.ContentBased(meta, join_names=False, **chon).fit(split.train)
    r = evaluate(tach, split, keep_per_user=True)
    p = r.pop("per_user")
    _save(pd.DataFrame([{"cach": "tach", f"ndcg@{K}": r[f"ndcg@{K}"],
                         f"recall@{K}": r[f"recall@{K}"]}]).set_index("cach").round(4),
          "exp_content_3_ten_nguoi.csv")
    print("| cách | ndcg@10 | recall@10 |")
    print("|---|---|---|")
    print(f"| nối (`james_cameron`) | {fmt_ci(bootstrap_ci(pu[B5][f'ndcg@{K}']))} "
          f"| {fmt_ci(bootstrap_ci(pu[B5][f'recall@{K}']))} |")
    print(f"| tách (`james cameron`) | {fmt_ci(bootstrap_ci(p[f'ndcg@{K}']))} "
          f"| {fmt_ci(bootstrap_ci(p[f'recall@{K}']))} |")
    print(f"\nNối − tách, ndcg@10: {fmt_diff(paired_diff_ci(pu[B5][f'ndcg@{K}'], p[f'ndcg@{K}']))}")


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
                     "luot_xem_trung_vi": float(np.median([dem.get(i, 0) for i in phim])),
                     "ty_le_top100": float(np.mean([i in top100 for i in phim]))})
    df = pd.DataFrame(rows).set_index("mo_hinh")
    _save(df.round(4), "exp_content_4_pho_bien.csv")
    print("| Mô hình | số phim khác nhau | phủ danh mục | lượt xem TB | lượt xem trung vị "
          "| tỉ lệ thuộc top-100 phổ biến |")
    print("|---|---|---|---|---|---|")
    for name, r in df.iterrows():
        print(f"| {name} | {r['so_phim_khac_nhau']:.0f} | {r['phu_danh_muc']:.1%} "
              f"| {r['luot_xem_tb']:.1f} | {r['luot_xem_trung_vi']:.0f} | {r['ty_le_top100']:.1%} |")


# ---------------------------------------------------------------- 5. phim moi --


def _folds(split):
    rng = np.random.default_rng(SEED)
    return np.array_split(rng.permutation(np.array(split.users)), N_FOLDS)


def phim_moi(split, meta, chon) -> None:
    print("\n## 5. Phim mới: phim test bị xóa khỏi train (item cold-start)\n")
    names = [POP, BPR4, B5]
    parts: dict[str, list] = {n: [] for n in names}
    xoa = 0
    for fold in _folds(split):
        targets = {int(split.test[int(u)]) for u in fold}
        keep = ~split.train["item_id"].isin(targets)
        xoa += int((~keep).sum())
        cut = replace(split, train=split.train[keep],
                      test={int(u): split.test[int(u)] for u in fold})
        for nm, m in ((POP, Popularity()), (BPR4, BPR()),
                      (B5, content.ContentBased(meta, **chon))):
            m.fit(cut.train)
            parts[nm].append(evaluate(m, cut, keep_per_user=True)["per_user"])
        log("  phim moi: xong mot phan")
    pu = {n: pd.concat(parts[n]) for n in names}
    rows = [{"mo_hinh": n, f"ndcg@{K}": pu[n][f"ndcg@{K}"].mean(),
             f"recall@{K}": pu[n][f"recall@{K}"].mean(), "auc": pu[n]["auc"].mean()}
            for n in names]
    _save(pd.DataFrame(rows).set_index("mo_hinh").round(4), "exp_content_5_phim_moi.csv",
          n_folds=N_FOLDS, so_dong_xoa=xoa)
    print(f"Mỗi lần lấy 1/{N_FOLDS} số user, xóa mọi tương tác train với phim test của họ "
          f"(tổng {xoa:,} dòng qua {N_FOLDS} lần), rồi đo trên chính các user đó. "
          f"Đủ {len(split.users)} user, CI 95%.\n")
    print(f"| Mô hình | recall@{K} | ndcg@{K} | AUC |")
    print("|---|---|---|---|")
    for n in names:
        print(f"| {n} | {fmt_ci(bootstrap_ci(pu[n][f'recall@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(pu[n][f'ndcg@{K}']))} | {fmt_ci(bootstrap_ci(pu[n]['auc']))} |")
    print(f"\nB5 − Popularity, ndcg@{K}: {fmt_diff(paired_diff_ci(pu[B5][f'ndcg@{K}'], pu[POP][f'ndcg@{K}']))}")
    print(f"B5 − BPR, ndcg@{K}: {fmt_diff(paired_diff_ci(pu[B5][f'ndcg@{K}'], pu[BPR4][f'ndcg@{K}']))}")


# --------------------------------------------------------------- 6. user moi --


def user_moi(split, meta, chon, pu_full_hist) -> None:
    print("\n## 6. User mới: cắt còn n tương tác gần nhất (như checklist 2.8)\n")
    names = [POP, BPR4, B5]
    pu: dict[tuple[str, str], pd.Series] = {}
    rows = []
    folds = _folds(split)
    for n in N_KEEP:
        parts: dict[str, list] = {nm: [] for nm in names}
        for fold in folds:
            cut = truncate_history(split, fold, n)
            cut = replace(cut, test={int(u): split.test[int(u)] for u in fold})
            for nm, m in ((POP, Popularity()), (BPR4, BPR()),
                          (B5, content.ContentBased(meta, **chon))):
                m.fit(cut.train)
                parts[nm].append(evaluate(m, cut, keep_per_user=True)["per_user"])
        for nm in names:
            df = pd.concat(parts[nm])
            pu[nm, str(n)] = df[f"ndcg@{K}"]
            rows.append({"n_giu": n, "mo_hinh": nm, f"ndcg@{K}": df[f"ndcg@{K}"].mean()})
        log(f"  user moi n = {n}: xong")
    for nm in names:
        pu[nm, "day_du"] = pu_full_hist[nm][f"ndcg@{K}"]
        rows.append({"n_giu": "day_du", "mo_hinh": nm,
                     f"ndcg@{K}": pu_full_hist[nm][f"ndcg@{K}"].mean()})
    _save(pd.DataFrame(rows).set_index(["n_giu", "mo_hinh"]).round(4),
          "exp_content_6_user_moi.csv", n_folds=N_FOLDS)

    print(f"ndcg@{K}, CI 95%, đủ {len(split.users)} user.\n")
    print("| giữ lại | " + " | ".join(names) + " | B5 − Popularity | B5 − BPR |")
    print("|---|" + "---|" * (len(names) + 2))
    for lv in [str(n) for n in N_KEEP] + ["day_du"]:
        cells = " | ".join(fmt_ci(bootstrap_ci(pu[nm, lv])) for nm in names)
        diffs = " | ".join(fmt_diff(paired_diff_ci(pu[B5, lv], pu[b, lv])) for b in (POP, BPR4))
        label = "đầy đủ" if lv == "day_du" else f"{lv} tương tác"
        print(f"| {label} | {cells} | {diffs} |")


# ------------------------------------------------------------- 7. explain --


def vi_du(split, model, titles) -> None:
    print("\n## 7. Ví dụ câu giải thích thật\n")
    shown = 0
    for u in split.users:
        rec = model.recommend(u, k=1)
        if not rec:
            continue
        txt = model.explain(u, rec[0])
        if not txt:
            continue
        print(f"- **User {u}** → *{titles.get(rec[0], rec[0])}*: \"{txt}\"")
        shown += 1
        if shown == 3:
            break


def main() -> None:
    ratings = movielens.load_ratings()
    movies = movielens.load_movies()
    titles = dict(zip(movies["item_id"], movies["title"]))
    meta = tmdb.load_metadata()
    split = leave_one_out(ratings, seed=SEED)
    print(f"# Thí nghiệm B5: gợi ý theo nội dung\n\nml-latest-small · seed {SEED} · "
          f"{len(split.users)} user kiểm tra · ItemKNN k = {K_ITEM} · BPR theo hằng số bpr.py\n")
    print("Metadata:\n\n```\n" + tmdb.coverage_report(meta) + "\n```")

    chon = tim_sieu_tham_so(split.validation(), meta)
    models = {POP: Popularity(), IKNN: ItemKNN(k=K_ITEM), BPR4: BPR(),
              B5: content.ContentBased(meta, titles=titles, **chon)}
    pu = bang_chinh(split, models)
    ten_nguoi(split, meta, chon, pu)
    pho_bien(split, models)
    phim_moi(split, meta, chon)
    user_moi(split, meta, chon, pu)
    vi_du(split, models[B5], titles)


if __name__ == "__main__":
    main()
