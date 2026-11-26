"""Thi nghiem cua bac B6 (luat ket hop theo phien) -- chon sieu tham so tren VALIDATION,
bao cao tren TEST.

    python scripts/exp_rules.py > docs/so-lieu-b6.md     # ~70 phut

Cau hoi cua bac B6: chi nhin PHIEN GAN NHAT cua user, bang luat khai pha tu phien cua
moi nguoi, thi doan duoc tuong tac tiep theo den dau so voi lich su day du (B1, B4).

1. CHON SIEU THAM SO tren validation (checklist 2.1), tim theo tang:
     a. session_gap x min_support (moi cau hinh khai pha mot lan);
     b. o cap tot nhat: context x metric x agg (rescore, khong khai pha lai);
     c. max_len = 3 o cau hinh tot nhat.
   Ket qua phai khop hang so trong recsys/models/rules.py -- lech thi canh bao.
2. BANG CHINH tren test: B6 canh B0, B1, B4, kem CI 95%, ca lay mau lan toan danh muc.
3. CUNG PHIEN hay PHIEN MOI: tach user theo viec phim test co nam trong phien cuoi
   cua train khong. Luat theo phien phai manh hon o nhom "cung phien".
4. PHAM VI CUA LUAT va thien lech pho bien cua top-10.
5. USER MOI (cat lich su, nhu checklist 2.8).
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

from recsys.data import movielens
from recsys.data.split import leave_one_out, truncate_history
from recsys.eval.provenance import save_result
from recsys.eval.runner import MAIN_K, evaluate
from recsys.eval.stats import bootstrap_ci, is_significant, paired_diff_ci
from recsys.models import rules
from recsys.models.baselines import Popularity
from recsys.models.bpr import BPR
from recsys.models.knn import K_ITEM, ItemKNN

K = MAIN_K
SEED = 42
OUT_DIR = Path("reports")
N_KEEP = (1, 3, 5, 10)
N_FOLDS = 4

GAPS = (1800, 86400)
#: Do thu ngay 19/11/2026: ndcg@10 validation tang deu khi ha min_support (0.01 ->
#: 0.0005: 0.21 -> 0.49, khoang nghi 30 phut). 0.0005 la "cap phim chung it nhat 2
#: phien" -- thap hon nua la dem moi cap dong xuat hien, khong con la khai pha luat.
SUPPORTS = (0.005, 0.002, 0.001, 0.0005)
CONTEXTS: tuple = (1, 3, 10, "session")
METRICS = ("confidence", "lift")
AGGS = ("max", "sum")
#: max_len = 3 do o min_support = max(gia tri da chon, nguong nay). Do thu ngay
#: 19/11/2026, khoang nghi 30 phut: o 0,002 max_len 3 sinh 6,3 trieu luat, khai pha
#: 455 giay, ndcg@10 validation 0.4143 so voi 0.4138 cua max_len 2. Khoang nghi 1 ngay
#: con nhieu hon. Nen so sanh max_len 2 va 3 o muc support re hon.
MAX_LEN_3_TU = 0.005

POP, IKNN, BPR4, B6 = "B0-Popularity", "B1-ItemKNN", "B4-BPR", "B6-Rules"


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

    def do(m, gap, ms, max_len, giay_khai_pha):
        r = evaluate(m, val)
        rows.append({"session_gap": gap, "min_support": ms, "max_len": max_len,
                     "context": str(m.context), "metric": m.metric, "agg": m.agg,
                     "so_luat": len(m.cons_), f"ndcg@{K}": r[f"ndcg@{K}"],
                     f"recall@{K}": r[f"recall@{K}"], "vo_cuc": r["vo_cuc"],
                     "giay_khai_pha": round(giay_khai_pha, 1)})
        log(f"  gap={gap:<6} ms={ms:<6} len={max_len} ctx={m.context!s:<8} "
            f"{m.metric:<10} {m.agg:<3} luat={len(m.cons_):<7} ndcg={r[f'ndcg@{K}']:.4f}")
        m._cache.clear()  # ~48 MB moi mo hinh; giu lai 6 mo hinh thi ton bo nho

    def khai_pha(gap, ms, max_len=2):
        t = time.time()
        m = rules.AssociationRules(session_gap=gap, min_support=ms, max_len=max_len,
                                   context="session", metric="confidence", agg="max")
        m.fit(val.train)
        return m, time.time() - t

    # a. gap x min_support, cham diem mac dinh.
    mined = {}
    for gap in GAPS:
        for ms in SUPPORTS:
            m, t = mined[gap, ms] = khai_pha(gap, ms)
            do(m, gap, ms, 2, t)
    g = pd.DataFrame(rows)
    a = g.loc[g[f"ndcg@{K}"].idxmax()]
    gap_b, ms_b = int(a["session_gap"]), float(a["min_support"])

    # b. context x metric x agg tren luat da khai pha.
    m, t = mined.pop((gap_b, ms_b))
    mined.clear()
    for ctx in CONTEXTS:
        for met in METRICS:
            for agg in AGGS:
                if (ctx, met, agg) != ("session", "confidence", "max"):
                    do(m.rescore(context=ctx, metric=met, agg=agg), gap_b, ms_b, 2, t)
    g = pd.DataFrame(rows)
    b = g.loc[g[f"ndcg@{K}"].idxmax()]
    ctx_b = b["context"] if b["context"] == "session" else int(b["context"])

    # c. max_len 3 so voi 2, cung khoang nghi, cung cach cham diem.
    ms_c = max(ms_b, MAX_LEN_3_TU)
    for ml in (2, 3):
        mc, tc = khai_pha(gap_b, ms_c, max_len=ml)
        do(mc.rescore(context=ctx_b, metric=b["metric"], agg=b["agg"]), gap_b, ms_c, ml, tc)
        del mc

    grid = pd.DataFrame(rows)
    _save(grid.set_index(["session_gap", "min_support", "max_len", "context", "metric",
                          "agg"]).round(4), "exp_rules_1_validation.csv")

    print("### a. Khoảng nghỉ cắt phiên và min_support\n")
    print("Ngữ cảnh là phiên cuối, điểm là confidence lớn nhất, max_len = 2.\n")
    print("| khoảng nghỉ | min_support | số luật | ndcg@10 | recall@10 | ứng viên −inf "
          "| giây khai phá |")
    print("|---|---|---|---|---|---|---|")
    for _, r in grid.iloc[: len(GAPS) * len(SUPPORTS)].iterrows():
        print(f"| {r['session_gap']} giây | {r['min_support']:g} | {r['so_luat']:,} "
              f"| {r[f'ndcg@{K}']:.4f} | {r[f'recall@{K}']:.4f} | {r['vo_cuc']:.1%} "
              f"| {r['giay_khai_pha']:.0f} |")

    print(f"\n### b. Cách chấm điểm (khoảng nghỉ {gap_b} giây, min_support {ms_b:g})\n")
    sub = grid[(grid["session_gap"] == gap_b) & (grid["min_support"] == ms_b)
               & (grid["max_len"] == 2)]
    cot = [(met, agg) for met in METRICS for agg in AGGS]
    print("| ngữ cảnh | " + " | ".join(f"{met}, {agg}" for met, agg in cot) + " |")
    print("|---|" + "---|" * len(cot))
    for ctx in CONTEXTS:
        cells = []
        for met, agg in cot:
            v = sub[(sub["context"] == str(ctx)) & (sub["metric"] == met)
                    & (sub["agg"] == agg)][f"ndcg@{K}"].iloc[0]
            cells.append(f"**{v:.4f}**" if v == sub[f"ndcg@{K}"].max() else f"{v:.4f}")
        ten = "phiên cuối" if ctx == "session" else f"{ctx} phim gần nhất"
        print(f"| {ten} | " + " | ".join(cells) + " |")

    print(f"\n### c. max_len 3 so với 2 (min_support {ms_c:g}, cách chấm điểm ở b)\n")
    print("| max_len | số luật | ndcg@10 | giây khai phá |")
    print("|---|---|---|---|")
    for _, r in grid.iloc[-2:].iterrows():
        print(f"| {r['max_len']} | {r['so_luat']:,} | {r[f'ndcg@{K}']:.4f} "
              f"| {r['giay_khai_pha']:.0f} |")

    best = grid.loc[grid[f"ndcg@{K}"].idxmax()]
    chon = {"session_gap": int(best["session_gap"]),
            "min_support": float(best["min_support"]),
            "max_len": int(best["max_len"]),
            "context": best["context"] if best["context"] == "session" else int(best["context"]),
            "metric": best["metric"], "agg": best["agg"]}
    print(f"\nChọn: {chon}, ndcg@10 validation {best[f'ndcg@{K}']:.4f}.")

    da_ghi = {"session_gap": rules.SESSION_GAP, "min_support": rules.MIN_SUPPORT,
              "max_len": rules.MAX_LEN, "context": rules.CONTEXT,
              "metric": rules.METRIC, "agg": rules.AGG}
    if chon != da_ghi:
        log(f"CANH BAO: rules.py dang ghi {da_ghi}. Sua hang so trong recsys/models/rules.py.")
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
    _save(pd.DataFrame(rows).set_index("mo_hinh").round(4), "exp_rules_2_test.csv")

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
    for metric in (f"ndcg@{K}", f"recall@{K}"):
        for b in (POP, IKNN, BPR4):
            c1 = fmt_diff(paired_diff_ci(pu[B6][metric], pu[b][metric]))
            c2 = fmt_diff(paired_diff_ci(pu_full[B6][metric], pu_full[b][metric]))
            print(f"| {B6} − {b} | {metric} | {c1} | {c2} |")

    print(f"\nToàn danh mục, ndcg@{K}: " + ", ".join(
        f"{n} {fmt_ci(bootstrap_ci(pu_full[n][f'ndcg@{K}']))}" for n in models))
    return pu


# ------------------------------------------------------------ 3. cung phien --


def cung_phien(split, ratings, models, pu) -> None:
    gap = models[B6].session_gap
    print(f"\n## 3. Phim test cùng phiên hay phiên mới (khoảng nghỉ {gap} giây)\n")
    t_test = ratings.set_index(["user_id", "item_id"])["timestamp"]
    t_cuoi = split.train.groupby("user_id")["timestamp"].max()
    cach = pd.Series({u: t_test[(u, split.test[u])] - t_cuoi[u] for u in split.users})
    nhom = np.where(cach <= gap, "cung_phien", "phien_moi")
    nhom = pd.Series(nhom, index=cach.index)
    rows = []
    for g in ("cung_phien", "phien_moi"):
        users = nhom.index[nhom == g]
        for n in models:
            rows.append({"nhom": g, "mo_hinh": n, "so_user": len(users),
                         f"ndcg@{K}": pu[n].loc[users, f"ndcg@{K}"].mean()})
    _save(pd.DataFrame(rows).set_index(["nhom", "mo_hinh"]).round(4),
          "exp_rules_3_cung_phien.csv", session_gap=gap)

    print(f"ndcg@{K}, lấy mẫu, CI 95%. \"Cùng phiên\": phim test cách rating cuối trong train "
          f"không quá {gap} giây.\n")
    print("| nhóm | số user | " + " | ".join(models) + " | B6 − ItemKNN | B6 − BPR |")
    print("|---|---|" + "---|" * (len(models) + 2))
    for g, ten in (("cung_phien", "cùng phiên"), ("phien_moi", "phiên mới")):
        users = nhom.index[nhom == g]
        cells = " | ".join(fmt_ci(bootstrap_ci(pu[n].loc[users, f"ndcg@{K}"])) for n in models)
        diffs = " | ".join(fmt_diff(paired_diff_ci(pu[B6].loc[users, f"ndcg@{K}"],
                                                   pu[b].loc[users, f"ndcg@{K}"]))
                           for b in (IKNN, BPR4))
        print(f"| {ten} | {len(users)} | {cells} | {diffs} |")


# ---------------------------------------------------------------- 4. pham vi --


def pham_vi(split, models) -> None:
    print("\n## 4. Phạm vi của luật và thiên lệch phổ biến của top-10\n")
    m = models[B6]
    dem = split.train["item_id"].value_counts()
    ve_phai = set(m.items_[np.unique(m.cons_)].tolist())
    n_items = split.train["item_id"].nunique()
    print(f"{len(m.cons_):,} luật trên {m.n_sessions_:,} phiên có từ 2 phim. "
          f"{len(ve_phai):,}/{n_items:,} phim ({len(ve_phai) / n_items:.1%}) làm vế phải "
          f"của ít nhất một luật. Lượt xem trung vị của các phim đó: "
          f"{np.median([dem[i] for i in ve_phai]):.0f}.\n")

    top100 = set(dem.index[:100])
    rows = []
    for name, mm in models.items():
        recs = [mm.recommend(u, k=K) for u in split.users]
        phim = [i for r in recs for i in r]
        rows.append({"mo_hinh": name,
                     "user_nhan_du_10": float(np.mean([len(r) == K for r in recs])),
                     "so_phim_khac_nhau": len(set(phim)),
                     "phu_danh_muc": len(set(phim)) / n_items,
                     "luot_xem_trung_vi": float(np.median([dem.get(i, 0) for i in phim])),
                     "ty_le_top100": float(np.mean([i in top100 for i in phim]))})
    df = pd.DataFrame(rows).set_index("mo_hinh")
    _save(df.round(4), "exp_rules_4_pham_vi.csv")
    print("| Mô hình | user nhận đủ 10 phim | số phim khác nhau | phủ danh mục "
          "| lượt xem trung vị | tỉ lệ thuộc top-100 phổ biến |")
    print("|---|---|---|---|---|---|")
    for name, r in df.iterrows():
        print(f"| {name} | {r['user_nhan_du_10']:.1%} | {r['so_phim_khac_nhau']:.0f} "
              f"| {r['phu_danh_muc']:.1%} | {r['luot_xem_trung_vi']:.0f} | {r['ty_le_top100']:.1%} |")


# --------------------------------------------------------------- 5. user moi --


def user_moi(split, chon, pu_full_hist) -> None:
    print("\n## 5. User mới: cắt còn n tương tác gần nhất (như checklist 2.8)\n")
    names = [POP, BPR4, B6]
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(np.array(split.users)), N_FOLDS)
    pu: dict[tuple[str, str], pd.Series] = {}
    rows = []
    parts: dict[tuple[int, str], list] = {(n, nm): [] for n in N_KEEP for nm in names}
    for fold in folds:
        # B6: khai pha MOT lan tren train da bo han user cua fold, roi fold_in lich su
        # da cat cua ho cho tung n. Khai pha lai cho moi (fold, n) mat ~2 gio o
        # min_support 0,0005. Khac fit lai o cho n phim giu lai cua user trong fold
        # khong gop vao support -- toi da 10 phim x 153 user tren ~100.000 dong.
        b6 = rules.AssociationRules(**chon)
        b6.fit(split.train[~split.train["user_id"].isin(fold)])
        for n in N_KEEP:
            cut = truncate_history(split, fold, n)
            cut = replace(cut, test={int(u): split.test[int(u)] for u in fold})
            for nm, m in ((POP, Popularity()), (BPR4, BPR())):
                m.fit(cut.train)
                parts[n, nm].append(evaluate(m, cut, keep_per_user=True)["per_user"])
            b6.fold_in(cut.train[cut.train["user_id"].isin(fold)])
            parts[n, B6].append(evaluate(b6, cut, keep_per_user=True)["per_user"])
        log("  user moi: xong mot fold")
    for n in N_KEEP:
        for nm in names:
            df = pd.concat(parts[n, nm])
            pu[nm, str(n)] = df[f"ndcg@{K}"]
            rows.append({"n_giu": n, "mo_hinh": nm, f"ndcg@{K}": df[f"ndcg@{K}"].mean()})
    for nm in names:
        pu[nm, "day_du"] = pu_full_hist[nm][f"ndcg@{K}"]
        rows.append({"n_giu": "day_du", "mo_hinh": nm,
                     f"ndcg@{K}": pu_full_hist[nm][f"ndcg@{K}"].mean()})
    _save(pd.DataFrame(rows).set_index(["n_giu", "mo_hinh"]).round(4),
          "exp_rules_5_user_moi.csv", n_folds=N_FOLDS)

    print(f"ndcg@{K}, CI 95%, đủ {len(split.users)} user.\n")
    print("| giữ lại | " + " | ".join(names) + " | B6 − Popularity | B6 − BPR |")
    print("|---|" + "---|" * (len(names) + 2))
    for lv in [str(n) for n in N_KEEP] + ["day_du"]:
        cells = " | ".join(fmt_ci(bootstrap_ci(pu[nm, lv])) for nm in names)
        diffs = " | ".join(fmt_diff(paired_diff_ci(pu[B6, lv], pu[b, lv])) for b in (POP, BPR4))
        label = "đầy đủ" if lv == "day_du" else f"{lv} tương tác"
        print(f"| {label} | {cells} | {diffs} |")


# ------------------------------------------------------------- 6. explain --


def vi_du(split, model, titles) -> None:
    print("\n## 6. Ví dụ câu giải thích thật\n")
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
    split = leave_one_out(ratings, seed=SEED)
    print(f"# Thí nghiệm B6: luật kết hợp theo phiên\n\nml-latest-small · seed {SEED} · "
          f"{len(split.users)} user kiểm tra · ItemKNN k = {K_ITEM} · BPR theo hằng số bpr.py")

    chon = tim_sieu_tham_so(split.validation())
    models = {POP: Popularity(), IKNN: ItemKNN(k=K_ITEM), BPR4: BPR(),
              B6: rules.AssociationRules(titles=titles, **chon)}
    pu = bang_chinh(split, models)
    cung_phien(split, ratings, models, pu)
    pham_vi(split, models)
    user_moi(split, chon, pu)
    vi_du(split, models[B6], titles)


if __name__ == "__main__":
    main()
