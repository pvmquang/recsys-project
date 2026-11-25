"""Thi nghiem cua bac B7 (lai ghep) -- chon sieu tham so tren VALIDATION, bao cao tren TEST.

    python scripts/exp_hybrid.py > docs/so-lieu-b7.md     # ~20-30 phut

Cau hoi cua bac B7: ghep BPR (B4), noi dung (B5) va Popularity (B0) theo cac can cu
da do o B4, B5 thi duoc gi, va mat gi.

1. CHON CACH CHUAN HOA tren validation (checklist 2.1): alpha chung x cach chuan hoa,
   moi user cung alpha.
1b. CHON ALPHA THEO SO TUONG TAC tren validation CO CAT LICH SU: alpha x dang ham
   (hang, nguong, co) x tham so, chon theo trung binh nDCG@10 cua 5 kich ban (con 1,
   3, 5, 10 tuong tac, va lich su day du). Ket qua 1 va 1b phai khop hang so trong
   recsys/models/hybrid.py -- lech thi canh bao.
2. BANG CHINH tren test: B7 canh B0, B3, B4, B5, kem CI 95%, lay mau va toan danh muc.
   Them: 23 user co phim test chua tung xuat hien trong train (checklist 2.6), va
   chenh lech tren cac user con lai de tach phan loi den tu nhom do.
3. DO PHU va THIEN LECH PHO BIEN cua top-10, ke ca ti le phim moi.
4. PHIM MOI (item cold-start): nhu exp_content muc 5, them do toan danh muc -- o do
   ~150 phim moi canh tranh nhau, khong chi mot.
5. USER MOI: cat con n tuong tac, them n = 0 (user chua co lich su). B7 do hai lan
   tren cung mot lan fit: alpha theo lich su (1b) va alpha chung (1).
6. VI DU explain() that cho ca ba nhanh.

Mot lan fit B7 la fit luon BPR, B5, Popularity ben trong; cac bang lay ba mo hinh
do tu chinh B7 (h.cf, h.content, h.fallback) de khoi huan luyen lai.

In Markdown ra stdout, tien do ra stderr. Moi bang ghi mot CSV kem .meta.json.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import movielens, tmdb
from recsys.data.split import leave_one_out, truncate_history
from recsys.eval import metrics
from recsys.eval.provenance import save_result
from recsys.eval.runner import MAIN_K, evaluate
from recsys.eval.stats import bootstrap_ci, is_significant, paired_diff_ci
from recsys.models import hybrid
from recsys.models.mf import MF

K = MAIN_K
SEED = 42
OUT_DIR = Path("reports")
N_KEEP = (0, 1, 3, 5, 10)
N_FOLDS = 4
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
#: Muc 1b. (dang ham, tham so). nguong 20: user lich su day du co >= 18 tuong tac
#: trong validation, nen nguong nay cham ca ho.
RAMP_ALPHAS = (0.05, 0.1, 0.2, 0.3)
RAMP_GRID = (("hang", 0), ("nguong", 3), ("nguong", 5), ("nguong", 10), ("nguong", 20),
             ("co", 1), ("co", 3), ("co", 10), ("co", 30))
N_KEEP_VAL = (1, 3, 5, 10)

POP, MF3, BPR4, B5, B7 = "B0-Popularity", "B3-MF", "B4-BPR", "B5-Content", "B7-Hybrid"
#: B7 voi alpha chung cua muc 1 -- cung mot doi tuong da fit, chi doi tham so.
B7C = "B7 α chung"


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


def thanh_phan(h: hybrid.Hybrid) -> dict:
    """B7 da fit -> {ten: mo hinh} cho ca bon, khong huan luyen lai."""
    return {POP: h.fallback, BPR4: h.cf, B5: h.content, B7: h}


@contextmanager
def cau_hinh(m, cfg: dict | None):
    """Tam doi tham so cua mo hinh da fit (alpha, ramp...), xong tra lai nhu cu."""
    cfg = cfg or {}
    cu = {k: getattr(m, k) for k in cfg}
    for k, v in cfg.items():
        setattr(m, k, v)
    try:
        yield m
    finally:
        for k, v in cu.items():
            setattr(m, k, v)


def ndcg_nhanh(model, split) -> pd.Series:
    """ndcg@K theo user, chep DUNG vong lap cua runner.evaluate nhung bo recommend()
    (chi can cho coverage, chiem phan lon thoi gian). Muc 1b so lai voi evaluate()
    mot lan va dung han neu lech."""
    out = {}
    for u in split.users:
        cand = split.candidates(u)
        s = np.asarray(model.score(u, cand), dtype=float)
        order = np.random.default_rng(u).permutation(len(cand))
        cand, s = cand[order], s[order]
        ranked = cand[np.argsort(-s, kind="stable")].tolist()
        out[u] = metrics.ndcg_at_k(ranked, {int(split.test[u])}, K)
    return pd.Series(out)


# ------------------------------------------------------------------ 1. chon --


def tim_sieu_tham_so(val, meta) -> dict:
    print("\n## 1. Chọn cách chuẩn hóa trên validation, mọi user cùng α (ndcg@10)\n")
    t = time.time()
    h = hybrid.Hybrid(meta, ramp="hang", ramp_n=0).fit(val.train)
    log(f"  fit B7 tren validation ({time.time() - t:.0f}s)")
    rows = []
    for norm in hybrid.NORMS:
        for a in ALPHAS:
            h.alpha, h.norm = a, norm
            r = evaluate(h, val)
            rf = evaluate(h, val, full_catalog=True)
            rows.append({"norm": norm, "alpha": a, f"ndcg@{K}": r[f"ndcg@{K}"],
                         f"recall@{K}": r[f"recall@{K}"],
                         f"ndcg@{K}_toan_danh_muc": rf[f"ndcg@{K}"],
                         f"coverage@{K}": r[f"coverage@{K}"]})
            log(f"  norm={norm:<4} alpha={a:<4} ndcg={r[f'ndcg@{K}']:.4f} "
                f"toan={rf[f'ndcg@{K}']:.4f}")
    ref = {n: evaluate(m, val)[f"ndcg@{K}"] for n, m in ((BPR4, h.cf), (B5, h.content))}
    grid = pd.DataFrame(rows)
    _save(grid.set_index(["norm", "alpha"]).round(4), "exp_hybrid_1_validation.csv")
    log(f"  chon xong ({time.time() - t:.0f}s)")

    print("Chọn theo ndcg@10 lấy mẫu, như mọi bậc trước. Cột toàn danh mục chỉ để tham "
          "khảo. Ô đậm là tốt nhất của từng cột.\n")
    cols = [f"ndcg@{K}", f"ndcg@{K}_toan_danh_muc", f"coverage@{K}"]
    best = {c: grid[c].max() for c in cols}
    print("| chuẩn hóa | alpha | ndcg@10 lấy mẫu | ndcg@10 toàn danh mục | coverage@10 |")
    print("|---|---|---|---|---|")
    for _, r in grid.iterrows():
        cells = [f"**{r[c]:.4f}**" if r[c] == best[c] else f"{r[c]:.4f}" for c in cols]
        print(f"| {r['norm']} | {r['alpha']:g} | " + " | ".join(cells) + " |")
    print(f"\nTham chiếu cùng validation, ndcg@10 lấy mẫu: BPR {ref[BPR4]:.4f}, "
          f"B5 {ref[B5]:.4f}.")

    b = grid.loc[grid[f"ndcg@{K}"].idxmax()]
    chon = {"alpha": float(b["alpha"]), "norm": str(b["norm"]), "ramp": "hang", "ramp_n": 0}
    print(f"\nChọn: chuẩn hóa {chon['norm']}, α chung {chon['alpha']:g}, ndcg@10 validation "
          f"{b[f'ndcg@{K}']:.4f}. Cách chuẩn hóa này giữ cho mục 1b.")
    if chon["norm"] != hybrid.NORM:
        log(f"CANH BAO: hybrid.py dang ghi NORM = {hybrid.NORM!r}. Sua trong "
            "recsys/models/hybrid.py.")
    return chon


# ------------------------------------------------------- 1b. alpha theo lich su --


def _folds(split):
    rng = np.random.default_rng(SEED)
    return np.array_split(rng.permutation(np.array(split.users)), N_FOLDS)


def alpha_theo_lich_su(val, meta, chung: dict) -> dict:
    print("\n## 1b. α theo số tương tác, chọn trên validation có cắt lịch sử (ndcg@10)\n")
    t = time.time()
    norm = chung["norm"]
    cfgs = [{"alpha": a, "ramp": r, "ramp_n": n} for a in RAMP_ALPHAS for r, n in RAMP_GRID]
    key = lambda c: (c["alpha"], c["ramp"], c["ramp_n"])  # noqa: E731
    parts: dict[tuple, dict[str, list]] = {key(c): {} for c in cfgs}
    ref_parts: dict[str, list] = {}  # BPR cung kich ban, de tham chieu

    def do(h, cut, muc):
        for c in cfgs:
            with cau_hinh(h, c):
                parts[key(c)].setdefault(muc, []).append(ndcg_nhanh(h, cut))
        ref_parts.setdefault(muc, []).append(ndcg_nhanh(h.cf, cut))

    # Lich su day du: mot lan fit. So ndcg_nhanh voi evaluate() o day.
    h = hybrid.Hybrid(meta, norm=norm).fit(val.train)
    with cau_hinh(h, cfgs[0]):
        that = evaluate(h, val, keep_per_user=True)["per_user"][f"ndcg@{K}"]
        nhanh = ndcg_nhanh(h, val)
    if not np.allclose(that.sort_index().to_numpy(), nhanh.sort_index().to_numpy()):
        raise RuntimeError("ndcg_nhanh lech evaluate() -- sua ndcg_nhanh theo runner.py")
    do(h, val, "day_du")
    log(f"  1b: day du xong ({time.time() - t:.0f}s)")

    for n in N_KEEP_VAL:
        for fold in _folds(val):
            cut = truncate_history(val, fold, n)
            cut = replace(cut, test={int(u): val.test[int(u)] for u in fold})
            do(hybrid.Hybrid(meta, norm=norm).fit(cut.train), cut, str(n))
        log(f"  1b: n = {n} xong ({time.time() - t:.0f}s)")

    muc_list = [str(n) for n in N_KEEP_VAL] + ["day_du"]
    rows = []
    for c in cfgs:
        p = parts[key(c)]
        row = {**c, **{m: float(pd.concat(p[m]).mean()) for m in muc_list}}
        row["trung_binh"] = float(np.mean([row[m] for m in muc_list]))
        rows.append(row)
    grid = pd.DataFrame(rows)
    ref = {m: float(pd.concat(ref_parts[m]).mean()) for m in muc_list}
    _save(grid.set_index(["alpha", "ramp", "ramp_n"]).round(4),
          "exp_hybrid_1b_alpha_theo_lich_su.csv", norm=norm, n_folds=N_FOLDS)

    print(f"Chuẩn hóa {norm}. Mỗi kịch bản cắt lịch sử của từng 1/{N_FOLDS} số user validation "
          "xuống n tương tác gần nhất, như mục 5 nhưng trên validation. Chọn theo cột trung "
          "bình (5 kịch bản, trọng số bằng nhau). Ô đậm là tốt nhất của từng cột.\n")
    print("Dạng hàm: `hang` = mọi user cùng α; `nguong m` = α bằng 0 khi dưới m tương tác; "
          "`co c` = α · n / (n + c).\n")
    cols = muc_list + ["trung_binh"]
    best = {c: grid[c].max() for c in cols}
    print("| α | dạng | " + " | ".join(f"{m} tương tác" for m in muc_list[:-1])
          + " | đầy đủ | trung bình |")
    print("|---|---|" + "---|" * len(cols))
    for _, r in grid.iterrows():
        dang = "hang" if r["ramp"] == "hang" else f"{r['ramp']} {r['ramp_n']:g}"
        cells = [f"**{r[c]:.4f}**" if r[c] == best[c] else f"{r[c]:.4f}" for c in cols]
        print(f"| {r['alpha']:g} | {dang} | " + " | ".join(cells) + " |")
    print("| BPR | | " + " | ".join(f"{ref[m]:.4f}" for m in muc_list)
          + f" | {np.mean(list(ref.values())):.4f} |")

    b = grid.loc[grid["trung_binh"].idxmax()]
    chon = {"alpha": float(b["alpha"]), "norm": norm, "ramp": str(b["ramp"]),
            "ramp_n": int(b["ramp_n"])}
    print(f"\nChọn: α {chon['alpha']:g}, {chon['ramp']} {chon['ramp_n']}, trung bình "
          f"{b['trung_binh']:.4f}.")
    log(f"  1b xong ({time.time() - t:.0f}s)")
    da_ghi = {"alpha": float(hybrid.ALPHA), "norm": hybrid.NORM, "ramp": hybrid.RAMP,
              "ramp_n": int(hybrid.RAMP_N)}
    if chon != da_ghi:
        log(f"CANH BAO: hybrid.py dang ghi {da_ghi}. Sua ALPHA, RAMP, RAMP_N trong "
            "recsys/models/hybrid.py.")
    return chon


# ------------------------------------------------------------------ 2. test --


def bang_chinh(split, models: dict, cfg: dict) -> dict:
    """cfg: ten -> tham so tam dat truoc khi do (B7C dung chung doi tuong voi B7)."""
    print("\n## 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)\n")
    pu, pu_full, rows = {}, {}, []
    for name, m in models.items():
        log(f"  danh gia {name}")
        with cau_hinh(m, cfg.get(name)):
            r = evaluate(m, split, keep_per_user=True)
            rf = evaluate(m, split, keep_per_user=True, full_catalog=True)
        pu[name] = r.pop("per_user")
        pu_full[name] = rf.pop("per_user")
        rows.append({"mo_hinh": name, **{c: r[c] for c in (
            f"recall@{K}", f"ndcg@{K}", "mrr", "auc", "vo_cuc", f"coverage@{K}")},
            f"ndcg@{K}_toan_danh_muc": rf[f"ndcg@{K}"],
            f"recall@{K}_toan_danh_muc": rf[f"recall@{K}"]})
    _save(pd.DataFrame(rows).set_index("mo_hinh").round(4), "exp_hybrid_2_test.csv")

    print(f"| Mô hình | recall@{K} | ndcg@{K} | AUC | coverage@{K} | ứng viên −inf |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        p = pu[r["mo_hinh"]]
        print(f"| {r['mo_hinh']} | {fmt_ci(bootstrap_ci(p[f'recall@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(p[f'ndcg@{K}']))} | {fmt_ci(bootstrap_ci(p['auc']))} "
              f"| {r[f'coverage@{K}']:.4f} | {r['vo_cuc']:.1%} |")
    print(f"\nCoverage của B7 chia cho {len(models[B7].all_items_):,} phim (danh mục đã mở "
          f"rộng), của các mô hình khác chia cho {len(models[BPR4].all_items_):,} phim có "
          "trong train.")

    print("\n### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục\n")
    print("| So sánh | độ đo | lấy mẫu | toàn danh mục |")
    print("|---|---|---|---|")
    for metric in (f"ndcg@{K}", f"recall@{K}", "auc"):
        for b in (POP, MF3, BPR4, B5, B7C):
            c1 = fmt_diff(paired_diff_ci(pu[B7][metric], pu[b][metric]))
            c2 = fmt_diff(paired_diff_ci(pu_full[B7][metric], pu_full[b][metric]))
            print(f"| {B7} − {b} | {metric} | {c1} | {c2} |")

    print(f"\nToàn danh mục, ndcg@{K}: " + ", ".join(
        f"{n} {fmt_ci(bootstrap_ci(pu_full[n][f'ndcg@{K}']))}" for n in models))

    trong_train = set(split.train["item_id"])
    la = [u for u in split.users if split.test[u] not in trong_train]
    print(f"\n### {len(la)} user có phim test chưa từng xuất hiện trong train (checklist 2.6)\n")
    nhanh = pd.Series([models[B7].branch(u, split.test[u]) for u in la]).value_counts()
    print("Nhánh B7 chấm phim test của nhóm này: "
          + ", ".join(f"{k} {v}" for k, v in nhanh.items()) + ".\n")
    print("Số user trúng top-10 trong nhóm này:\n")
    print("| Mô hình | lấy mẫu | toàn danh mục |")
    print("|---|---|---|")
    for n in models:
        h1 = int((pu[n].loc[la, f"recall@{K}"] > 0).sum())
        h2 = int((pu_full[n].loc[la, f"recall@{K}"] > 0).sum())
        print(f"| {n} | {h1}/{len(la)} | {h2}/{len(la)} |")

    # Tach phan loi den tu 23 user tren: o do BPR cham phim dung -inf nen AUC ~ 0.
    con = [u for u in split.users if u not in set(la)]
    print(f"\nTrên {len(con)} user còn lại (phim test đã có người xem), lấy mẫu:\n")
    for metric in (f"ndcg@{K}", "auc"):
        for b in (MF3, BPR4, B5, B7C):
            d = paired_diff_ci(pu[B7].loc[con, metric], pu[b].loc[con, metric])
            print(f"- {B7} − {b}, {metric}: {fmt_diff(d)}")
    return pu


# ---------------------------------------------------------------- 3. pho bien --


def pho_bien(split, models: dict, cfg: dict) -> None:
    print("\n## 3. Độ phủ và thiên lệch phổ biến của top-10 (mọi user)\n")
    dem = split.train["item_id"].value_counts()
    top100 = set(dem.index[:100])
    trong_train = set(dem.index)
    n_all = len(models[B7].all_items_)
    rows = []
    for name, m in models.items():
        with cau_hinh(m, cfg.get(name)):
            recs = [m.recommend(u, k=K) for u in split.users]
        phim = [i for r in recs for i in r]
        rows.append({"mo_hinh": name,
                     "so_phim_khac_nhau": len(set(phim)),
                     "phu_danh_muc": len(set(phim)) / n_all,
                     "luot_xem_trung_vi": float(np.median([dem.get(i, 0) for i in phim])),
                     "ty_le_top100": float(np.mean([i in top100 for i in phim])),
                     "ty_le_phim_moi": float(np.mean([i not in trong_train for i in phim])),
                     "so_phim_moi_khac_nhau": len({i for i in phim if i not in trong_train})})
    df = pd.DataFrame(rows).set_index("mo_hinh")
    _save(df.round(4), "exp_hybrid_3_pho_bien.csv")
    print(f"Phủ danh mục tính trên cùng mẫu số {n_all:,} phim cho mọi mô hình. "
          f"Có {n_all - len(trong_train)} phim chưa ai xem trong train.\n")
    print("| Mô hình | số phim khác nhau | phủ danh mục | lượt xem trung vị "
          "| tỉ lệ thuộc top-100 phổ biến | tỉ lệ phim mới | số phim mới khác nhau |")
    print("|---|---|---|---|---|---|---|")
    for name, r in df.iterrows():
        print(f"| {name} | {r['so_phim_khac_nhau']:.0f} | {r['phu_danh_muc']:.1%} "
              f"| {r['luot_xem_trung_vi']:.0f} | {r['ty_le_top100']:.1%} "
              f"| {r['ty_le_phim_moi']:.2%} | {r['so_phim_moi_khac_nhau']:.0f} |")


# ---------------------------------------------------------------- 4. phim moi --


def phim_moi(split, meta, chon) -> None:
    print("\n## 4. Phim mới: phim test bị xóa khỏi train (item cold-start)\n")
    names = [POP, BPR4, B5, B7]
    parts = {(n, g): [] for n in names for g in ("mau", "toan")}
    xoa = 0
    for fold in _folds(split):
        targets = {int(split.test[int(u)]) for u in fold}
        keep = ~split.train["item_id"].isin(targets)
        xoa += int((~keep).sum())
        cut = replace(split, train=split.train[keep],
                      test={int(u): split.test[int(u)] for u in fold})
        h = hybrid.Hybrid(meta, **chon).fit(cut.train)
        for nm, m in thanh_phan(h).items():
            parts[nm, "mau"].append(evaluate(m, cut, keep_per_user=True)["per_user"])
            parts[nm, "toan"].append(
                evaluate(m, cut, keep_per_user=True, full_catalog=True)["per_user"])
        log(f"  phim moi: xong mot phan ({len(targets)} phim moi)")
    pu = {key: pd.concat(v) for key, v in parts.items()}
    rows = [{"mo_hinh": n, "giao_thuc": g, f"ndcg@{K}": pu[n, g][f"ndcg@{K}"].mean(),
             f"recall@{K}": pu[n, g][f"recall@{K}"].mean(), "auc": pu[n, g]["auc"].mean()}
            for n in names for g in ("mau", "toan")]
    _save(pd.DataFrame(rows).set_index(["mo_hinh", "giao_thuc"]).round(4),
          "exp_hybrid_4_phim_moi.csv", n_folds=N_FOLDS, so_dong_xoa=xoa, **chon)
    print(f"Mỗi lần lấy 1/{N_FOLDS} số user, xóa mọi tương tác train với phim test của họ "
          f"(tổng {xoa:,} dòng qua {N_FOLDS} lần), rồi đo trên chính các user đó. Toàn danh "
          "mục: phim đúng cạnh tranh với mọi phim mới khác của cùng lần xóa. "
          f"Đủ {len(split.users)} user, CI 95%.\n")
    print(f"| Mô hình | recall@{K} lấy mẫu | ndcg@{K} lấy mẫu | ndcg@{K} toàn danh mục |")
    print("|---|---|---|---|")
    for n in names:
        print(f"| {n} | {fmt_ci(bootstrap_ci(pu[n, 'mau'][f'recall@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(pu[n, 'mau'][f'ndcg@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(pu[n, 'toan'][f'ndcg@{K}']))} |")
    print()
    for b in (BPR4, B5):
        for g, ten in (("mau", "lấy mẫu"), ("toan", "toàn danh mục")):
            d = paired_diff_ci(pu[B7, g][f"ndcg@{K}"], pu[b, g][f"ndcg@{K}"])
            print(f"- B7 − {b}, ndcg@{K} {ten}: {fmt_diff(d)}")


# --------------------------------------------------------------- 5. user moi --


def user_moi(split, meta, chon, chung, pu_full_hist) -> None:
    print("\n## 5. User mới: cắt còn n tương tác gần nhất (như checklist 2.8)\n")
    names = [POP, BPR4, B7C, B7]
    cfg = {B7C: chung}
    pu: dict[tuple[str, str], pd.Series] = {}
    rows = []
    folds = _folds(split)
    for n in N_KEEP:
        parts: dict[str, list] = {nm: [] for nm in names}
        for fold in folds:
            cut = truncate_history(split, fold, n)
            cut = replace(cut, test={int(u): split.test[int(u)] for u in fold})
            h = hybrid.Hybrid(meta, **chon).fit(cut.train)
            ms = {**thanh_phan(h), B7C: h}
            for nm in names:
                with cau_hinh(ms[nm], cfg.get(nm)):
                    parts[nm].append(evaluate(ms[nm], cut, keep_per_user=True)["per_user"])
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
          "exp_hybrid_5_user_moi.csv", n_folds=N_FOLDS, **chon)

    print(f"ndcg@{K}, CI 95%, đủ {len(split.users)} user. 0 tương tác = user bị xóa khỏi "
          "train, BPR khi đó chấm −inf mọi phim và xếp hạng là xáo ngẫu nhiên. "
          f"{B7C}: cùng lần fit, mọi user dùng α {chung['alpha']:g} (mục 1).\n")
    print("| giữ lại | " + " | ".join(names) + f" | B7 − Popularity | B7 − BPR | B7 − {B7C} |")
    print("|---|" + "---|" * (len(names) + 3))
    for lv in [str(n) for n in N_KEEP] + ["day_du"]:
        cells = " | ".join(fmt_ci(bootstrap_ci(pu[nm, lv])) for nm in names)
        diffs = " | ".join(fmt_diff(paired_diff_ci(pu[B7, lv], pu[b, lv]))
                           for b in (POP, BPR4, B7C))
        label = "đầy đủ" if lv == "day_du" else f"{lv} tương tác"
        print(f"| {label} | {cells} | {diffs} |")


# ------------------------------------------------------------- 6. explain --


def vi_du(split, h, titles) -> None:
    print("\n## 6. Ví dụ câu giải thích thật, mỗi nhánh một ví dụ\n")
    ten = lambda i: titles.get(i, i)  # noqa: E731

    # Nhanh tron: top-1 cua ba user dau.
    shown = 0
    for u in split.users:
        rec = h.recommend(u, k=1)
        if rec and h.branch(u, rec[0]) == hybrid.TRON:
            print(f"- **User {u}**, nhánh trộn → *{ten(rec[0])}*: \"{h.explain(u, rec[0])}\"")
            shown += 1
            if shown == 2:
                break

    # Nhanh noi dung: phim moi xep cao nhat trong top-50 cua mot user.
    trong_train = set(split.train["item_id"])
    for u in split.users:
        moi = [i for i in h.recommend(u, k=50) if i not in trong_train]
        if moi:
            i = moi[0]
            print(f"- **User {u}**, phim chưa ai xem → *{ten(i)}*: \"{h.explain(u, i)}\"")
            break

    # Nhanh pho bien: user khong co trong train.
    u_moi = int(max(split.users)) + 10_000
    rec = h.recommend(u_moi, k=1)
    if rec:
        print(f"- **User chưa có lịch sử** → *{ten(rec[0])}*: \"{h.explain(u_moi, rec[0])}\"")


def main() -> None:
    ratings = movielens.load_ratings()
    movies = movielens.load_movies()
    titles = dict(zip(movies["item_id"], movies["title"]))
    meta = tmdb.load_metadata()
    split = leave_one_out(ratings, seed=SEED)
    print(f"# Thí nghiệm B7: lai ghép\n\nml-latest-small · seed {SEED} · "
          f"{len(split.users)} user kiểm tra · BPR, B5 theo hằng số bpr.py, content.py\n")

    val = split.validation()
    chung = tim_sieu_tham_so(val, meta)
    chon = alpha_theo_lich_su(val, meta, chung)
    t = time.time()
    h = hybrid.Hybrid(meta, titles=titles, **chon).fit(split.train)
    log(f"  fit B7 tren train ({time.time() - t:.0f}s)")
    # MF (B3) hoa BPR ve ndcg@10 (docs/so-lieu-b4.md) nen dua vao bang chinh de so.
    models = {POP: h.fallback, MF3: MF().fit(split.train), BPR4: h.cf, B5: h.content,
              B7C: h, B7: h}
    cfg = {B7C: chung}
    pu = bang_chinh(split, models, cfg)
    pho_bien(split, models, cfg)
    phim_moi(split, meta, chon)
    user_moi(split, meta, chon, chung, pu)
    vi_du(split, h, titles)


if __name__ == "__main__":
    main()
