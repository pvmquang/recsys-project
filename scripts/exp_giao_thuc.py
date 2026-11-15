"""Thi nghiem cho dot 2 cua checklist -- khong doi giao thuc chung, chi do them.

    python scripts/exp_giao_thuc.py            # ca hai thi nghiem (~5 phut)
    python scripts/exp_giao_thuc.py --chi 2.8  # chi cold-start

2.4  TOAN DANH MUC -- xep hang item dung giua MOI item user chua xem (~9.700),
     thay vi 99 item am lay mau deu. Lay mau deu co loi cho Popularity (Krichene &
     Rendle, KDD 2020); neu thu tu xep hang cac mo hinh doi khi bo lay mau thi
     bang chinh khong dung duoc.
2.5  AUC XU LY THE HOA -- da nam trong runner; o day in them chenh lech AUC kem CI
     va ti le ung vien -inf de doc AUC cho dung.
2.8  COLD-START NHAN TAO -- chia user kiem tra thanh 4 phan; lan luot cat lich su
     train cua MOT phan xuong n tuong tac gan nhat (n = 1, 3, 5, 10), do tren chinh
     phan do. Ghep 4 phan lai thi user nao cung duoc do o moi n, va ghep cap duoc
     voi lan do day du lich su.

Moi thi nghiem ghi CSV trong reports/ va in bang Markdown -- so trong README lay
tu day. Chenh lech nao CI 95% chua 0 thi KHONG viet "mo hinh X thang".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import movielens
from recsys.data.split import leave_one_out, truncate_history
from recsys.eval.provenance import save_result
from recsys.eval.runner import MAIN_K, evaluate
from recsys.eval.stats import bootstrap_ci, is_significant, paired_diff_ci
from recsys.models.baselines import Popularity, RandomRec
from recsys.models.knn import K_ITEM, K_USER, ItemKNN, UserKNN

K = MAIN_K
SEED = 42
OUT_DIR = Path("reports")
N_KEEP = (1, 3, 5, 10)
N_FOLDS = 4

POP, IKNN, UKNN = "B0-Popularity", "B1-ItemKNN", "B2-UserKNN"


def make_models() -> dict:
    return {POP: Popularity(), IKNN: ItemKNN(k=K_ITEM), UKNN: UserKNN(k=K_USER)}


def fmt_ci(ci) -> str:
    mean, lo, hi = ci
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def fmt_diff(ci) -> str:
    mean, lo, hi = ci
    verdict = "**có ý nghĩa**" if is_significant(ci) else "nhiễu"
    return f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict}"


def _save(df: pd.DataFrame, name: str) -> None:
    save_result(df, OUT_DIR / name, seed=SEED, dataset="ml-latest-small", k_item=K_ITEM, k_user=K_USER,
                n_folds=N_FOLDS, n_giu=list(N_KEEP))
    print(f"\nĐã ghi: `{OUT_DIR / name}`")


# --------------------------------------------------------------- 2.4 + 2.5 --


def toan_danh_muc(split) -> None:
    print("\n## 2.4 + 2.5: lấy mẫu 99 item âm so với toàn danh mục")
    models = {"B0-Random": RandomRec(), **make_models()}
    pu: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    for name, model in models.items():
        model.fit(split.train)
        for proto, full in (("lay_mau", False), ("toan_danh_muc", True)):
            r = evaluate(model, split, keep_per_user=True, full_catalog=full)
            pu[name, proto] = r.pop("per_user")
            rows.append({
                "mo_hinh": name, "giao_thuc": proto,
                f"recall@{K}": r[f"recall@{K}"], f"ndcg@{K}": r[f"ndcg@{K}"],
                "mrr": r["mrr"], "auc": r["auc"], "vo_cuc": r["vo_cuc"],
            })
    bang = pd.DataFrame(rows).set_index(["giao_thuc", "mo_hinh"]).round(4)
    _save(bang, "exp_2_4_toan_danh_muc.csv")

    print(f"\n| Mô hình | ndcg@{K} lấy mẫu | ndcg@{K} toàn danh mục | recall@{K} toàn danh mục "
          "| AUC lấy mẫu (hòa = 0.5) | ứng viên −inf |")
    print("|---|---|---|---|---|---|")
    for name in models:
        a, b = pu[name, "lay_mau"], pu[name, "toan_danh_muc"]
        print(f"| {name} | {fmt_ci(bootstrap_ci(a[f'ndcg@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(b[f'ndcg@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(b[f'recall@{K}']))} "
              f"| {fmt_ci(bootstrap_ci(a['auc']))} | {a['vo_cuc'].mean():.1%} |")

    # 2.5: AUC = P(dung > sai) + 0.5 P(hoa). Phim dung -inf thi AUC = 0.5 x ti le
    # item sai cung -inf -- tuc la AUC tron hai cau hoi: "co cham duoc phim dung
    # khong" va "cham roi thi xep co dung khong". Tach ra de doc cho dung.
    print("\n### 2.5: tách AUC (lấy mẫu): chấm được phim đúng không, và chấm rồi thì xếp đúng không\n")
    print("| Mô hình | phim đúng có điểm −inf | AUC khi phim đúng có điểm | AUC khi phim đúng −inf |")
    print("|---|---|---|---|")
    for name in (POP, IKNN, UKNN):
        model = models[name]
        auc_ok, auc_inf = [], []
        for u in split.users:
            c = split.candidates(u)
            s = model.score(u, c)
            a = float(pu[name, "lay_mau"].loc[u, "auc"])
            (auc_inf if np.isneginf(s[0]) else auc_ok).append(a)
        n = len(auc_ok) + len(auc_inf)
        m_inf = f"{np.mean(auc_inf):.4f}" if auc_inf else "không có"
        print(f"| {name} | {len(auc_inf) / n:.1%} ({len(auc_inf)}/{n}) "
              f"| {np.mean(auc_ok):.4f} | {m_inf} |")

    print("\n### Chênh lệch ghép cặp: thứ tự có giữ khi bỏ lấy mẫu không?\n")
    print("| So sánh | độ đo | lấy mẫu | toàn danh mục |")
    print("|---|---|---|---|")
    pairs = [(IKNN, POP), (UKNN, POP), (IKNN, UKNN)]
    for metric in (f"ndcg@{K}", f"recall@{K}", "auc"):
        for a, b in pairs:
            cells = [fmt_diff(paired_diff_ci(pu[a, p][metric], pu[b, p][metric]))
                     for p in ("lay_mau", "toan_danh_muc")]
            print(f"| {a} − {b} | {metric} | {cells[0]} | {cells[1]} |")


# --------------------------------------------------------------------- 2.8 --


def cold_start(split) -> None:
    print("\n## 2.8: cold-start nhân tạo (cắt lịch sử train còn n tương tác gần nhất)")
    rng = np.random.default_rng(SEED)
    users = np.array(split.users)
    folds = np.array_split(rng.permutation(users), N_FOLDS)

    # Moc: lich su day du, do tren CUNG split goc.
    pu: dict[tuple[str, str], pd.Series] = {}
    long_rows = []

    def add(level, name, df):
        pu[name, level] = df[f"ndcg@{K}"]
        long_rows.append({
            "n_giu": level, "mo_hinh": name,
            f"ndcg@{K}": df[f"ndcg@{K}"].mean(),
            f"recall@{K}": df[f"recall@{K}"].mean(),
            "vo_cuc": df["vo_cuc"].mean(),
        })

    for n in N_KEEP:
        parts: dict[str, list[pd.DataFrame]] = {}
        for fold in folds:
            cut = truncate_history(split, fold, n)
            cut = replace(cut, test={int(u): split.test[int(u)] for u in fold})
            for name, model in make_models().items():
                model.fit(cut.train)
                parts.setdefault(name, []).append(
                    evaluate(model, cut, keep_per_user=True)["per_user"]
                )
        for name, ps in parts.items():
            add(str(n), name, pd.concat(ps))
        print(f"  n = {n}: xong", file=sys.stderr)

    for name, model in make_models().items():
        model.fit(split.train)
        add("day_du", name, evaluate(model, split, keep_per_user=True)["per_user"])

    bang = pd.DataFrame(long_rows).set_index(["n_giu", "mo_hinh"]).round(4)
    _save(bang, "exp_2_8_cold_start.csv")

    levels = [str(n) for n in N_KEEP] + ["day_du"]
    names = [POP, IKNN, UKNN]
    print(f"\nndcg@{K}, CI 95%. Mỗi dòng đo trên đủ {len(users)} user "
          f"({N_FOLDS} phần, mỗi lần chỉ cắt một phần).\n")
    print("| giữ lại | " + " | ".join(names)
          + " | ItemKNN − Popularity | UserKNN − Popularity | UserKNN − ItemKNN |")
    print("|---|" + "---|" * (len(names) + 3))
    for lv in levels:
        cells = [fmt_ci(bootstrap_ci(pu[nm, lv])) for nm in names]
        d1 = fmt_diff(paired_diff_ci(pu[IKNN, lv], pu[POP, lv]))
        d2 = fmt_diff(paired_diff_ci(pu[UKNN, lv], pu[POP, lv]))
        d3 = fmt_diff(paired_diff_ci(pu[UKNN, lv], pu[IKNN, lv]))
        label = "đầy đủ" if lv == "day_du" else f"{lv} tương tác"
        print(f"| {label} | " + " | ".join(cells) + f" | {d1} | {d2} | {d3} |")

    print("\nTỉ lệ ứng viên −inf (mô hình không có ý kiến):\n")
    print("| giữ lại | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    vc = bang["vo_cuc"]
    for lv in levels:
        label = "đầy đủ" if lv == "day_du" else f"{lv} tương tác"
        print(f"| {label} | " + " | ".join(f"{vc[lv, nm]:.1%}" for nm in names) + " |")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chi", choices=["2.4", "2.8"], default=None)
    args = p.parse_args()

    ratings = movielens.load_ratings()
    split = leave_one_out(ratings, seed=SEED)
    print(f"# Thí nghiệm giao thức (đợt 2)\n\nml-latest-small · seed {SEED} · "
          f"k láng giềng: ItemKNN {K_ITEM}, UserKNN {K_USER} (chọn trên validation) · "
          f"{len(split.users)} user kiểm tra")
    if args.chi in (None, "2.4"):
        toan_danh_muc(split)
    if args.chi in (None, "2.8"):
        cold_start(split)


if __name__ == "__main__":
    main()
