"""Bo do do danh gia he goi y (bai L8).

Moi ham nhan:
    ranked   -- danh sach item_id DA XEP HANG giam dan theo diem
    relevant -- tap item dung (ground truth)
va tra ve mot so thuc.

=========================================================================
DOC KY CHO NGUOI VIET BAO CAO: BA DO DO DUOI DAY TRUNG NHAU
=========================================================================
Giao thuc cua do an giu lai DUNG MOT item moi user. Khi |relevant| = 1:

    Recall@K == HitRate@K      (ca hai = 1 neu item dung nam trong top-K)
    MAP@K    == MRR cat o K    (AP voi mot item dung = 1/thu_hang)

Chung KHONG phai ba bang chung doc lap. Bay ba cot giong het nhau roi ket
luan "ba do do deu xac nhan mo hinh X tot nhat" la noi qua, va nguoi cham
nhin ra ngay. Bao cao nen:
  - bay Recall@K va nDCG@K la chinh (mot do "co trung khong", mot do "trung
    o vi tri nao"),
  - neu ro trong phan giao thuc rang MAP va MRR trung nhau o thiet ke nay.

Muon ba do do that su doc lap thi phai giu NHIEU item moi user -- doi giao
thuc, va phai doi cho MOI bac mo hinh cung luc.
=========================================================================
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _hits(ranked: Sequence[int], relevant: set[int], k: int) -> np.ndarray:
    """Mang 0/1 do dai k: vi tri i la 1 neu ranked[i] nam trong relevant."""
    top = list(ranked)[:k]
    return np.array([1.0 if item in relevant else 0.0 for item in top])


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """Trong k item goi y, bao nhieu phan tram la dung."""
    if k <= 0:
        return 0.0
    return float(_hits(ranked, relevant, k).sum() / k)


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """Trong so item dung, bao nhieu phan tram lot vao top-k."""
    if not relevant:
        return 0.0
    return float(_hits(ranked, relevant, k).sum() / len(relevant))


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """Co it nhat mot item dung trong top-k khong (1 hoac 0)."""
    return float(_hits(ranked, relevant, k).sum() > 0)


def reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    """1 / thu hang cua item dung DAU TIEN. Khong co thi 0.

    Thu hang dem tu 1, khong phai tu 0 -- dem tu 0 thi item o dinh cho
    reciprocal rank vo cuc.
    """
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def average_precision_at_k(
    ranked: Sequence[int], relevant: set[int], k: int = 10
) -> float:
    """AP@K = trung binh cua P@i tai moi vi tri i co item dung.

    Mau so la min(|relevant|, k) chu khong phai so item dung tim duoc:
    chia cho so tim duoc thi mot mo hinh tim duoc dung 1 item o vi tri 1 se
    dat AP = 1.0 tuyet doi, bang voi mo hinh tim duoc ca 10 item.
    """
    if not relevant:
        return 0.0
    hits = _hits(ranked, relevant, k)
    if hits.sum() == 0:
        return 0.0
    precisions = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precisions * hits).sum() / min(len(relevant), k))


def dcg_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """DCG@K = tong rel_i / log2(i+1), i dem tu 1 (cong thuc slide L8)."""
    hits = _hits(ranked, relevant, k)
    discounts = 1.0 / np.log2(np.arange(1, len(hits) + 1) + 1)
    return float((hits * discounts).sum())


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """nDCG@K = DCG@K / IDCG@K. IDCG = truong hop moi item dung deu o dinh."""
    if not relevant:
        return 0.0
    ideal_hits = np.ones(min(len(relevant), k))
    discounts = 1.0 / np.log2(np.arange(1, len(ideal_hits) + 1) + 1)
    idcg = float((ideal_hits * discounts).sum())
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


def auc(ranked: Sequence[int], relevant: set[int]) -> float:
    """Ti le cap (item dung, item sai) duoc xep dung thu tu.

    Nhan CA danh sach ung vien, khong cat o k -- AUC do chat luong xep hang
    tren toan bo danh sach, cat di la do mot thu khac.
    """
    ranked = list(ranked)
    n = len(ranked)
    n_pos = sum(1 for item in ranked if item in relevant)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Duyet tu tren xuong: moi khi gap item sai, cong so item dung DA gap
    # (= so cap (dung, sai) ma item dung duoc xep trên item sai nay).
    correct_pairs = 0
    pos_so_far = 0
    for item in ranked:
        if item in relevant:
            pos_so_far += 1
        else:
            correct_pairs += pos_so_far
    return correct_pairs / (n_pos * n_neg)


def coverage(all_recommended: Iterable[Iterable[int]], n_items_total: int) -> float:
    """Ti le danh muc tung duoc goi y cho it nhat mot user.

    Day la do do NGOAI DO CHINH XAC quan trong nhat cua do an. Mot mo hinh
    co nDCG cao nhung coverage 3% nghia la no chi biet goi y phim noi tieng
    -- dung ky thuat, nhung vo dung voi mot nen tang thuc te.
    """
    if n_items_total <= 0:
        return 0.0
    seen: set[int] = set()
    for rec in all_recommended:
        seen.update(rec)
    return len(seen) / n_items_total
