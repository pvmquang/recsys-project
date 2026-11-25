"""Bac B7 -- LAI GHEP: chuyen theo user, chuyen theo phim, tron co trong so.

Ba nhanh, moi nhanh co can cu DO DUOC o cac bac truoc:

  1. User CHUA CO lich su trong train  -> Popularity (B0).
     Loc cong tac va noi dung deu khong co ho so, moi diem -inf. Popularity la
     bang chung duy nhat con lai.
  2. Phim loc cong tac KHONG CHAM DUOC (phim chua ai xem) -> noi dung (B5).
     Xoa phim test khoi train: Popularity, BPR ve nDCG@10 = 0; B5 van 0.252
     (docs/so-lieu-b5.md, muc phim moi). 23/610 phim test chua tung xuat hien
     trong train (checklist 2.6) -- B0-B4 trung 0/23, B5 trung 5/23.
  3. Con lai -> TRON (1 - a_u) * BPR + a_u * B5, tren thang da chuan hoa. a_u co
     the giam theo so tuong tac cua user (alpha_for, RAMPS).

KHONG chuyen sang Popularity khi user co IT lich su. Can cu cua dot 2 ("Popularity
tot nhat o 1 tuong tac") khong con dung voi BPR: o 1 tuong tac BPR hon Popularity
+0.076 nDCG@10, co y nghia (docs/so-lieu-b4.md). Chuyen o do la lam kem di.

=========================================================================
CHUAN HOA: VI SAO CAN, VA VI SAO THEO TUNG USER, TREN CA DANH MUC
Diem BPR khong gioi han (b_i + p_u . q_i), diem B5 la cosine trong [0, 1]. Cong
thang thi nhanh co thang lon hon thang. Moi nhanh vi vay doi diem sang thang
chung, rieng cho tung user:
    norm="rank"  phan vi cua diem trong MOI phim cua danh muc nhanh do (0..1)
    norm="z"     (diem - trung binh) / do lech chuan, cung tren ca danh muc
Thong ke lay tren CA DANH MUC, khong tren danh sach ung vien dang cham: neu lay
tren ung vien thi diem cua cung mot phim doi theo 99 phim am di kem, va score()
voi recommend() se xep hang khac nhau.

PHIM CHUA AI XEM: diem = diem noi dung da chuan hoa. Doc theo cong thuc tron, day
la thay diem BPR con thieu bang chinh diem noi dung: (1 - a) * n_ct + a * n_ct.
Khong them tham so nao rieng cho phim moi.
He qua do duoc: phim moi chiem 2,07% o top-10 trong khi chi la 0,4% danh muc (41
phim). Gia thuyet, chua do: phim cu ma B5 xep cao thuong it nguoi xem, phan vi BPR
thap keo diem tron cua chung xuong, con phim moi khong bi keo. Tren toan
danh muc dieu nay khong lam B7 kem BPR (docs/so-lieu-b7.md).

Hai dau mut kiem duoc:
    alpha = 0  phim da co nguoi xem xep DUNG nhu BPR (chuan hoa giu thu tu);
               B5 chi con dung cho phim moi. Day la lai ghep chuyen thuan.
    alpha = 1  moi phim xep nhu B5.
=========================================================================

MO RONG DANH MUC
`recommend()` cua hop dong chi duyet `all_items_`, ma fit() gan bang phim CO
trong train -- nen phim moi khong bao gio len giao dien (README, muc B5). _fit()
o day noi them moi phim co dac trung noi dung vao `all_items_`. Khong ghi de
fit() hay recommend(). He qua: mau so coverage cua B7 la 9.742 phim, cua B0-B5 la
so phim trong train (9.701 o test). Lech 0,4%, ghi lai de khong so nham.

GIAI THICH: explain() lay cau cua NHANH DONG GOP NHIEU HON vao diem, va noi ro
khi phim chua ai xem hoac khi user chua co lich su.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from recsys.base import Recommender
from recsys.models.baselines import Popularity
from recsys.models.bpr import BPR
from recsys.models.content import ContentBased

#: Sieu tham so CHON TREN VALIDATION (checklist 2.1). Xem scripts/exp_hybrid.py --
#: script chay lai phep chon va canh bao neu ket qua khac so ghi o day.
#: ml-latest-small, seed 42, ndcg@10 lay mau tren validation (docs/so-lieu-b7.md):
#:   rank, alpha 0.1   -> 0.5161      z, alpha 0.1    -> 0.5117
#:   rank, alpha 0.05  -> 0.5124      rank, alpha 0.2 -> 0.5078
#:   rank, alpha 0     -> 0.4991 (chuyen thuan), alpha 1 -> 0.2571 (= B5)
#:   tham chieu BPR 0.4928
#: alpha chon tren user co lich su day du. O user con 1-3 tuong tac, alpha 0.1 lam
#: B7 KEM BPR co y nghia (-0.030 o 1 tuong tac), vi nhanh noi dung yeu nhat o do.
#: RAMP, RAMP_N: alpha theo so tuong tac (xem Hybrid.alpha_for), chon tren validation
#: co cat lich su (1, 3, 5, 10 tuong tac va day du), theo trung binh nDCG@10 5 kich ban:
#:   alpha 0.1, co 30   -> 0.4816 (alpha_u = 0.1 * n / (n + 30))
#:   alpha 0.05, co 30  -> 0.4811     alpha 0.1, hang -> 0.4720
#:   tham chieu BPR     -> 0.4831 -- CAO HON moi cau hinh B7. Phan thieu khong do alpha:
#:   voi alpha_u = 0 o user bi cat (dong nguong 20), B7 van thap hon BPR 0.003-0.010 o
#:   ca bon muc, vi phim moi duoc xep bang diem noi dung (docs/so-lieu-b7.md).
#: co 30 nam o BIEN cua luoi (1, 3, 10, 30) -- chua thu lon hon.
ALPHA = 0.1
NORM = "rank"
RAMP = "co"
RAMP_N = 30

NORMS = ("rank", "z")
#: hang   alpha_u = alpha, moi user nhu nhau
#: nguong alpha_u = alpha neu n_u >= RAMP_N, nguoc lai 0 (chuyen thuan sang BPR)
#: co     alpha_u = alpha * n_u / (n_u + RAMP_N), tang dan theo lich su
RAMPS = ("hang", "nguong", "co")

#: Ten nhanh, tra ve boi Hybrid.branch().
PHO_BIEN, NOI_DUNG, TRON, KHONG = "pho_bien", "noi_dung", "tron", "khong"


def _percentile(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Phan vi giua (mid-rank) cua x trong ref_sorted. Diem bang nhau -> bang nhau."""
    n = len(ref_sorted)
    if n == 0:
        return np.zeros(len(x))
    lo = np.searchsorted(ref_sorted, x, side="left")
    hi = np.searchsorted(ref_sorted, x, side="right")
    return (lo + hi) / (2.0 * n)


class Hybrid(Recommender):
    """B7 -- lai ghep BPR + noi dung + Popularity. Xem docstring cua module.

    metadata: bang phim cho nhanh noi dung (tmdb.load_metadata()). Truyen vao, mo
        hinh khong tu doc file.
    cf, content, fallback: thay mo hinh thanh phan (de test, hoac thu MF thay BPR).
        Mac dinh BPR(), ContentBased(metadata), Popularity() voi hang so da chon.
    alpha, norm, ramp, ramp_n: doc moi lan score(), nen doi sau khi fit van dung --
        script chon sieu tham so dua vao dieu nay de chi huan luyen mot lan.
    """

    name = "B7-Hybrid"

    def __init__(
        self,
        metadata: pd.DataFrame | None = None,
        alpha: float = ALPHA,
        norm: str = NORM,
        ramp: str = RAMP,
        ramp_n: float = RAMP_N,
        cf: Recommender | None = None,
        content: Recommender | None = None,
        fallback: Recommender | None = None,
        extend_catalog: bool = True,
        titles: Mapping[int, str] | None = None,
    ):
        if norm not in NORMS:
            raise ValueError(f"norm phai la mot trong {NORMS}")
        if ramp not in RAMPS:
            raise ValueError(f"ramp phai la mot trong {RAMPS}")
        if ramp_n < 0:
            raise ValueError("ramp_n phai >= 0")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha phai nam trong [0, 1]")
        if content is None and metadata is None:
            raise ValueError("Can metadata (hoac truyen san mo hinh content)")
        self.alpha = alpha
        self.norm = norm
        self.ramp = ramp
        self.ramp_n = ramp_n
        self.cf = cf if cf is not None else BPR(titles=titles)
        self.content = content if content is not None else ContentBased(metadata, titles=titles)
        self.fallback = fallback if fallback is not None else Popularity()
        self.extend_catalog = extend_catalog

    # ------------------------------------------------------------------ fit --

    def _fit(self, train: pd.DataFrame) -> None:
        self.cf.fit(train)
        self.content.fit(train)
        self.fallback.fit(train)
        self.cf_items_ = np.asarray(self.cf.all_items_)
        self.ct_items_ = self._content_catalog()
        if self.extend_catalog:
            self.all_items_ = np.union1d(self.all_items_, self.ct_items_)
        #: user_id -> so tuong tac trong train, cho alpha_for().
        self.n_hist_ = train.groupby("user_id").size().astype(int).to_dict()
        #: user_id -> thong ke chuan hoa, tinh khi can (xem _ref).
        self.ref_: dict[int, tuple] = {}

    def alpha_for(self, user_id: int) -> float:
        """Trong so noi dung cho user nay, theo so tuong tac trong train (xem RAMPS).

        VI SAO: B5 yeu nhat o user it lich su (nDCG@10 0.160 o 1 tuong tac, BPR 0.437),
        nen mot alpha chung chon tren lich su day du keo B7 xuong duoi BPR o 1-3
        tuong tac (docs/so-lieu-b7.md).
        """
        n = self.n_hist_.get(int(user_id), 0)
        if self.ramp == "nguong":
            return self.alpha if n >= self.ramp_n else 0.0
        if self.ramp == "co":
            return self.alpha * n / (n + self.ramp_n) if n + self.ramp_n > 0 else 0.0
        return self.alpha

    def _content_catalog(self) -> np.ndarray:
        """Moi phim nhanh noi dung cham duoc: co trong metadata va co dac trung."""
        c = self.content
        if hasattr(c, "meta_items_") and hasattr(c, "has_feat_"):
            return np.sort(np.asarray(c.meta_items_)[c.has_feat_])
        return np.asarray(c.all_items_)

    # ---------------------------------------------------------- chuan hoa --

    def _ct_raw(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Diem noi dung, -inf (khong chung dac trung nao) doi thanh 0 = cosine that."""
        s = np.asarray(self.content.score(user_id, item_ids), dtype=float)
        return np.where(np.isfinite(s), s, 0.0)

    def _ref(self, user_id: int) -> tuple:
        """(diem CF da sap, tb, dlc, diem noi dung da sap, tb, dlc) tren ca danh muc."""
        u = int(user_id)
        if u not in self.ref_:
            cf = np.asarray(self.cf.score(u, self.cf_items_), dtype=float)
            cf = np.sort(cf[np.isfinite(cf)])
            ct = np.sort(self._ct_raw(u, self.ct_items_))
            self.ref_[u] = (cf, *self._mean_std(cf), ct, *self._mean_std(ct))
        return self.ref_[u]

    @staticmethod
    def _mean_std(x: np.ndarray) -> tuple[float, float]:
        if len(x) == 0:
            return 0.0, 1.0
        sd = float(x.std())
        return float(x.mean()), (sd if sd > 0 else 1.0)

    def _normalize(self, x: np.ndarray, ref_sorted, mean, std) -> np.ndarray:
        if self.norm == "rank":
            return _percentile(ref_sorted, x)
        return (x - mean) / std

    def _parts(self, user_id: int, item_ids: np.ndarray):
        """(n_cf, n_ct, cf_ok, ct_ok): diem da chuan hoa cua hai nhanh va mat na
        "nhanh do co y kien". n_cf o cho cf_ok = False la vo nghia."""
        items = np.asarray(item_ids)
        cf = np.asarray(self.cf.score(user_id, items), dtype=float)
        ct_full = np.asarray(self.content.score(user_id, items), dtype=float)
        cf_ok, ct_ok = np.isfinite(cf), np.isfinite(ct_full)
        cf_s, cf_m, cf_sd, ct_s, ct_m, ct_sd = self._ref(user_id)
        n_cf = np.zeros(len(items))
        n_cf[cf_ok] = self._normalize(cf[cf_ok], cf_s, cf_m, cf_sd)
        n_ct = self._normalize(np.where(ct_ok, ct_full, 0.0), ct_s, ct_m, ct_sd)
        return n_cf, n_ct, cf_ok, ct_ok

    # ---------------------------------------------------------------- score --

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Diem lai ghep. -inf khi ca ba nhanh deu khong co bang chung.

        User khong co trong train: diem Popularity. Phim CF khong cham duoc: diem
        noi dung da chuan hoa. Con lai: (1 - alpha) * n_cf + alpha * n_ct.
        """
        items = np.asarray(item_ids)
        if int(user_id) not in self.seen_:
            return np.asarray(self.fallback.score(user_id, items), dtype=float)
        out = np.full(len(items), -np.inf)
        if len(items) == 0:
            return out
        n_cf, n_ct, cf_ok, ct_ok = self._parts(user_id, items)
        a = self.alpha_for(user_id)
        tron = cf_ok
        out[tron] = (1 - a) * n_cf[tron] + a * n_ct[tron]
        moi = ~cf_ok & ct_ok
        out[moi] = n_ct[moi]
        return out

    def branch(self, user_id: int, item_id: int) -> str:
        """Nhanh quyet dinh diem cua (user, phim): pho_bien, noi_dung, tron, khong."""
        if int(user_id) not in self.seen_:
            ok = np.isfinite(self.fallback.score(user_id, np.array([item_id]))[0])
            return PHO_BIEN if ok else KHONG
        _, _, cf_ok, ct_ok = self._parts(user_id, np.array([item_id]))
        if cf_ok[0]:
            return TRON
        return NOI_DUNG if ct_ok[0] else KHONG

    # -------------------------------------------------------------- explain --

    def explain(self, user_id: int, item_id: int) -> str:
        nhanh = self.branch(user_id, item_id)
        if nhanh == PHO_BIEN:
            return "Ban chua co lich su xem. " + self.fallback.explain(user_id, item_id)
        if nhanh == NOI_DUNG:
            txt = self.content.explain(user_id, item_id)
            return "Phim moi, chua ai danh gia. " + txt if txt else ""
        if nhanh == KHONG:
            return ""
        n_cf, n_ct, _, _ = self._parts(user_id, np.array([item_id]))
        a = self.alpha_for(user_id)
        cf_phan = (1 - a) * n_cf[0]
        ct_phan = a * n_ct[0]
        chinh, phu = (self.cf, self.content) if cf_phan >= ct_phan else (self.content, self.cf)
        txt = chinh.explain(user_id, item_id)
        return txt or phu.explain(user_id, item_id)
