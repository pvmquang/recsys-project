"""Bac B4 -- BPR-MF: xep hang theo CAP, tu phan hoi tiem an (Rendle et al., UAI 2009).

Cung dang diem voi MF: moi user/item mot vector f chieu, them bias cua item:
    x_ui = b_i + p_u . q_i
Khac B3 o MUC TIEU hoc. MF implicit (B3) hoi quy tung o cua ma tran ve 0/1 --
mot bai toan du doan TUNG DIEM. BPR khong quan tam gia tri tuyet doi, chi doi:
    voi moi user u, phim da xem i phai co diem CAO HON phim chua xem j.
Ham muc tieu (BPR-Opt): toi da
    sum_{(u,i,j)} ln sigma(x_ui - x_uj)  -  reg * ||tham so||^2
Day dung la dang lien tuc cua AUC -- tuc thang vao cai giao thuc danh gia do.

=========================================================================
CACH HOC: SGD THEO LO NHO (mini-batch), LAY MAU PHIM AM DEU
Moi epoch duyet het cac cap (u, i) da xem theo thu tu xao tron. Moi cap ghep
voi mot phim j lay deu trong danh muc, lay lai neu u da xem j. Mot lo B cap
tinh gradient cung luc bang numpy; cac dong trung nhau trong lo duoc CONG
(np.add.at), giong chay B buoc SGD tuan tu nhung khong co vong lap Python.

De xuat de tai ghi "B3 MF-SGD, B4 BPR-MF". B3 da doi sang ALS (xem mf.py); B4
van la SGD vi BPR khong co nghiem dong -- ham sigma khong cho tach thanh hoi
quy ridge nhu MF implicit.
=========================================================================

GIAI THICH: KHONG CHINH XAC NHU B3
Voi MF implicit, p_u la ham tuyen tinh cua cac phim da xem nen diem tach dung
thanh dong gop tung phim. O BPR, p_u la tham so tu do hoc bang SGD, khong co
dang thuc do. explain() vi vay la giai thich HAU KIEM: nen ba phim da xem co
vector gan q_i nhat (cosine). No noi "phim nay giong nhung phim ban da xem",
khong noi "diem nay den tu dau". Bao cao phai ghi ro su khac nhau nay.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit

from recsys.base import Recommender
from recsys.models.knn import build_matrix

#: Sieu tham so CHON TREN VALIDATION (checklist 2.1). Xem scripts/exp_bpr.py --
#: script do chay lai phep chon va canh bao neu ket qua khac so ghi o day.
#: ml-latest-small, seed 42, ndcg@10 tren validation (docs/so-lieu-b4.md):
#:   f=64 lr=0.1 reg=0.01, 80 epoch -> 0.4928 (f=128, 160 epoch: 0.4927 -- gan nhu bang)
#:   tham chieu MF implicit 0.5118, ItemKNN k=80 0.4768
#: BATCH khong do, giu 1024.
FACTORS = 64
LR = 0.1
REG = 0.01
N_EPOCHS = 80
BATCH = 1024


class BPR(Recommender):
    """B4 -- BPR-MF. Xem docstring cua module."""

    name = "B4-BPR"

    def __init__(
        self,
        factors: int = FACTORS,
        lr: float = LR,
        reg: float = REG,
        n_epochs: int = N_EPOCHS,
        batch_size: int = BATCH,
        seed: int = 0,
        titles: Mapping[int, str] | None = None,
    ):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        #: item_id -> ten phim, chi cho explain(). Truyen vao, khong tu doc file.
        self.titles = titles

    # ------------------------------------------------------------------ fit --

    def _fit(self, train: pd.DataFrame) -> None:
        self.R_, self.users_, self.items_, self.uidx_, self.iidx_ = build_matrix(train)
        n_users, n_items = self.R_.shape
        self.rng_ = np.random.default_rng(self.seed)
        self.P_ = self.rng_.normal(0, 0.1, (n_users, self.factors))
        self.Q_ = self.rng_.normal(0, 0.1, (n_items, self.factors))
        self.b_ = np.zeros(n_items)

        # Cac cap (u, i) da xem, va khoa u * n_items + i da sap xep de kiem tra
        # "u da xem j chua" bang tim kiem nhi phan, khong can tap hop Python.
        self.pos_u_ = np.repeat(np.arange(n_users), np.diff(self.R_.indptr))
        self.pos_i_ = self.R_.indices.copy()
        self.keys_ = np.sort(self.pos_u_.astype(np.int64) * n_items + self.pos_i_)
        #: Trung binh -ln sigma(x_ui - x_uj) cua tung epoch -- phai giam dan.
        self.loss_: list[float] = []
        self.train_more(self.n_epochs)

    def train_more(self, n_epochs: int) -> "BPR":
        """Chay them `n_epochs` epoch tren mo hinh da fit.

        fit(n_epochs=a+b) cho DUNG ket qua nhu fit(n_epochs=a) roi train_more(b):
        cung mot bo sinh so ngau nhien, tieu thu cung thu tu. Script chon so epoch
        dua vao dieu nay -- danh gia sau 10, 20, 40... epoch ma chi huan luyen mot lan.
        """
        for _ in range(n_epochs):
            self.loss_.append(self._epoch())
        return self

    def _sample_negatives(self, u: np.ndarray) -> np.ndarray:
        """Moi user trong `u` mot phim j lay deu, KHONG nam trong phim u da xem."""
        n_items = self.R_.shape[1]
        j = self.rng_.integers(0, n_items, size=len(u))
        bad = self._seen(u, j)
        while bad.any():
            j[bad] = self.rng_.integers(0, n_items, size=int(bad.sum()))
            bad[bad] = self._seen(u[bad], j[bad])
        return j

    def _seen(self, u: np.ndarray, j: np.ndarray) -> np.ndarray:
        key = u.astype(np.int64) * self.R_.shape[1] + j
        pos = np.searchsorted(self.keys_, key)
        pos = np.minimum(pos, len(self.keys_) - 1)
        return self.keys_[pos] == key

    def _epoch(self) -> float:
        order = self.rng_.permutation(len(self.pos_u_))
        P, Q, b = self.P_, self.Q_, self.b_
        lr, reg = self.lr, self.reg
        tong = 0.0
        for lo in range(0, len(order), self.batch_size):
            sel = order[lo : lo + self.batch_size]
            u, i = self.pos_u_[sel], self.pos_i_[sel]
            j = self._sample_negatives(u)
            pu, qi, qj = P[u], Q[i], Q[j]
            x = b[i] - b[j] + np.einsum("bf,bf->b", pu, qi - qj)
            tong -= log_expit(x).sum()
            g = expit(-x)  # d ln sigma(x) / dx
            # Tang theo gradient (toi DA hoa BPR-Opt). Tinh het truoc khi cong
            # de moi so hang dung tham so CU, nhu SGD tuan tu tren tung cap.
            dP = g[:, None] * (qi - qj) - reg * pu
            dQi = g[:, None] * pu - reg * qi
            dQj = -g[:, None] * pu - reg * qj
            np.add.at(P, u, lr * dP)
            np.add.at(Q, i, lr * dQi)
            np.add.at(Q, j, lr * dQj)
            np.add.at(b, i, lr * (g - reg * b[i]))
            np.add.at(b, j, lr * (-g - reg * b[j]))
        return tong / len(order)

    # ---------------------------------------------------------------- score --

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        out = np.full(len(item_ids), -np.inf)
        u = self.uidx_.get(int(user_id))
        if u is None:
            return out  # user chua co trong train: khong co vector, khong bang chung
        pos, idx = [], []
        for p, it in enumerate(item_ids):
            j = self.iidx_.get(int(it))
            if j is not None:
                pos.append(p)
                idx.append(j)
        if not idx:
            return out
        idx = np.asarray(idx)
        out[np.asarray(pos)] = self.b_[idx] + self.Q_[idx] @ self.P_[u]
        return out

    # -------------------------------------------------------------- explain --

    def similar_seen(self, user_id: int, item_id: int) -> dict[int, float]:
        """Cosine giua q_i va vector cua tung phim user DA XEM.

        Hau kiem, KHONG phai phan ra diem -- xem docstring cua module.
        Tra ve {item_id da xem: cosine}.
        """
        u = self.uidx_.get(int(user_id))
        i = self.iidx_.get(int(item_id))
        if u is None or i is None:
            return {}
        idx = self.R_.indices[self.R_.indptr[u] : self.R_.indptr[u + 1]]
        if len(idx) == 0:
            return {}
        Y = self.Q_[idx]
        q = self.Q_[i]
        cos = (Y @ q) / (np.linalg.norm(Y, axis=1) * np.linalg.norm(q) + 1e-12)
        return {int(self.items_[j]): float(c) for j, c in zip(idx, cos)}

    def explain(self, user_id: int, item_id: int) -> str:
        sim = self.similar_seen(user_id, item_id)
        top = [j for j, c in sorted(sim.items(), key=lambda kv: -kv[1]) if c > 0][:3]
        if not top:
            return ""
        ten = ", ".join(self._title(j) for j in top)
        return f"Giong cac phim ban da xem: {ten}"

    def _title(self, item_id: int) -> str:
        if self.titles is None:
            return str(item_id)
        return self.titles.get(item_id, str(item_id))
