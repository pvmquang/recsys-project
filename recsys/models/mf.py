"""Bac B3 -- PHAN RA MA TRAN (matrix factorization), huan luyen bang ALS.

Moi user u va moi item i duoc bieu dien bang mot vector f chieu (p_u, q_i).
Diem = tich vo huong p_u . q_i: hai vector cung huong thi user "hop gu" item.
Khac KNN o cho: KNN so sanh truc tiep hai user/hai item qua nhung gi CUNG danh
gia; MF nen ca ma tran ve f "khau vi an" va so sanh qua do -- nen hai phim chua
tung co chung nguoi xem van co the gan nhau.

=========================================================================
HAI CHE DO, VA CHUNG TRA LOI HAI CAU HOI KHAC NHAU (giong sum/weighted cua KNN)

    kind="implicit"  hoc tu viec user CO XEM hay KHONG (Hu, Koren & Volinsky,
                     ICDM 2008). Moi o cua ma tran deu duoc tinh: o da xem la 1
                     voi do tin cay 1 + alpha, o chua xem la 0 voi do tin cay 1.
                     -> xep hang top-K. Khop bai toan DU DOAN TUONG TAC ma nhom
                        da chot (checklist 2.2). MAC DINH.
    kind="explicit"  hoc RATING tren cac o da danh gia, co bias:
                     r_hat = mu + b_u + b_i + p_u . q_i   (Koren, Bell & Volinsky,
                     IEEE Computer 2009 -- "SVD" cua Netflix Prize)
                     -> du doan rating cho MAE/RMSE.

Explicit chi nhin o DA danh gia, nen no khong biet phim nao duoc xem nhieu. Dung
no de xep hang thi mot phim it nguoi xem ma ca hai nguoi deu cham 5 sao se dung
dau. Bao cao nen do CA HAI roi giai thich -- cung bai hoc voi "weighted" cua KNN.
=========================================================================

VI SAO ALS, KHONG PHAI SGD
Co dinh Q thi tim P la bai toan hoi quy ridge co nghiem dong, va nguoc lai. Moi
buoc la mot loat phep giai he f x f bang numpy -- khong can vong lap Python qua
100.000 rating moi epoch, va ket qua xac dinh hoan toan theo seed (khong phu
thuoc thu tu xao du lieu nhu SGD). Voi implicit, ALS con la cach DUY NHAT kha
thi: ma tran co 610 x 9.724 = 5,9 trieu o, SGD phai duyet het, ALS thi khong
(meo YtY ben duoi).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from recsys.base import Recommender
from recsys.models.knn import build_matrix

#: Sieu tham so CHON TREN VALIDATION (checklist 2.1). Xem scripts/exp_mf.py --
#: script do chay lai phep chon va canh bao neu ket qua khac so ghi o day.
#: Mac dinh cua MF() la implicit; explicit chi dung de so sanh.
#: ml-latest-small, seed 42, ndcg@10 tren validation (docs/so-lieu-b3.md):
#:   implicit f=32 reg=10 alpha=10 -> 0.5118 (f=16: 0.4904, f=64: 0.5003, f=128: 0.4782)
#:   explicit f=32 reg=30          -> 0.2705 (f=16 va f=64 gan nhu bang)
#:   tham chieu ItemKNN k=80       -> 0.4768
FACTORS = 32
REG = 10.0
ALPHA = 10.0
FACTORS_EXPLICIT = 32
REG_EXPLICIT = 30.0
N_ITER = 15


def _ridge_rows(
    B, Y: np.ndarray, base: np.ndarray, scale: float, rhs: np.ndarray,
    block_floats: float = 4e6, outer_floats: float = 2e7,
) -> np.ndarray:
    """Giai (base + scale * G[u]) x[u] = rhs[u] cho moi dong u cua B.

    G[u] = sum_{j trong dong u cua B} y_j y_j^T. Dong rong de x[u] = 0.

    XU LY THEO KHOI, KHONG DUNG HET G MOT LUC: G day du cua buoc item la
    9.724 x f x f so -- 320 MB voi f = 64, 1,3 GB voi f = 128, va con nhan them
    vai ban sao khi cong voi base. Ban dau tien lam vay va mot lan fit mat >10 phut
    vi tran bo nho. Moi khoi o day ~block_floats so (~32 MB).

    Hai cach tinh G, chon theo bo nho:
      - Y it dong (buoc item: Y = P, 610 user): lap bang ngoai y_j y_j^T cho moi
        j MOT lan, roi nhan ma tran thua B -- moi dong cua G la to hop cua cac
        bang ngoai do, khong co vong lap Python.
      - Y nhieu dong (buoc user: Y = Q, 9.724 item): bang ngoai qua lon. Lap
        tung dong (chi 610 dong), moi dong dung |I_u| vector.
    """
    n_rows, f = B.shape[0], Y.shape[1]
    out = np.zeros(rhs.shape)
    has = np.diff(B.indptr) > 0
    outer = None
    if Y.shape[0] * f * f <= outer_floats:
        outer = (Y[:, :, None] * Y[:, None, :]).reshape(Y.shape[0], f * f)
    step = max(1, int(block_floats // (f * f)))
    for lo in range(0, n_rows, step):
        hi = min(lo + step, n_rows)
        if outer is not None:
            G = (B[lo:hi] @ outer).reshape(hi - lo, f, f)
        else:
            G = np.zeros((hi - lo, f, f))
            for r in range(lo, hi):
                idx = B.indices[B.indptr[r] : B.indptr[r + 1]]
                if len(idx):
                    Yi = Y[idx]
                    G[r - lo] = Yi.T @ Yi
        h = has[lo:hi]
        if h.any():
            A = base + scale * G[h]
            out[lo:hi][h] = np.linalg.solve(A, rhs[lo:hi][h][..., None])[..., 0]
    return out


class MF(Recommender):
    """B3 -- phan ra ma tran, ALS. Xem docstring cua module."""

    name = "B3-MF"

    def __init__(
        self,
        factors: int = FACTORS,
        reg: float = REG,
        alpha: float = ALPHA,
        n_iter: int = N_ITER,
        kind: str = "implicit",
        seed: int = 0,
        titles: Mapping[int, str] | None = None,
    ):
        if kind not in ("implicit", "explicit"):
            raise ValueError("kind phai la 'implicit' hoac 'explicit'")
        self.factors = factors
        self.reg = reg
        #: Chi dung cho implicit: o da xem co do tin cay 1 + alpha.
        self.alpha = alpha
        self.n_iter = n_iter
        self.kind = kind
        self.seed = seed
        #: item_id -> ten phim, chi cho explain(). Truyen vao, khong tu doc file.
        self.titles = titles

    # ------------------------------------------------------------------ fit --

    def _fit(self, train: pd.DataFrame) -> None:
        self.R_, self.users_, self.items_, self.uidx_, self.iidx_ = build_matrix(train)
        self.Rcsc_ = self.R_.tocsc()
        n_users, n_items = self.R_.shape
        rng = np.random.default_rng(self.seed)
        # Khoi tao nho: bat dau gan 0 thi buoc ALS dau tien gan nhu chi hoc bias.
        self.P_ = rng.normal(0, 0.01, (n_users, self.factors))
        self.Q_ = rng.normal(0, 0.01, (n_items, self.factors))

        if self.kind == "explicit":
            self.mu_ = float(self.R_.data.mean())
            self.bu_ = np.zeros(n_users)
            self.bi_ = np.zeros(n_items)

        # Moi vong: giai Q theo P, roi P theo Q. Ket thuc bang buoc P nen P khop
        # DUNG voi Q cuoi cung -- explain() dua vao dang thuc do.
        Rt = self.Rcsc_.T.tocsr()  # dong = item
        for _ in range(self.n_iter):
            if self.kind == "implicit":
                self.Q_ = self._implicit_step(self.P_, Rt)
                self.P_ = self._implicit_step(self.Q_, self.R_)
            else:
                self.Q_, self.bi_ = self._explicit_step(self.P_, self.bu_, Rt)
                self.P_, self.bu_ = self._explicit_step(self.Q_, self.bi_, self.R_)

    def _implicit_step(self, Y: np.ndarray, M) -> np.ndarray:
        """Giai moi dong cua M (user, hoac item neu M la ma tran chuyen vi).

        Voi dong u co tap cot da xem I_u:
            (Y^T C_u Y + reg I) x_u = Y^T C_u p_u
        Meo cua Hu et al.: Y^T C_u Y = Y^T Y + alpha * Y_I^T Y_I. Y^T Y tinh MOT
        lan cho ca buoc; phan con lai chi dung |I_u| dong. Khong co meo nay thi
        moi user phai nhan qua 9.724 item, ke ca nhung item ho chua xem.
        """
        f = Y.shape[1]
        base = Y.T @ Y + self.reg * np.eye(f)
        B = M.copy()
        B.data = np.ones_like(B.data)  # implicit: chi can biet DA XEM hay chua
        rhs = (1.0 + self.alpha) * (B @ Y)
        return _ridge_rows(B, Y, base, self.alpha, rhs)

    def _explicit_step(self, Y: np.ndarray, b_other: np.ndarray, M):
        """Hoi quy ridge cho tung dong: dac trung [y_j, 1], muc tieu r - mu - b_j.

        Cot 1 them vao la de hoc bias cua dong nay cung luc voi vector nhan to.
        """
        f = Y.shape[1]
        Z = np.hstack([Y, np.ones((Y.shape[0], 1))])
        B = M.copy()
        B.data = np.ones_like(B.data)
        T = M.copy()
        T.data = M.data - self.mu_ - b_other[M.indices]  # muc tieu cua tung o
        sol = _ridge_rows(B, Z, self.reg * np.eye(f + 1), 1.0, T @ Z)
        return sol[:, :f], sol[:, f]

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
        s = self.Q_[idx] @ self.P_[u]
        if self.kind == "explicit":
            s = s + self.mu_ + self.bu_[u] + self.bi_[idx]
        out[np.asarray(pos)] = s
        return out

    # -------------------------------------------------------------- explain --

    def contributions(self, user_id: int, item_id: int) -> dict[int, float]:
        """Dong gop cua tung phim user DA XEM vao diem p_u . q_i.

        Vi p_u la nghiem ridge, p_u = W_u * sum_j z_j * t_j, nen
            p_u . q_i = sum_j  (q_i^T W_u z_j) * t_j
        Moi so hang la phan dong gop cua phim j. Day la cach giai thich cua chinh
        Hu et al. 2008 (muc 5), va la dang thuc CHINH XAC chu khong xap xi -- tong
        cac dong gop bang dung p_u . q_i (phan nhan to, chua tinh bias).

        Tra ve {item_id da xem: dong gop}.
        """
        u = self.uidx_.get(int(user_id))
        i = self.iidx_.get(int(item_id))
        if u is None or i is None:
            return {}
        lo, hi = self.R_.indptr[u], self.R_.indptr[u + 1]
        idx = self.R_.indices[lo:hi]
        if len(idx) == 0:
            return {}
        f = self.factors
        Q = self.Q_
        if self.kind == "implicit":
            Yi = Q[idx]
            A = Q.T @ Q + self.reg * np.eye(f) + self.alpha * (Yi.T @ Yi)
            v = np.linalg.solve(A, Q[i])  # W_u q_i (W_u doi xung)
            contrib = (Yi @ v) * (1.0 + self.alpha)
        else:
            Z = np.hstack([Q[idx], np.ones((len(idx), 1))])
            A = Z.T @ Z + self.reg * np.eye(f + 1)
            v = np.linalg.solve(A, np.append(Q[i], 0.0))
            t = self.R_.data[lo:hi] - self.mu_ - self.bi_[idx]
            contrib = (Z @ v) * t
        return {int(self.items_[j]): float(c) for j, c in zip(idx, contrib)}

    def explain(self, user_id: int, item_id: int) -> str:
        contrib = self.contributions(user_id, item_id)
        top = [j for j, c in sorted(contrib.items(), key=lambda kv: -kv[1]) if c > 0][:3]
        if not top:
            return ""
        ten = ", ".join(self._title(j) for j in top)
        return f"Vi ban da xem {ten}"

    def _title(self, item_id: int) -> str:
        if self.titles is None:
            return str(item_id)
        return self.titles.get(item_id, str(item_id))
