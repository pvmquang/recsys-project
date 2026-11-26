"""Bac B8 -- LightGCN: loc cong tac voi hoc bieu dien tren DO THI (He et al., SIGIR 2020).

Do thi hai phia: moi user, moi phim la mot nut; user da xem phim thi co canh.
Moi nut co mot vector goc e^(0). Lan truyen K lop, moi lop lay trung binh co
chuan hoa cua hang xom, KHONG co ma tran trong so, KHONG co ham kich hoat:
    E^(k+1) = A~ E^(k),     A~ = D^(-1/2) A D^(-1/2)
    E       = (E^(0) + E^(1) + ... + E^(K)) / (K + 1)
    x_ui    = e_u . e_i
Ham muc tieu giong het B4 (BPR-Opt). Khac B4 dung mot cho: vector cua user
khong con la tham so tu do, ma tron them vector cua phim user da xem (lop 1),
cua nhung user khac da xem cac phim do (lop 2), ... Voi K = 0 mo hinh la
BPR-MF khong bias -- script dung dieu nay de tach phan dong gop cua do thi.

De xuat de tai ghi "B8 NeuMF hoac LightGCN". Chon LightGCN vi:
- No so sanh thang voi B4: cung ham mat mat, cung dang diem, chi them do thi.
  NeuMF doi ca ham mat mat lan dang diem, nen chenh lech khong quy duoc cho mot
  thay doi nao.
- Rendle et al. (RecSys 2020) do lai NeuMF: tich vo huong hoc ky hon MLP. Do
  NeuMF ma thua BPR thi chi lap lai ket qua do.

=========================================================================
VI SAO KHONG CAN PYTORCH
Lan truyen la TUYEN TINH: E = M E^(0) voi M = sum_k A~^k / (K + 1). Gradient
theo E^(0) la M^T G, ma A~ doi xung nen M^T = M. Lan truyen nguoc vi vay chinh
la lan truyen xuoi ap len ma tran gradient G -- K phep nhan ma tran thua.
Khong co tham so nao khac. Tu viet bang scipy duoc, khong phai cai ~200 MB
PyTorch, va tranh chuyen ban CPU/CUDA khac nhau giua ba may (requirements.txt).

Toi uu bang Adam nhu ban goc. SGD cua B4 cap nhat thua tung dong; o day gradient
qua lan truyen roi vao gan het cac nut, nen cap nhat day dac moi lo.
=========================================================================

KE THUA B4
score(), explain(), lay mau phim am va train_more() lay nguyen tu BPR. Sau moi
lan huan luyen, P_ va Q_ duoc gan bang vector CUOI (sau lan truyen), b_ bang 0,
nen score() cua BPR tinh dung e_u . e_i. Cau giai thich vi vay cung la HAU KIEM
nhu B4: ba phim da xem co vector cuoi gan phim goi y nhat.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, log_expit

from recsys.models.bpr import BPR
from recsys.models.knn import build_matrix

#: Sieu tham so CHON TREN VALIDATION (checklist 2.1). Xem scripts/exp_lightgcn.py --
#: script do chay lai phep chon va canh bao neu ket qua khac so ghi o day.
#: ml-latest-small, seed 42, ndcg@10 tren validation (docs/so-lieu-b8.md):
#:   K=4 lr=0.01 reg=1e-4, 300 epoch -> 0.5077 (K=3: 0.5054, K=0: 0.4492 o 100 epoch)
#:   tham chieu BPR 0.4928, MF implicit 0.5118
#: K = 4 va 300 epoch deu nam o BIEN cua luoi, chua thu lon hon.
#: BATCH khong nam trong luoi: do truoc tren validation, K = 3, f = 64. Lo 2.048 (nhu ban
#: goc) mat ~1,2 giay moi epoch vi moi lo can 2K phep nhan ma tran thua; lo 8.192 voi
#: lr 0.02 dat 0.4973 sau 160 epoch, nhanh hon ~3 lan tinh theo thoi gian.
#: Gia: fit tren train mat ~380 giay, BPR mat ~32 giay.
FACTORS = 64
N_LAYERS = 4
LR = 0.01
REG = 1e-4
N_EPOCHS = 300
BATCH = 8192

#: Tham so Adam, lay theo mac dinh cua ban goc. Khong do.
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


class LightGCN(BPR):
    """B8 -- LightGCN. Xem docstring cua module."""

    name = "B8-LightGCN"

    def __init__(
        self,
        factors: int = FACTORS,
        n_layers: int = N_LAYERS,
        lr: float = LR,
        reg: float = REG,
        n_epochs: int = N_EPOCHS,
        batch_size: int = BATCH,
        seed: int = 0,
        titles: Mapping[int, str] | None = None,
    ):
        super().__init__(factors=factors, lr=lr, reg=reg, n_epochs=n_epochs,
                         batch_size=batch_size, seed=seed, titles=titles)
        self.n_layers = n_layers

    # ------------------------------------------------------------------ fit --

    def _fit(self, train: pd.DataFrame) -> None:
        self.R_, self.users_, self.items_, self.uidx_, self.iidx_ = build_matrix(train)
        n_users, n_items = self.R_.shape
        self.A_ = normalized_adjacency(self.R_)
        self.rng_ = np.random.default_rng(self.seed)
        n_nodes = n_users + n_items
        # float32: nhan ma tran thua nhanh gap ~2 lan float64, sai so khong dang ke
        # so voi nhieu cua SGD.
        self.E0_ = self.rng_.normal(0, 0.1, (n_nodes, self.factors)).astype(np.float32)
        self.m_ = np.zeros_like(self.E0_)
        self.v_ = np.zeros_like(self.E0_)
        self.t_ = 0

        self.pos_u_ = np.repeat(np.arange(n_users), np.diff(self.R_.indptr))
        self.pos_i_ = self.R_.indices.copy()
        self.keys_ = np.sort(self.pos_u_.astype(np.int64) * n_items + self.pos_i_)
        #: Trung binh -ln sigma(x_ui - x_uj) cua tung epoch -- phai giam dan.
        self.loss_: list[float] = []
        self.train_more(self.n_epochs)

    def train_more(self, n_epochs: int) -> "LightGCN":
        """Nhu BPR.train_more, roi tinh lai vector cuoi cho score()."""
        super().train_more(n_epochs)
        self._refresh()
        return self

    def propagate(self, X: np.ndarray) -> np.ndarray:
        """M X = (X + A~ X + ... + A~^K X) / (K + 1). Dung cho ca xuoi lan nguoc."""
        out = X.copy()
        cur = X
        for _ in range(self.n_layers):
            cur = self.A_ @ cur
            out += cur
        out /= self.n_layers + 1
        return out

    def _refresh(self) -> None:
        n_users = self.R_.shape[0]
        E = self.propagate(self.E0_).astype(np.float64)
        self.P_, self.Q_ = E[:n_users], E[n_users:]
        self.b_ = np.zeros(self.R_.shape[1])

    def _epoch(self) -> float:
        n_users = self.R_.shape[0]
        order = self.rng_.permutation(len(self.pos_u_))
        tong = 0.0
        for lo in range(0, len(order), self.batch_size):
            sel = order[lo : lo + self.batch_size]
            u, i = self.pos_u_[sel], self.pos_i_[sel]
            j = self._sample_negatives(u)
            loss, grad = self.batch_grad(u, i + n_users, j + n_users)
            tong += loss * len(sel)
            self._adam(grad)
        return tong / len(order)

    def batch_grad(self, u: np.ndarray, i: np.ndarray, j: np.ndarray) -> tuple[float, np.ndarray]:
        """Mat mat cua mot lo va gradient theo E0_. u, i, j la chi so NUT.

        Mat mat = trung binh -ln sigma(x_ui - x_uj) + reg/(2B) * tong ||e^(0)||^2
        tren cac nut trong lo, nhu ban goc. Ham tra ve chi phan BPR cua mat mat
        (de theo doi loss_), gradient thi gom ca L2. tests/test_lightgcn.py so
        gradient nay voi dao ham so.
        """
        B = len(u)
        dtype = self.E0_.dtype
        E = self.propagate(self.E0_)
        eu, ei, ej = E[u], E[i], E[j]
        x = np.einsum("bf,bf->b", eu, ei - ej)
        loss = -float(log_expit(x).mean())
        g = (-expit(-x) / B).astype(dtype)[:, None]  # d mat mat / d x
        # Cong don gradient cua 3B dong vao N nut bang mot phep nhan ma tran
        # thua (nut x 3B). Nhanh hon np.add.at ~10 lan, cung ket qua.
        nodes = np.concatenate([u, i, j])
        S = sparse.csr_matrix((np.ones(3 * B, dtype), (nodes, np.arange(3 * B))),
                              shape=(len(self.E0_), 3 * B))
        grad = self.propagate(S @ np.concatenate([g * (ei - ej), g * eu, -g * eu]))
        grad += S @ (dtype.type(self.reg / B) * self.E0_[nodes])
        return loss, grad

    def _adam(self, grad: np.ndarray) -> None:
        self.t_ += 1
        self.m_ *= BETA1
        self.m_ += (1 - BETA1) * grad
        self.v_ *= BETA2
        grad *= grad
        self.v_ += (1 - BETA2) * grad
        # Gop hieu chinh do lech vao buoc hoc, nhu torch.optim.Adam.
        step = self.lr * np.sqrt(1 - BETA2 ** self.t_) / (1 - BETA1 ** self.t_)
        denom = np.sqrt(self.v_)
        denom += EPS * np.sqrt(1 - BETA2 ** self.t_)
        self.E0_ -= self.E0_.dtype.type(step) * self.m_ / denom


def normalized_adjacency(R: sparse.csr_matrix) -> sparse.csr_matrix:
    """A~ = D^(-1/2) A D^(-1/2) cua do thi hai phia, A = [[0, R], [R^T, 0]].

    R chi can vi tri khac 0, gia tri rating bi bo: LightGCN hoc tu tuong tac.
    Nut bac 0 (khong xay ra voi build_matrix) giu hang 0 thay vi chia cho 0.
    """
    B = (R != 0).astype(np.float32)
    A = sparse.bmat([[None, B], [B.T, None]], format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    np.divide(1.0, np.sqrt(deg), out=inv, where=deg > 0)
    d = sparse.diags(inv).astype(np.float32)
    return (d @ A @ d).tocsr()
