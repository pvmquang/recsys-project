"""Bac B1 va B2 -- loc cong tac dua tren LANG GIENG GAN (bai L2).

    B1  Item-KNN   tuong dong giua ITEM, tinh tu ma tran rating
    B2  User-KNN   tuong quan Pearson giua USER

=========================================================================
KHAC HAN B5 (noi dung) DU CUNG DUNG CHU "COSINE"
B5 do tuong dong bang METADATA (the loai, dien vien, dao dien).
B1 do tuong dong bang HANH VI: hai phim giong nhau vi CUNG MOT NHOM NGUOI
thich chung, khong can biet chung noi ve cai gi.

Ma nguon tham khao cua do an chi co duong B5, va con thieu ho so nguoi dung.
Hai bac nay la mo hinh dau tien trong ca do an that su dung den du lieu
NGUOI DUNG -- tuc dieu ma README cua repo goc noi no lam nhung khong lam.
=========================================================================

HAI CACH CHAM DIEM, VA CHUNG TRA LOI HAI CAU HOI KHAC NHAU

    score_mode="sum"       diem = TONG do tuong dong voi nhung gi user da xem
                           -> xep hang top-K
    score_mode="weighted"  diem = trung binh rating co trong so (cong thuc
                           du doan rating o slide L2 tr.17/22)
                           -> du doan MAE/RMSE

Bo do do cua do an do XEP HANG, nen "sum" la che do mac dinh. Dung
"weighted" de xep hang thi phan lon ung vien khong co lang gieng nao tung
duoc user danh gia => diem khong xac dinh (xem `_NO_EVIDENCE`), va bang ket
qua tut hang. Do la mot KET QUA DANG BAO CAO, khong phai loi -- bao cao nen
do CA HAI che do roi giai thich vi sao chung khac nhau.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.base import Recommender

#: Diem cho ung vien khong co MOT bang chung nao (khong lang gieng nao cua no
#: tung duoc user danh gia). Tra 0 la sai: 0 co the CAO HON diem that cua item
#: khac, va item khong bang chung se leo len dau danh sach (xem Recommender.score).
_NO_EVIDENCE = -np.inf


def build_matrix(train: pd.DataFrame):
    """Ma tran thua user x item + hai bang tra chi so.

    Tra ve (R, users, items, uidx, iidx).
    """
    dup = train.duplicated(subset=["user_id", "item_id"]).sum()
    if dup:
        # csr_matrix CONG don cac o trung nhau, nen mot cap (user, item) xuat
        # hien hai lan se thanh rating 9.0 -- khong nem, chi lam lech mo hinh.
        raise ValueError(
            f"train co {dup} cap (user_id, item_id) trung nhau. "
            "Loai trung truoc khi huan luyen."
        )

    users = np.sort(train["user_id"].unique())
    items = np.sort(train["item_id"].unique())
    uidx = {int(u): i for i, u in enumerate(users)}
    iidx = {int(v): i for i, v in enumerate(items)}

    rows = train["user_id"].map(uidx).to_numpy()
    cols = train["item_id"].map(iidx).to_numpy()
    vals = train["rating"].to_numpy(dtype=float)
    R = sparse.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    return R, users, items, uidx, iidx


def topk_similarity(
    M: sparse.csr_matrix,
    k: int,
    *,
    block: int = 256,
    min_sim: float = 0.0,
    significance: int = 0,
    binary: sparse.csr_matrix | None = None,
) -> sparse.csr_matrix:
    """Cosine giua cac DONG cua `M`, chi giu `k` lang gieng manh nhat moi dong.

    M da tru trung binh thi cosine nay CHINH LA tuong quan Pearson (xem
    UserKNN ben duoi -- ke ca phan xap xi).

    VI SAO PHAI CAT TOP-K NGAY TRONG LUC TINH, khong tinh het roi cat:
    ma tran tuong dong day cua ml-latest-small la 9.724 x 9.724 x 8 byte
    = 756 MB. Tinh theo khoi roi cat ngay giu bo nho co chan, va van la thu
    tu "tien xu ly truoc" ma slide L2 tr.23-24 noi (Amazon 2003).

    significance: he so gamma cua slide L2 tr.19 -- ha trong so cua cap chi
        cung danh gia IT item. 0 = tat. Can `binary` de dem so item chung.
    min_sim: chi giu lang gieng co tuong dong LON HON nguong nay. Mac dinh 0
        nen tuong dong AM bi bo -- xem ghi chu o UserKNN.
    """
    if significance > 0 and binary is None:
        raise ValueError("significance > 0 thi phai truyen `binary` de dem item chung")

    norms = np.sqrt(M.multiply(M).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    Mn = sparse.diags(1.0 / norms) @ M

    n = M.shape[0]
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []

    for start in range(0, n, block):
        end = min(start + block, n)
        S = (Mn[start:end] @ Mn.T).toarray()

        if significance > 0:
            co = (binary[start:end] @ binary.T).toarray()
            S *= np.minimum(co, significance) / significance

        for r in range(end - start):
            row = S[r]
            row[start + r] = 0.0  # bo chinh no -- item nao cung giong chinh no 100%
            row[row <= min_sim] = 0.0

            nz = int(np.count_nonzero(row))
            kk = min(k, nz)
            if kk == 0:
                indptr.append(len(indices))
                continue
            top = np.argpartition(-row, kk - 1)[:kk]
            top = top[row[top] > 0]
            indices.extend(top.tolist())
            data.extend(row[top].tolist())
            indptr.append(len(indices))

    return sparse.csr_matrix(
        (data, indices, indptr), shape=(n, n)
    )


class _KnnBase(Recommender):
    """Phan dung chung cua B1 va B2."""

    def __init__(self, k: int = 40, score_mode: str = "sum", min_sim: float = 0.0):
        if score_mode not in ("sum", "weighted"):
            raise ValueError("score_mode phai la 'sum' hoac 'weighted'")
        self.k = k
        self.score_mode = score_mode
        self.min_sim = min_sim

    def _cand_index(self, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tach ung vien thanh (vi tri trong item_ids, chi so cot) cho item BIET."""
        pos, idx = [], []
        for p, it in enumerate(item_ids):
            j = self.iidx_.get(int(it))
            if j is not None:
                pos.append(p)
                idx.append(j)
        return np.asarray(pos, dtype=int), np.asarray(idx, dtype=int)


class ItemKNN(_KnnBase):
    """B1 -- tuong dong ITEM tinh tu ma tran rating (slide L2 tr.20-24).

    Diem cua item j cho user u:
        sum      : tong sim(j, m) voi moi m la item u DA danh gia
        weighted : sum(sim(j,m) * r(u,m)) / sum(sim(j,m))   <- du doan rating
    """

    name = "B1-ItemKNN"

    def __init__(
        self,
        k: int = 40,
        score_mode: str = "sum",
        min_sim: float = 0.0,
        adjusted: bool = False,
    ):
        super().__init__(k, score_mode, min_sim)
        #: adjusted=True: tru trung binh CUA USER truoc khi tinh cosine
        #: ("adjusted cosine"). Bu cho viec nguoi cham diem cao/thap khac nhau.
        #: Mac dinh TAT vi slide L2 tr.21 noi cosine tron -- bat len la mot
        #: thi nghiem dang do, khong phai mac dinh im lang.
        self.adjusted = adjusted

    def _fit(self, train: pd.DataFrame) -> None:
        self.R_, self.users_, self.items_, self.uidx_, self.iidx_ = build_matrix(train)
        M = self.R_.T.tocsr()  # dong = item

        if self.adjusted:
            mu = np.asarray(self.R_.sum(axis=1)).ravel() / np.maximum(
                np.diff(self.R_.indptr), 1
            )
            Rc = self.R_.copy()
            Rc.data -= np.repeat(mu, np.diff(self.R_.indptr))
            M = Rc.T.tocsr()

        self.S_ = topk_similarity(M, self.k, min_sim=self.min_sim)
        self.n_items_ = self.R_.shape[1]

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        out = np.full(len(item_ids), _NO_EVIDENCE, dtype=float)
        u = self.uidx_.get(int(user_id))
        if u is None:
            return out  # user chua co trong train: khong mot bang chung nao

        row = self.R_[u]
        if row.nnz == 0:
            return out

        pos, idx = self._cand_index(item_ids)
        if len(idx) == 0:
            return out

        mask = np.zeros(self.n_items_)
        mask[row.indices] = 1.0
        den = self.S_[idx] @ mask  # tong sim voi nhung item u da danh gia

        if self.score_mode == "sum":
            # den == 0 nghia la KHONG lang gieng nao cua item nay tung duoc user
            # danh gia -- khong bang chung, khong phai "bang chung bang 0".
            out[pos] = np.where(den > 0, den, _NO_EVIDENCE)
            return out

        vals = np.zeros(self.n_items_)
        vals[row.indices] = row.data
        num = self.S_[idx] @ vals
        ok = den > 0
        got = np.full(len(idx), _NO_EVIDENCE, dtype=float)
        got[ok] = num[ok] / den[ok]
        out[pos] = got
        return out

    def explain(self, user_id: int, item_id: int) -> str:
        u = self.uidx_.get(int(user_id))
        j = self.iidx_.get(int(item_id))
        if u is None or j is None:
            return ""
        row = self.R_[u]
        srow = self.S_[j]
        shared = np.intersect1d(srow.indices, row.indices)
        if len(shared) == 0:
            return ""
        w = {m: s for m, s in zip(srow.indices, srow.data)}
        best = sorted(shared, key=lambda m: -w[m])[:3]
        ten = ", ".join(str(self.items_[m]) for m in best)
        return f"Vi ban da danh gia {len(shared)} phim tuong tu (gan nhat: {ten})"


class UserKNN(_KnnBase):
    """B2 -- tuong quan Pearson giua USER (slide L2 tr.12-19).

    Diem cua item i cho user u:
        sum      : tong sim(u, v) voi moi v la user DA danh gia i
        weighted : r_tb(u) + sum(sim(u,v) * (r(v,i) - r_tb(v))) / sum|sim(u,v)|
                   <- dung cong thuc slide L2 tr.17

    XAP XI CAN BIET: Pearson dung nghia tinh trung binh tren cac item HAI BEN
    CUNG danh gia. O day tru trung binh tren TOAN BO item cua tung user roi
    lay cosine -- tuc o trong coi nhu "danh gia dung bang trung binh cua minh".
    Day la cach moi thu vien lam (tinh duoc theo ma tran thua), nhung no KHONG
    bang cong thuc trong slide. Neu bao cao dan cong thuc Pearson thi phai noi
    ra cho xap xi nay.

    ⚠ USER CHAM MOI PHIM CUNG MOT DIEM THI KHONG CO LANG GIENG NAO. Phuong sai
    cua ho bang 0 => vector sau khi tru trung binh toan 0 => Pearson voi BAT KY
    ai cung khong xac dinh. Ho se khong nhan duoc goi y nao tu bac nay (moi diem
    = -inf). Day la tinh chat cua Pearson, khong phai loi cai dat -- va no la
    mot ly do that de bac B7 phai co nhanh lai ghep kieu chuyen dich.
    """

    name = "B2-UserKNN"

    def __init__(
        self,
        k: int = 40,
        score_mode: str = "sum",
        min_sim: float = 0.0,
        significance: int = 0,
    ):
        super().__init__(k, score_mode, min_sim)
        #: gamma cua slide L2 tr.19. Hai user chi cung danh gia 2 phim thi
        #: tuong quan 1.0 cua ho khong dang tin bang hai user cung danh gia 50
        #: phim; he so nay ha trong so theo so item chung. 0 = tat.
        self.significance = significance

    def _fit(self, train: pd.DataFrame) -> None:
        self.R_, self.users_, self.items_, self.uidx_, self.iidx_ = build_matrix(train)

        counts = np.maximum(np.diff(self.R_.indptr), 1)
        self.mu_ = np.asarray(self.R_.sum(axis=1)).ravel() / counts

        Rc = self.R_.copy()
        Rc.data -= np.repeat(self.mu_, np.diff(self.R_.indptr))

        binary = None
        if self.significance > 0:
            binary = self.R_.copy()
            binary.data = np.ones_like(binary.data)

        self.S_ = topk_similarity(
            Rc,
            self.k,
            min_sim=self.min_sim,
            significance=self.significance,
            binary=binary,
        )
        # Cot = item, de lay "nhung user da danh gia item i" bang mot phep nhan.
        self.Rcsc_ = self.R_.tocsc()
        self.Rc_csc_ = Rc.tocsc()
        self.n_users_ = self.R_.shape[0]

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        out = np.full(len(item_ids), _NO_EVIDENCE, dtype=float)
        u = self.uidx_.get(int(user_id))
        if u is None:
            return out

        srow = self.S_[u]
        if srow.nnz == 0:
            return out  # khong lang gieng nao: khong doan duoc gi

        sim = np.zeros(self.n_users_)
        sim[srow.indices] = srow.data

        pos, idx = self._cand_index(item_ids)
        if len(idx) == 0:
            return out

        # Cot j cua R nhi phan: user nao da danh gia item j.
        cols = self.Rcsc_[:, idx]
        binc = cols.copy()
        binc.data = np.ones_like(binc.data)
        den = np.asarray(sim @ binc).ravel()  # tong sim cua lang gieng da danh gia j

        if self.score_mode == "sum":
            out[pos] = np.where(den > 0, den, _NO_EVIDENCE)
            return out

        num = np.asarray(sim @ self.Rc_csc_[:, idx]).ravel()
        ok = den > 0
        got = np.full(len(idx), _NO_EVIDENCE, dtype=float)
        got[ok] = self.mu_[u] + num[ok] / den[ok]
        out[pos] = got
        return out

    def explain(self, user_id: int, item_id: int) -> str:
        u = self.uidx_.get(int(user_id))
        j = self.iidx_.get(int(item_id))
        if u is None or j is None:
            return ""
        srow = self.S_[u]
        raters = set(self.Rcsc_[:, j].indices.tolist())
        n = len(raters & set(srow.indices.tolist()))
        if n == 0:
            return ""
        return f"{n} nguoi co so thich giong ban da danh gia phim nay"
