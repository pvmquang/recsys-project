"""Chot B8 -- LightGCN (recsys/models/lightgcn.py).

Cung du lieu do choi HAI CUM nhu test_bpr.py: user 1-8 chi xem phim 1-8, user
9-16 chi xem phim 9-16. Lan truyen nguoc viet tay nen co mot bai so gradient
voi dao ham so.
"""

import numpy as np
from scipy import sparse
from scipy.special import log_expit

from recsys.models.lightgcn import LightGCN, normalized_adjacency
from tests.test_bpr import TRAIN, chua_xem


def hoc_xong(**kw) -> LightGCN:
    tham_so = dict(factors=4, n_layers=2, lr=0.05, reg=1e-3, n_epochs=100, batch_size=16)
    tham_so.update(kw)
    return LightGCN(**tham_so).fit(TRAIN)


def ma_tran_lan_truyen(m: LightGCN) -> np.ndarray:
    """M = (I + A~ + ... + A~^K) / (K + 1), tinh DAY DAC tu dinh nghia."""
    R = (m.R_.toarray() != 0).astype(float)
    nu, ni = R.shape
    A = np.block([[np.zeros((nu, nu)), R], [R.T, np.zeros((ni, ni))]])
    d = 1 / np.sqrt(A.sum(axis=1))
    At = d[:, None] * A * d[None, :]
    M, P = np.eye(len(A)), np.eye(len(A))
    for _ in range(m.n_layers):
        P = P @ At
        M = M + P
    return M / (m.n_layers + 1)


class TestDoThi:
    def test_ma_tran_ke_chuan_hoa_doi_xung_va_dung_cong_thuc(self):
        R = np.array([[1, 1, 0], [0, 1, 0]], dtype=float)
        A = normalized_adjacency(sparse.csr_matrix(R)).toarray()
        assert np.allclose(A, A.T)
        # user 0 bac 2, phim 1 bac 2 -> 1/sqrt(2*2); user 1 bac 1, phim 1 -> 1/sqrt(1*2)
        assert np.isclose(A[0, 2 + 1], 0.5)
        assert np.isclose(A[1, 2 + 1], 1 / np.sqrt(2))
        assert A[0, 1] == 0 and A[2, 3] == 0  # khong co canh user-user, phim-phim
        # phim 2 khong ai xem: hang va cot bang 0, khong NaN
        assert np.isfinite(A).all() and not A[2 + 2].any()

    def test_vector_cuoi_bang_trung_binh_cac_lop(self):
        m = hoc_xong(n_epochs=3)
        E = ma_tran_lan_truyen(m) @ m.E0_.astype(float)
        nu = m.R_.shape[0]
        assert np.allclose(m.P_, E[:nu], atol=1e-5)
        assert np.allclose(m.Q_, E[nu:], atol=1e-5)

    def test_khong_lop_nao_thi_la_vector_goc(self):
        """K = 0 la BPR-MF khong bias -- script dung de tach dong gop cua do thi."""
        m = hoc_xong(n_layers=0, n_epochs=3)
        nu = m.R_.shape[0]
        assert np.array_equal(m.P_, m.E0_[:nu].astype(float))
        assert not m.b_.any()


class TestGradient:
    def test_gradient_khop_dao_ham_so(self):
        m = hoc_xong(n_epochs=0)
        m.E0_ = m.E0_.astype(np.float64)
        rng = np.random.default_rng(1)
        nu, ni = m.R_.shape
        u = rng.integers(0, nu, 6)
        i = nu + rng.integers(0, ni, 6)
        j = nu + rng.integers(0, ni, 6)

        def mat_mat(E0):
            E = ma_tran_lan_truyen(m) @ E0
            x = np.einsum("bf,bf->b", E[u], E[i] - E[j])
            nodes = np.concatenate([u, i, j])
            return -log_expit(x).mean() + m.reg / (2 * len(u)) * (E0[nodes] ** 2).sum()

        _, grad = m.batch_grad(u, i, j)
        h = 1e-6
        for r, c in zip(rng.integers(0, nu + ni, 20), rng.integers(0, m.factors, 20)):
            E = m.E0_.copy()
            E[r, c] += h
            tren = mat_mat(E)
            E[r, c] -= 2 * h
            duoi = mat_mat(E)
            assert np.isclose(grad[r, c], (tren - duoi) / (2 * h), rtol=1e-4, atol=1e-8)


class TestCoBan:
    def test_item_la_va_user_la_ra_am_vo_cuc(self):
        m = hoc_xong(n_epochs=2)
        s = m.score(1, np.array([1, 999]))
        assert np.isfinite(s[0]) and np.isneginf(s[1])
        assert np.isneginf(m.score(999, np.array([1, 2]))).all()

    def test_cung_seed_cung_ket_qua(self):
        a, b = hoc_xong(n_epochs=5, seed=3), hoc_xong(n_epochs=5, seed=3)
        assert np.array_equal(a.E0_, b.E0_)

    def test_train_more_bang_fit_mot_lan(self):
        """Script chon so epoch dua vao dang thuc nay, giong B4."""
        mot_lan = hoc_xong(n_epochs=7)
        hai_lan = hoc_xong(n_epochs=3).train_more(4)
        assert np.array_equal(mot_lan.E0_, hai_lan.E0_)
        assert np.array_equal(mot_lan.P_, hai_lan.P_)
        assert mot_lan.loss_ == hai_lan.loss_

    def test_loss_giam(self):
        m = hoc_xong(n_epochs=50)
        assert len(m.loss_) == 50
        assert m.loss_[-1] < 0.5 * m.loss_[0]

    def test_rating_khong_anh_huong(self):
        """LightGCN hoc tu tuong tac: doi rating thi ket qua khong doi."""
        khac = TRAIN.assign(rating=1.0)
        a = hoc_xong(n_epochs=5)
        b = LightGCN(factors=4, n_layers=2, lr=0.05, reg=1e-3, n_epochs=5,
                     batch_size=16).fit(khac)
        assert np.array_equal(a.E0_, b.E0_)


class TestXepHang:
    def test_xep_phim_CUNG_CUM_len_tren(self):
        m = hoc_xong()
        for u, cum, khac in ((1, range(1, 9), range(9, 17)), (9, range(9, 17), range(1, 9))):
            trong = m.score(u, np.array(chua_xem(u, cum)))
            ngoai = m.score(u, np.array(list(khac)))
            assert trong.min() > ngoai.max()

    def test_recommend_khong_tra_phim_da_xem_va_uu_tien_cung_cum(self):
        m = hoc_xong()
        assert set(m.recommend(1, k=3)) == set(chua_xem(1, range(1, 9)))


class TestExplain:
    def test_explain_chi_neu_phim_da_xem(self):
        titles = {i: f"Phim {i}" for i in range(1, 17)}
        m = hoc_xong(titles=titles)
        i = m.recommend(1, k=1)[0]
        txt = m.explain(1, i)
        assert txt.startswith("Giong cac phim ban da xem: Phim ")
        seen = set(TRAIN.loc[TRAIN["user_id"] == 1, "item_id"])
        for ten in txt.split(": ", 1)[1].split(", "):
            assert int(ten.split()[1]) in seen

    def test_explain_rong_khi_khong_biet_user_hay_item(self):
        m = hoc_xong(n_epochs=2)
        assert m.explain(999, 1) == ""
        assert m.explain(1, 999) == ""
