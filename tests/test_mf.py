"""Chot B3 -- phan ra ma tran (recsys/models/mf.py).

Du lieu do choi co HAI CUM ro rang: user 1-8 chi xem phim 1-8, user 9-16 chi
xem phim 9-16. Mot MF dung phai hoc ra hai "khau vi" do va xep phim cung cum
len tren phim khac cum, du chua ai xem cap phim do cung nhau.
"""

import numpy as np
import pandas as pd
import pytest

from recsys.models.mf import MF, _ridge_rows
from scipy import sparse


def hai_cum(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, 17):
        cum = range(1, 9) if u <= 8 else range(9, 17)
        # Moi user xem 5/8 phim cua cum minh -- con 3 phim chua xem de thu.
        for t, i in enumerate(rng.choice(list(cum), size=5, replace=False)):
            rows.append((u, int(i), float(rng.integers(3, 6)), t))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


TRAIN = hai_cum()


def chua_xem(u: int, cum: range) -> list[int]:
    seen = set(TRAIN.loc[TRAIN["user_id"] == u, "item_id"])
    return [i for i in cum if i not in seen]


@pytest.mark.parametrize("kind", ["implicit", "explicit"])
class TestChung:
    def test_item_la_va_user_la_ra_am_vo_cuc(self, kind):
        m = MF(factors=2, kind=kind, n_iter=5).fit(TRAIN)
        s = m.score(1, np.array([1, 999]))
        assert np.isfinite(s[0]) and np.isneginf(s[1])
        assert np.isneginf(m.score(999, np.array([1, 2]))).all()

    def test_cung_seed_cung_ket_qua(self, kind):
        a = MF(factors=3, kind=kind, n_iter=5, seed=1).fit(TRAIN)
        b = MF(factors=3, kind=kind, n_iter=5, seed=1).fit(TRAIN)
        assert np.array_equal(a.P_, b.P_) and np.array_equal(a.Q_, b.Q_)

    def test_phan_ra_explain_cong_lai_DUNG_bang_diem(self, kind):
        """Tong dong gop cua cac phim da xem = p_u . q_i, chinh xac (khong xap xi)."""
        m = MF(factors=3, kind=kind, n_iter=5).fit(TRAIN)
        for u in (1, 9):
            for i in (2, 12):
                c = sum(m.contributions(u, i).values())
                p = m.P_[m.uidx_[u]] @ m.Q_[m.iidx_[i]]
                assert c == pytest.approx(p, abs=1e-9)

    def test_dong_gop_chi_gom_phim_user_da_xem(self, kind):
        m = MF(factors=3, kind=kind, n_iter=5).fit(TRAIN)
        seen = set(TRAIN.loc[TRAIN["user_id"] == 1, "item_id"])
        assert set(m.contributions(1, 3)) == seen


class TestImplicit:
    def test_xep_phim_CUNG_CUM_len_tren(self):
        m = MF(factors=2, reg=0.1, alpha=10, n_iter=15).fit(TRAIN)
        for u, cum, khac in ((1, range(1, 9), range(9, 17)), (9, range(9, 17), range(1, 9))):
            trong = m.score(u, np.array(chua_xem(u, cum)))
            ngoai = m.score(u, np.array(list(khac)))
            assert trong.min() > ngoai.max()

    def test_recommend_khong_tra_phim_da_xem_va_uu_tien_cung_cum(self):
        m = MF(factors=2, reg=0.1, alpha=10, n_iter=15).fit(TRAIN)
        rec = m.recommend(1, k=3)
        assert set(rec) == set(chua_xem(1, range(1, 9)))

    def test_explain_neu_ten_phim_da_xem(self):
        titles = {i: f"Phim {i}" for i in range(1, 17)}
        m = MF(factors=2, reg=0.1, alpha=10, n_iter=15, titles=titles).fit(TRAIN)
        i = m.recommend(1, k=1)[0]
        txt = m.explain(1, i)
        assert txt.startswith("Vi ban da xem Phim ")
        seen = set(TRAIN.loc[TRAIN["user_id"] == 1, "item_id"])
        for ten in txt.removeprefix("Vi ban da xem ").split(", "):
            assert int(ten.split()[1]) in seen

    def test_explain_rong_khi_khong_biet_user_hay_item(self):
        m = MF(factors=2, n_iter=3).fit(TRAIN)
        assert m.explain(999, 1) == ""
        assert m.explain(1, 999) == ""


class TestExplicit:
    def test_hoc_duoc_rating_tren_tap_train(self):
        """Reg nho, du nhan to: sai so tren train phai nho hon doan bang trung binh."""
        m = MF(factors=4, reg=0.01, kind="explicit", n_iter=30).fit(TRAIN)
        pred = np.array([m.score(u, np.array([i]))[0] for u, i in
                         zip(TRAIN["user_id"], TRAIN["item_id"])])
        rmse = np.sqrt(np.mean((pred - TRAIN["rating"]) ** 2))
        rmse_tb = np.sqrt(np.mean((TRAIN["rating"].mean() - TRAIN["rating"]) ** 2))
        assert rmse < 0.5 * rmse_tb

    def test_kind_sai_nem_loi(self):
        with pytest.raises(ValueError):
            MF(kind="svd")


class TestRidgeRows:
    @pytest.mark.parametrize("outer_floats", [1e9, 0])  # ca hai nhanh tinh G
    def test_khop_giai_tung_dong_bang_tay(self, outer_floats):
        rng = np.random.default_rng(0)
        B = sparse.random(30, 12, density=0.3, format="csr", random_state=1)
        B.data[:] = 1.0
        Y = rng.normal(size=(12, 4))
        base = 0.5 * np.eye(4)
        rhs = rng.normal(size=(30, 4))
        got = _ridge_rows(B, Y, base, 2.0, rhs, block_floats=40, outer_floats=outer_floats)
        for r in range(30):
            idx = B.indices[B.indptr[r] : B.indptr[r + 1]]
            if len(idx) == 0:
                assert (got[r] == 0).all()
                continue
            A = base + 2.0 * Y[idx].T @ Y[idx]
            assert np.allclose(got[r], np.linalg.solve(A, rhs[r]))
