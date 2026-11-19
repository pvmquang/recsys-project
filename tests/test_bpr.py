"""Chot B4 -- BPR-MF (recsys/models/bpr.py).

Cung du lieu do choi HAI CUM nhu test_mf.py: user 1-8 chi xem phim 1-8, user
9-16 chi xem phim 9-16.
"""

import numpy as np
import pandas as pd

from recsys.models.bpr import BPR


def hai_cum(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, 17):
        cum = range(1, 9) if u <= 8 else range(9, 17)
        for t, i in enumerate(rng.choice(list(cum), size=5, replace=False)):
            rows.append((u, int(i), float(rng.integers(3, 6)), t))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


TRAIN = hai_cum()


def chua_xem(u: int, cum: range) -> list[int]:
    seen = set(TRAIN.loc[TRAIN["user_id"] == u, "item_id"])
    return [i for i in cum if i not in seen]


def hoc_xong(**kw) -> BPR:
    # reg = 0.05, khong nho hon: voi reg = 0.01 va 200 epoch, BPR hoc thuoc 16 user
    # va day ca phim CUNG cum chua xem xuong (3/10 seed truot test xep hang).
    # Phim chua xem nao cung la phim am -- chi tong quat hoa moi keo chung len.
    tham_so = dict(factors=2, lr=0.05, reg=0.05, n_epochs=100, batch_size=16)
    tham_so.update(kw)
    return BPR(**tham_so).fit(TRAIN)


class TestCoBan:
    def test_item_la_va_user_la_ra_am_vo_cuc(self):
        m = hoc_xong(n_epochs=2)
        s = m.score(1, np.array([1, 999]))
        assert np.isfinite(s[0]) and np.isneginf(s[1])
        assert np.isneginf(m.score(999, np.array([1, 2]))).all()

    def test_cung_seed_cung_ket_qua(self):
        a, b = hoc_xong(n_epochs=5, seed=3), hoc_xong(n_epochs=5, seed=3)
        assert np.array_equal(a.P_, b.P_) and np.array_equal(a.b_, b.b_)

    def test_train_more_bang_fit_mot_lan(self):
        """Script chon so epoch dua vao dang thuc nay."""
        mot_lan = hoc_xong(n_epochs=7)
        hai_lan = hoc_xong(n_epochs=3).train_more(4)
        assert np.array_equal(mot_lan.P_, hai_lan.P_)
        assert np.array_equal(mot_lan.Q_, hai_lan.Q_)
        assert mot_lan.loss_ == hai_lan.loss_

    def test_loss_giam(self):
        m = hoc_xong(n_epochs=50)
        assert len(m.loss_) == 50
        assert m.loss_[-1] < 0.5 * m.loss_[0]


class TestLayMauAm:
    def test_phim_am_khong_bao_gio_la_phim_da_xem(self):
        m = hoc_xong(n_epochs=0)
        u = np.repeat(np.arange(16), 200)
        j = m._sample_negatives(u)
        for uu, jj in zip(u, j):
            assert m.R_[uu, jj] == 0

    def test_phim_am_phu_het_phim_chua_xem(self):
        """Lay deu: user 1 (cum 1-8) phai gap du 11 phim chua xem, khong lech."""
        m = hoc_xong(n_epochs=0)
        u0 = m.uidx_[1]
        j = m._sample_negatives(np.full(5000, u0))
        assert len(np.unique(j)) == 16 - 5


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

    def test_similar_seen_la_cosine_cua_dung_phim_da_xem(self):
        m = hoc_xong(n_epochs=5)
        sim = m.similar_seen(1, 3)
        assert set(sim) == set(TRAIN.loc[TRAIN["user_id"] == 1, "item_id"])
        assert all(-1 - 1e-9 <= c <= 1 + 1e-9 for c in sim.values())

    def test_explain_rong_khi_khong_biet_user_hay_item(self):
        m = hoc_xong(n_epochs=2)
        assert m.explain(999, 1) == ""
        assert m.explain(1, 999) == ""
