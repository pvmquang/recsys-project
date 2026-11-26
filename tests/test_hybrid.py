"""Chot B7 -- lai ghep (recsys/models/hybrid.py).

Du lieu do choi giong test_content: phim 1-4 Action, 5-8 Romance, phim 9 Action
nhung KHONG AI XEM (phim moi), phim 10 khong co dac trung nao va cung chua ai xem.
"""

import numpy as np
import pandas as pd
import pytest

from recsys.models.baselines import Popularity
from recsys.models.bpr import BPR
from recsys.models.content import ContentBased
from recsys.models.hybrid import NOI_DUNG, PHO_BIEN, TRON, Hybrid, _percentile

META = pd.DataFrame(
    {
        "item_id": list(range(1, 11)),
        "title": [f"Phim {i} ({1980 + i})" for i in range(1, 11)],
        "genres": [["Action"]] * 4 + [["Romance"]] * 4 + [["Action"], []],
        "directors": [[] for _ in range(10)],
        "cast": [[] for _ in range(10)],
        "keywords": [[] for _ in range(10)],
        "overview": [""] * 10,
    }
)

TRAIN = pd.DataFrame(
    [
        (1, 1, 5.0, 1), (1, 2, 4.0, 2),
        (2, 5, 5.0, 1), (2, 6, 4.0, 2), (2, 1, 3.0, 3),
        (3, 1, 4.0, 1), (3, 3, 4.0, 2), (3, 5, 4.0, 3), (3, 7, 4.0, 4),
        (4, 4, 4.0, 1), (4, 8, 4.0, 2), (4, 1, 4.0, 3),
    ],
    columns=["user_id", "item_id", "rating", "timestamp"],
)
DA_XEM = np.array(sorted(TRAIN["item_id"].unique()))


def hoc(cf=None, **kw) -> Hybrid:
    # Mac dinh moi user cung alpha, de test doc lap voi RAMP da chon tren MovieLens.
    kw.setdefault("ramp", "hang")
    return Hybrid(
        content=ContentBased(META, fields=("genres",)),
        cf=cf if cf is not None else Popularity(),
        **kw,
    ).fit(TRAIN)


def bpr_nho() -> BPR:
    return BPR(factors=4, n_epochs=20, batch_size=4, seed=1)


class TestChuyenNhanh:
    def test_user_moi_nhan_diem_popularity(self):
        m = hoc()
        items = np.array([1, 2, 5, 9])
        pop = Popularity().fit(TRAIN).score(999, items)
        np.testing.assert_array_equal(m.score(999, items), pop)
        assert m.branch(999, 1) == PHO_BIEN

    def test_phim_chua_ai_xem_lay_diem_noi_dung(self):
        m = hoc()
        assert np.isneginf(m.cf.score(1, np.array([9]))[0])
        s = m.score(1, np.array([9]))
        assert np.isfinite(s[0])
        assert m.branch(1, 9) == NOI_DUNG
        assert m.branch(1, 3) == TRON

    def test_khong_nhanh_nao_co_bang_chung_thi_am_vo_cuc(self):
        """Phim 10 chua ai xem va khong co dac trung; 999 khong ton tai."""
        m = hoc()
        assert np.isneginf(m.score(1, np.array([10, 999]))).all()

    def test_recommend_dua_phim_moi_len(self):
        """Hop dong chi duyet phim trong train; B7 mo rong danh muc o _fit()."""
        m = hoc(alpha=1.0)
        assert 9 in m.recommend(1, k=10)
        assert 9 in m.all_items_ and 9 not in m.cf.all_items_

    def test_tat_mo_rong_danh_muc_thi_khong_co_phim_moi(self):
        m = hoc(alpha=1.0, extend_catalog=False)
        assert 9 not in m.recommend(1, k=10)


class TestTron:
    @pytest.mark.parametrize("norm", ["rank", "z"])
    def test_alpha_0_xep_phim_da_co_nguoi_xem_dung_nhu_cf(self, norm):
        m = hoc(cf=bpr_nho(), alpha=0.0, norm=norm)
        for u in (1, 2, 3):
            h = m.score(u, DA_XEM)
            c = m.cf.score(u, DA_XEM)
            # Chuan hoa la ham don dieu tang: thu tu tung cap giu nguyen.
            assert (np.sign(np.subtract.outer(h, h)) == np.sign(np.subtract.outer(c, c))).all()

    def test_alpha_1_xep_nhu_noi_dung(self):
        m = hoc(cf=bpr_nho(), alpha=1.0)
        items = np.array([2, 3, 4, 5, 6, 7, 8, 9])
        h = m.score(1, items)
        c = m.content.score(1, items)
        ok = np.isfinite(c)
        assert (np.argsort(-h[ok], kind="stable") == np.argsort(-c[ok], kind="stable")).all()

    def test_diem_khong_doi_theo_danh_sach_ung_vien(self):
        """Thong ke chuan hoa lay tren ca danh muc, khong tren ung vien."""
        m = hoc(cf=bpr_nho(), alpha=0.3, norm="z")
        mot = m.score(1, np.array([3]))[0]
        nhieu = m.score(1, np.array([7, 3, 9, 5]))[1]
        assert mot == pytest.approx(nhieu)

    def test_tham_so_sai_bao_loi(self):
        with pytest.raises(ValueError):
            Hybrid(META, norm="minmax")
        with pytest.raises(ValueError):
            Hybrid(META, alpha=1.5)
        with pytest.raises(ValueError):
            Hybrid()


class TestAlphaTheoLichSu:
    """TRAIN: user 1 co 2 tuong tac, user 3 co 4."""

    def test_hang_moi_user_cung_alpha(self):
        m = hoc(alpha=0.4)
        assert m.alpha_for(1) == m.alpha_for(3) == 0.4

    def test_nguong_user_it_lich_su_ve_alpha_0(self):
        m = hoc(alpha=0.4, ramp="nguong", ramp_n=3)
        assert m.alpha_for(1) == 0.0
        assert m.alpha_for(3) == 0.4

    def test_co_rut_tang_theo_so_tuong_tac(self):
        m = hoc(alpha=0.4, ramp="co", ramp_n=2)
        assert m.alpha_for(1) == pytest.approx(0.4 * 2 / 4)
        assert m.alpha_for(3) == pytest.approx(0.4 * 4 / 6)

    def test_user_duoi_nguong_xep_phim_cu_nhu_cf(self):
        m = hoc(cf=bpr_nho(), alpha=1.0, ramp="nguong", ramp_n=3)
        h, c = m.score(1, DA_XEM), m.cf.score(1, DA_XEM)
        assert (np.sign(np.subtract.outer(h, h)) == np.sign(np.subtract.outer(c, c))).all()

    def test_explain_dung_alpha_cua_user(self):
        """alpha = 1 nhung user 1 duoi nguong: cau giai thich lay tu CF."""
        m = hoc(alpha=1.0, ramp="nguong", ramp_n=3)
        assert m.explain(1, 3).startswith("Phim pho bien")

    def test_ramp_sai_bao_loi(self):
        with pytest.raises(ValueError):
            Hybrid(META, ramp="log")
        with pytest.raises(ValueError):
            Hybrid(META, ramp="co", ramp_n=-1)


class TestPhanVi:
    def test_phan_vi_giua_va_diem_bang_nhau(self):
        ref = np.sort(np.array([0.0, 0.0, 1.0, 2.0]))
        p = _percentile(ref, np.array([0.0, 1.0, 2.0, 5.0, -1.0]))
        np.testing.assert_allclose(p, [0.25, 0.625, 0.875, 1.0, 0.0])


class TestGiaiThich:
    def test_user_moi(self):
        assert hoc().explain(999, 1).startswith("Ban chua co lich su xem")

    def test_phim_moi(self):
        txt = hoc().explain(1, 9)
        assert txt.startswith("Phim moi, chua ai danh gia.")
        assert "Action" in txt

    def test_phim_da_co_nguoi_xem_lay_cau_cua_nhanh_dong_gop_nhieu_hon(self):
        assert hoc(alpha=0.0).explain(1, 3).startswith("Phim pho bien")
        assert hoc(alpha=1.0).explain(1, 3).startswith("Vi ban hay xem phim co")

    def test_khong_bang_chung_thi_rong(self):
        assert hoc().explain(1, 10) == ""
