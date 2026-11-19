"""Chot B5 -- goi y theo noi dung (recsys/models/content.py).

Du lieu do choi: phim 1-4 la Action, phim 5-8 la Romance, phim 9 la Action
nhung KHONG AI XEM (phim moi), phim 10 khong co the loai nao.
"""

import numpy as np
import pandas as pd
import pytest

from recsys.models.content import ContentBased, decade_token

META = pd.DataFrame(
    {
        "item_id": list(range(1, 11)),
        "title": [f"Phim {i} ({1980 + i})" for i in range(1, 11)],
        "genres": [["Action"]] * 4 + [["Romance"]] * 4 + [["Action"], []],
        "directors": [["James Cameron"], [], [], [], ["Nora Ephron"], [], [], [], [], []],
        "cast": [[] for _ in range(10)],
        "keywords": [[] for _ in range(10)],
        "overview": [""] * 10,
    }
)

# User 1 xem phim Action, user 2 xem phim Romance. User 3 cham phim Action thap.
TRAIN = pd.DataFrame(
    [
        (1, 1, 5.0, 1), (1, 2, 4.0, 2),
        (2, 5, 5.0, 1), (2, 6, 4.0, 2),
        (3, 1, 1.0, 1), (3, 5, 5.0, 2), (3, 2, 1.0, 3),
    ],
    columns=["user_id", "item_id", "rating", "timestamp"],
)


def hoc(**kw) -> ContentBased:
    kw.setdefault("fields", ("genres",))
    return ContentBased(META, **kw).fit(TRAIN)


class TestCoBan:
    def test_user_xem_action_xep_action_tren_romance(self):
        m = hoc()
        s = m.score(1, np.array([3, 4, 7, 8]))
        assert s[:2].min() > s[2:].max()

    def test_phim_chua_ai_xem_van_co_diem(self):
        """Khac B1-B4: phim 9 khong co trong train nhung co the loai."""
        m = hoc()
        assert 9 not in set(TRAIN["item_id"])
        s = m.score(1, np.array([9]))
        assert np.isfinite(s[0]) and s[0] > 0

    def test_phim_khong_dac_trung_user_la_phim_la_ra_am_vo_cuc(self):
        m = hoc()
        assert np.isneginf(m.score(1, np.array([10, 999]))).all()
        assert np.isneginf(m.score(999, np.array([1, 2]))).all()

    def test_khong_chung_dac_trung_nao_ra_am_vo_cuc_khong_phai_0(self):
        """Diem 0 se xep tren phim co diem am; -inf la "khong co bang chung"."""
        m = hoc()
        assert np.isneginf(m.score(1, np.array([7]))[0])

    def test_recommend_tra_phim_moi_va_bo_phim_khong_lien_quan(self):
        m = hoc()
        rec = m.recommend(1, k=10)
        assert set(rec) <= {3, 4}  # recommend chi duyet all_items_ = phim trong train
        assert 7 not in rec

    def test_fields_sai_bao_loi(self):
        with pytest.raises(ValueError):
            ContentBased(META, fields=("genres", "poster"))


class TestTrongSo:
    def test_rating_lam_phim_cham_thap_keo_ho_so_ra_xa(self):
        """User 3 cham Action 1 sao, Romance 5 sao: weighting='rating' dao thu tu."""
        nhi_phan = hoc(weighting="binary").score(3, np.array([3, 7]))
        theo_rating = hoc(weighting="rating").score(3, np.array([3, 7]))
        assert nhi_phan[0] > nhi_phan[1]  # 2 phim Action > 1 phim Romance
        assert theo_rating[1] > theo_rating[0]

    def test_half_life_uu_tien_phim_xem_gan_nhat(self):
        # User 4 xem hai phim Action roi mot phim Romance. Khong giam thi Action
        # thang 2-1; giam nhanh thi phim cuoi (Romance) lan at.
        train = pd.concat([TRAIN, pd.DataFrame(
            [(4, 1, 4.0, 1), (4, 2, 4.0, 2), (4, 5, 4.0, 3)],
            columns=TRAIN.columns)])
        khong = ContentBased(META, fields=("genres",)).fit(train).score(4, np.array([3, 7]))
        nhanh = ContentBased(META, fields=("genres",), half_life=0.25).fit(train).score(
            4, np.array([3, 7]))
        assert khong[0] > khong[1]
        assert nhanh[1] > nhanh[0]

    def test_user_cham_deu_mot_muc_khong_mat_ho_so(self):
        train = pd.DataFrame([(7, 1, 3.0, 1), (7, 2, 3.0, 2)], columns=TRAIN.columns)
        m = ContentBased(META, fields=("genres",), weighting="rating").fit(train)
        assert np.isfinite(m.score(7, np.array([3]))[0])


class TestGiaiThich:
    def test_contributions_cong_lai_bang_dung_score(self):
        m = hoc(fields=("genres", "year", "directors"), weighting="rating", half_life=2)
        for u, i in ((1, 3), (3, 7), (2, 9)):
            s = m.score(u, np.array([i]))[0]
            if np.isfinite(s):
                assert sum(m.contributions(u, i).values()) == pytest.approx(s)
                assert sum(m.shared_features(u, i).values()) == pytest.approx(s)

    def test_explain_neu_dac_trung_va_phim_da_xem(self):
        m = hoc(fields=("genres",), titles={1: "Alien", 2: "Aliens"})
        txt = m.explain(1, 3)
        assert txt.startswith("Vi ban hay xem phim co: Action")
        assert "Alien" in txt

    def test_explain_hien_ten_nguoi_dung_chinh_ta(self):
        meta = META.copy()
        meta.at[2, "directors"] = ["James Cameron"]  # phim 3 cung dao dien phim 1
        m = ContentBased(meta, fields=("directors",)).fit(TRAIN)
        assert "James Cameron" in m.explain(1, 3)

    def test_explain_rong_khi_khong_lien_quan(self):
        m = hoc()
        assert m.explain(1, 7) == ""
        assert m.explain(999, 1) == ""


class TestDacTrung:
    def test_decade_token(self):
        assert decade_token("Alien (1979)") == "decade_1970"
        assert decade_token("Toy Story (1995) ") == "decade_1990"
        assert decade_token("Khong co nam") is None

    def test_nam_lam_phim_cung_thap_nien_gan_hon(self):
        meta = META.copy()
        meta["genres"] = [["Drama"]] * 10
        meta["title"] = ["A (1975)", "B (1978)", "C (1979)", "D (2015)"] + [
            f"X ({2000 + i})" for i in range(6)]
        m = ContentBased(meta, fields=("genres", "year")).fit(TRAIN)
        s = m.score(1, np.array([3, 4]))
        assert s[0] > s[1]

    def test_tach_ten_nguoi_tao_tuong_dong_gia(self):
        """Loi cua ma nguon tham khao: 'Chris Evans' va 'Chris Pratt' chung token 'chris'."""
        meta = META.copy()
        meta["cast"] = [["Chris Evans"], [], ["Chris Pratt"]] + [[] for _ in range(7)]
        noi = ContentBased(meta, fields=("cast",)).fit(TRAIN)
        tach = ContentBased(meta, fields=("cast",), join_names=False).fit(TRAIN)
        assert np.isneginf(noi.score(1, np.array([3]))[0])
        assert np.isfinite(tach.score(1, np.array([3]))[0])
