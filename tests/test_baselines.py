"""Chot hai mo hinh B0 va hop dong `Recommender`.

Moi bac B1-B8 sau nay deu ke thua `Recommender`, nen cac bai trong
`TestHopDong` that ra dang chot hanh vi CHUNG cua ca tam bac -- khong chi
rieng B0.
"""

import numpy as np
import pandas as pd
import pytest

from recsys.models.baselines import Popularity, RandomRec

# Item 10 pho bien nhat (3 luot), roi 20 (2 luot), roi 30 (1 luot).
TRAIN = pd.DataFrame(
    {
        "user_id": [1, 1, 2, 2, 3, 3],
        "item_id": [10, 20, 10, 20, 10, 30],
        "rating": [5.0] * 6,
        "timestamp": range(6),
    }
)


class TestPopularity:
    def test_xep_hang_theo_do_pho_bien(self):
        m = Popularity().fit(TRAIN)
        scores = m.score(1, np.array([30, 10, 20]))
        # 10 co 3 luot > 20 co 2 luot > 30 co 1 luot
        assert scores[1] > scores[2] > scores[0]

    def test_item_LA_tra_ve_am_vo_cung_chu_khong_phai_0(self):
        """Tra ve 0 cho item chua tung thay la mot loi im lang: 0 co the CAO
        HON diem that cua item khac (vi du khi diem la log-odds am), va item
        la se leo len dau danh sach goi y ma khong gi bao."""
        m = Popularity().fit(TRAIN)
        assert m.score(1, np.array([999]))[0] == -np.inf

    def test_moi_user_nhan_cung_mot_xep_hang(self):
        """Popularity KHONG ca nhan hoa -- day chinh la dieu lam no thanh
        baseline tot: moi bac sau phai chung minh ca nhan hoa co dang cong
        suc hay khong."""
        m = Popularity().fit(TRAIN)
        items = np.array([10, 20, 30])
        assert np.array_equal(m.score(1, items), m.score(2, items))

    def test_explain_noi_ra_con_so(self):
        m = Popularity().fit(TRAIN)
        assert "3" in m.explain(1, 10)


class TestRandom:
    def test_lap_lai_duoc_giua_cac_lan_chay(self):
        items = np.array([10, 20, 30])
        a = RandomRec(seed=0).fit(TRAIN).score(1, items)
        b = RandomRec(seed=0).fit(TRAIN).score(1, items)
        assert np.array_equal(a, b)

    def test_hai_user_khac_nhau_ra_diem_khac_nhau(self):
        m = RandomRec(seed=0).fit(TRAIN)
        items = np.array([10, 20, 30])
        assert not np.array_equal(m.score(1, items), m.score(2, items))

    def test_seed_KHONG_chong_len_nhau_giua_cac_user(self):
        """Gieo bang `seed + user_id` thi (seed=1, user=1) trung (seed=0, user=2):
        doi seed chi xoay vong cung mot bo luong so, va trung binh tren moi
        user gan nhu khong doi. Chay "nhieu seed" khi do la chay mot lan."""
        items = np.arange(50)
        a = RandomRec(seed=1).fit(TRAIN).score(1, items)
        b = RandomRec(seed=0).fit(TRAIN).score(2, items)
        assert not np.array_equal(a, b)


class TestHopDong:
    """Hanh vi ma MOI bac B0-B8 phai co, do `Recommender` bao dam."""

    @pytest.mark.parametrize("model", [Popularity(), RandomRec()])
    def test_fit_ghi_nho_danh_muc_va_item_da_xem(self, model):
        model.fit(TRAIN)
        assert set(model.all_items_) == {10, 20, 30}
        assert model.seen_[1] == {10, 20}

    @pytest.mark.parametrize("model", [Popularity(), RandomRec()])
    def test_recommend_tra_ve_dung_k_item(self, model):
        model.fit(TRAIN)
        assert len(model.recommend(1, k=2, exclude_seen=False)) == 2

    @pytest.mark.parametrize("model", [Popularity(), RandomRec()])
    def test_recommend_loai_item_user_DA_xem(self, model):
        model.fit(TRAIN)
        rec = model.recommend(1, k=3, exclude_seen=True)
        # User 1 da xem 10 va 20 => chi con 30
        assert rec == [30]

    @pytest.mark.parametrize("model", [Popularity(), RandomRec()])
    def test_xin_nhieu_hon_so_item_con_lai_thi_khong_nem(self, model):
        model.fit(TRAIN)
        rec = model.recommend(1, k=99, exclude_seen=True)
        assert len(rec) == 1

    @pytest.mark.parametrize("model", [Popularity(), RandomRec()])
    def test_user_la_khong_nem(self, model):
        """User khong co trong train (cold start) phai ra duoc danh sach,
        khong duoc nem: ho la ca mot kich ban ung dung cua de tai."""
        model.fit(TRAIN)
        assert len(model.recommend(999, k=2)) == 2

    def test_recommend_xep_dung_thu_tu_giam_dan(self):
        m = Popularity().fit(TRAIN)
        assert m.recommend(999, k=3) == [10, 20, 30]

    @pytest.mark.parametrize("model", [Popularity(), RandomRec()])
    def test_explain_luon_tra_ve_chuoi(self, model):
        model.fit(TRAIN)
        assert isinstance(model.explain(1, 10), str)
