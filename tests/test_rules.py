"""Chot B6 -- luat ket hop theo phien (recsys/models/rules.py).

Du lieu do choi: 10 user, moi user mot phien sang (phim 1, 2, 3 hoac 1, 2) va mot
phien chieu (phim 4, 5, 6 hoac 4, 5). Hai phien cach nhau 1 ngay. Phim 9 chi do
user 10 cham, rieng mot phien.
"""

import numpy as np
import pandas as pd
import pytest

from recsys.models.rules import AssociationRules, session_ids

NGAY = 86_400


def lam_train() -> pd.DataFrame:
    rows = []
    for u in range(1, 11):
        sang = [1, 2, 3] if u <= 6 else [1, 2]
        chieu = [4, 5, 6] if u <= 4 else [4, 5]
        for t, i in enumerate(sang):
            rows.append((u, i, 4.0, t * 60))
        for t, i in enumerate(chieu):
            rows.append((u, i, 4.0, NGAY + t * 60))
    rows.append((10, 9, 4.0, 3 * NGAY))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


TRAIN = lam_train()


def hoc(**kw) -> AssociationRules:
    kw.setdefault("min_support", 0.2)
    kw.setdefault("session_gap", 1800)
    kw.setdefault("context", "session")  # test viet cho ngu canh phien, khong theo hang so
    return AssociationRules(**kw).fit(TRAIN)


class TestPhien:
    def test_cat_phien_theo_khoang_nghi(self):
        df, sid = session_ids(TRAIN, gap=1800)
        u1 = sid[df["user_id"].to_numpy() == 1]
        assert len(np.unique(u1)) == 2
        u10 = sid[df["user_id"].to_numpy() == 10]
        assert len(np.unique(u10)) == 3  # sang, chieu, phim 9 rieng

    def test_gap_lon_gop_thanh_mot_phien(self):
        df, sid = session_ids(TRAIN, gap=2 * NGAY)
        assert len(np.unique(sid[df["user_id"].to_numpy() == 1])) == 1

    def test_phien_mot_phim_khong_tinh_la_giao_dich(self):
        m = hoc()
        assert m.n_sessions_ == 20  # 10 user x 2 phien, bo phien {9} cua user 10


class TestLuat:
    def test_confidence_dung_dinh_nghia(self):
        """3 -> 2: moi phien co phim 3 deu co phim 2 => confidence 1."""
        m = hoc()
        r = m.rules_
        dong = r[(r["antecedents"] == frozenset({m.iidx_[3]}))
                 & (r["consequents"] == frozenset({m.iidx_[2]}))]
        assert dong["confidence"].iloc[0] == pytest.approx(1.0)
        # 1 -> 3: 6/10 phien co phim 1 co phim 3.
        dong = r[(r["antecedents"] == frozenset({m.iidx_[1]}))
                 & (r["consequents"] == frozenset({m.iidx_[3]}))]
        assert dong["confidence"].iloc[0] == pytest.approx(0.6)

    def test_khong_co_luat_giua_hai_phien_khac_nhau(self):
        """Phim 1 va 4 cung user nhung khac phien: khong co luat (khac ItemKNN)."""
        m = hoc()
        r = m.rules_
        sai = r[(r["antecedents"] == frozenset({m.iidx_[1]}))
                & (r["consequents"] == frozenset({m.iidx_[4]}))]
        assert sai.empty

    def test_max_len_3_sinh_luat_ve_trai_hai_phim(self):
        m = hoc(max_len=3)
        assert (m.ante_size_ == 2).any()
        assert (hoc(max_len=2).ante_size_ == 1).all()


class TestChamDiem:
    def test_ngu_canh_la_phien_cuoi(self):
        """User 7 vua xem phien chieu {4, 5}: phim 6 co diem, phim 3 thi khong."""
        m = hoc()
        s = m.score(7, np.array([6, 3]))
        assert np.isfinite(s[0]) and np.isneginf(s[1])

    def test_ngu_canh_n_phim_gan_nhat(self):
        """context=5 voi user 7: {1, 2, 4, 5} -> phim 3 va 6 deu co diem."""
        m = hoc(context=5)
        assert np.isfinite(m.score(7, np.array([3, 6]))).all()

    def test_phim_la_user_la_va_khong_luat_ra_am_vo_cuc(self):
        m = hoc()
        assert np.isneginf(m.score(7, np.array([999, 9]))).all()
        assert np.isneginf(m.score(999, np.array([1, 6]))).all()

    def test_khong_goi_y_phim_ngay_trong_ngu_canh(self):
        m = hoc()
        assert np.isneginf(m.score(7, np.array([4, 5]))).all()

    def test_lift_va_confidence(self):
        """max: diem la luat manh nhat; sum: cong cac luat."""
        m = hoc(agg="max")
        tong = hoc(agg="sum")
        a, b = m.score(7, np.array([6]))[0], tong.score(7, np.array([6]))[0]
        assert b > a  # hai luat 4->6 va 5->6 cung khop
        assert hoc(metric="lift").score(7, np.array([6]))[0] > 1

    def test_recommend_chi_tra_phim_co_luat(self):
        m = hoc()
        assert m.recommend(7, k=10) == [6]

    def test_khong_du_support_thi_moi_diem_am_vo_cuc(self):
        m = hoc(min_support=0.99)
        assert np.isneginf(m.score(1, np.array([1, 2, 3, 6]))).all()
        assert m.recommend(1) == []

    def test_tham_so_sai_bao_loi(self):
        for kw in ({"metric": "jaccard"}, {"agg": "mean"}, {"context": 0}, {"max_len": 1}):
            with pytest.raises(ValueError):
                AssociationRules(**kw)


class TestGiaiThich:
    def test_explain_neu_luat_va_so_phien(self):
        m = hoc(titles={4: "Alien", 5: "Aliens", 6: "Alien 3"})
        b = m.best_rule(7, 6)
        assert b["so_phien"] == 4 and b["confidence"] == pytest.approx(0.4)
        txt = m.explain(7, 6)
        assert txt.startswith("Nguoi cham Alien")
        assert "do tin cay 40%, 4 phien" in txt

    def test_explain_rong_khi_khong_co_luat(self):
        m = hoc()
        assert m.explain(7, 3) == ""
        assert m.explain(999, 6) == ""


class TestRescore:
    def test_rescore_bang_fit_lai(self):
        """Script chon sieu tham so dua vao dang thuc nay."""
        moi = hoc(context=5, metric="lift", agg="sum")
        doi = hoc().rescore(context=5, metric="lift", agg="sum")
        ung_vien = np.array([1, 2, 3, 4, 5, 6, 9])
        for u in range(1, 11):
            assert np.array_equal(moi.score(u, ung_vien), doi.score(u, ung_vien))

    def test_rescore_xoa_bo_nho_dem(self):
        m = hoc()
        truoc = m.score(7, np.array([3]))[0]
        m.rescore(context=5)
        assert np.isneginf(truoc) and np.isfinite(m.score(7, np.array([3]))[0])

    def test_rescore_kiem_tham_so(self):
        with pytest.raises(ValueError):
            hoc().rescore(agg="mean")


class TestFoldIn:
    def test_fold_in_lai_chinh_train_khong_doi_gi(self):
        m = hoc()
        ung_vien = np.array([1, 2, 3, 4, 5, 6, 9])
        truoc = {u: m.score(u, ung_vien) for u in range(1, 11)}
        m.fold_in(TRAIN)
        for u in range(1, 11):
            assert np.array_equal(truoc[u], m.score(u, ung_vien))

    def test_user_moi_dung_luat_cu(self):
        """User 50 khong co luc khai pha: vua cham phim 4 => phim 5, 6 co diem."""
        m = hoc()
        assert np.isneginf(m.score(50, np.array([5, 6]))).all()
        moi = pd.DataFrame([(50, 4, 4.0, 0), (50, 777, 4.0, 10)], columns=TRAIN.columns)
        m.fold_in(moi)
        assert np.isfinite(m.score(50, np.array([5, 6]))).all()
        assert m.seen_[50] == {4, 777}
        assert m.recommend(50, k=5) and 4 not in m.recommend(50, k=5)

    def test_fold_in_thay_han_lich_su_cu_cua_user(self):
        """User 7 cat con 1 phim (phim 1): ngu canh khong con phien {4, 5}."""
        m = hoc()
        m.fold_in(pd.DataFrame([(7, 1, 4.0, 0)], columns=TRAIN.columns))
        s = m.score(7, np.array([2, 6]))
        assert np.isfinite(s[0]) and np.isneginf(s[1])
