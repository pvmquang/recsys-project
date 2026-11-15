"""Chot cac thay doi dot 2 khong can bieu quyet: 2.4, 2.5, 2.8.

  2.4  danh gia tren toan danh muc   -- Split.full_candidates
  2.5  AUC xu ly the hoa + ti le -inf -- metrics.auc_from_scores, cot vo_cuc
  2.8  cold-start nhan tao           -- split.truncate_history
"""

import numpy as np
import pandas as pd
import pytest

from recsys.base import Recommender
from recsys.data.split import leave_one_out, truncate_history
from recsys.eval import metrics as m
from recsys.eval.runner import evaluate
from recsys.models.baselines import Popularity, RandomRec
from recsys.models.knn import ItemKNN, UserKNN
from tests.test_runner import make_ratings

SPLIT = leave_one_out(make_ratings(), n_negatives=20)


class ConstantRec(Recommender):
    """Moi ung vien cung mot diem -- mo hinh "khong biet gi"."""

    name = "hang-so"

    def __init__(self, value: float = 0.0):
        self.value = value

    def _fit(self, train):
        pass

    def score(self, user_id, item_ids):
        return np.full(len(item_ids), self.value)


# ---------------------------------------------------------------- 2.5 ----


class TestAucTuDiem:
    def test_hoan_hao_va_te_nhat(self):
        assert m.auc_from_scores([5.0], [1.0, 2.0, 3.0]) == 1.0
        assert m.auc_from_scores([0.0], [1.0, 2.0, 3.0]) == 0.0

    def test_cap_bang_diem_tinh_mot_nua(self):
        assert m.auc_from_scores([2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.5)

    def test_am_vo_cuc_bang_am_vo_cuc_la_hoa(self):
        """Ca item dung lan item sai deu -inf: khong mo hinh nao co y kien, AUC
        phai la 0.5 chu khong phai 0 hay 1 tuy luot xao."""
        ninf = -np.inf
        assert m.auc_from_scores([ninf], [ninf, ninf, 1.0]) == pytest.approx(1 / 3)

    def test_nan_nem_loi(self):
        with pytest.raises(ValueError):
            m.auc_from_scores([np.nan], [1.0])

    def test_khop_auc_tu_thu_tu_khi_khong_co_hoa(self):
        rng = np.random.default_rng(0)
        s = rng.random(50)
        items = np.arange(50)
        ranked = items[np.argsort(-s)].tolist()
        assert m.auc_from_scores(s[:1], s[1:]) == pytest.approx(m.auc(ranked, {0}))


class TestRunnerAuc:
    def test_mo_hinh_hang_so_ra_dung_0_5(self):
        """Truoc day AUC phu thuoc thu tu xao: mo hinh hang so ra AUC ngau nhien
        quanh 0.5 theo tung user. Gio phai DUNG 0.5 o moi user."""
        res = evaluate(ConstantRec().fit(SPLIT.train), SPLIT, keep_per_user=True)
        assert (res["per_user"]["auc"] == 0.5).all()

    def test_vo_cuc_cua_mo_hinh_hang_so_bang_0(self):
        res = evaluate(ConstantRec().fit(SPLIT.train), SPLIT)
        assert res["vo_cuc"] == 0.0

    def test_vo_cuc_bang_1_khi_moi_diem_am_vo_cuc(self):
        res = evaluate(ConstantRec(-np.inf).fit(SPLIT.train), SPLIT)
        assert res["vo_cuc"] == 1.0
        assert res["auc"] == 0.5


# ---------------------------------------------------------------- 2.4 ----


class TestToanDanhMuc:
    def test_item_dung_o_vi_tri_0(self):
        for u in SPLIT.users:
            assert SPLIT.full_candidates(u)[0] == SPLIT.test[u]

    def test_khong_chua_item_da_xem_va_khong_trung(self):
        seen = SPLIT.train.groupby("user_id")["item_id"].apply(set)
        for u in SPLIT.users:
            c = SPLIT.full_candidates(u)
            assert len(c) == len(set(c.tolist()))
            assert not (set(c.tolist()) & seen[u])

    def test_phu_toan_bo_danh_muc(self):
        seen = SPLIT.train.groupby("user_id")["item_id"].apply(set)
        for u in SPLIT.users:
            c = set(SPLIT.full_candidates(u).tolist())
            assert c | seen[u] == set(SPLIT.all_items.tolist())

    def test_chua_moi_item_am_lay_mau(self):
        for u in SPLIT.users:
            assert set(SPLIT.negatives[u].tolist()) <= set(SPLIT.full_candidates(u).tolist())

    def test_random_tren_toan_danh_muc_ra_auc_quanh_0_5(self):
        res = evaluate(RandomRec().fit(SPLIT.train), SPLIT, full_catalog=True)
        assert 0.35 < res["auc"] < 0.65

    def test_hit_rate_toan_danh_muc_khong_cao_hon_lay_mau(self):
        """Them ung vien chi co the day item dung xuong, khong keo len."""
        model = Popularity().fit(SPLIT.train)
        mau = evaluate(model, SPLIT, keep_per_user=True)["per_user"]
        full = evaluate(model, SPLIT, keep_per_user=True, full_catalog=True)["per_user"]
        assert (full["mrr"] <= mau["mrr"] + 1e-12).all()


# ---------------------------------------------------------------- 2.8 ----


class TestCatLichSu:
    USERS = SPLIT.users[:5]

    @pytest.mark.parametrize("n", [0, 1, 3, 5])
    def test_user_bi_cat_con_dung_n_tuong_tac(self, n):
        s = truncate_history(SPLIT, self.USERS, n)
        counts = s.train.groupby("user_id").size()
        for u in self.USERS:
            assert counts.get(u, 0) == n

    def test_giu_tuong_tac_GAN_NHAT(self):
        s = truncate_history(SPLIT, self.USERS, 3)
        for u in self.USERS:
            goc = SPLIT.train[SPLIT.train["user_id"] == u]
            moi = s.train[s.train["user_id"] == u]
            assert moi["timestamp"].min() >= goc["timestamp"].nlargest(3).min()

    def test_user_khac_va_tap_kiem_tra_giu_nguyen(self):
        s = truncate_history(SPLIT, self.USERS, 1)
        con_lai = ~SPLIT.train["user_id"].isin(self.USERS)
        pd.testing.assert_frame_equal(
            s.train[~s.train["user_id"].isin(self.USERS)], SPLIT.train[con_lai]
        )
        assert s.test == SPLIT.test
        assert s.negatives is SPLIT.negatives

    def test_khong_doi_split_goc(self):
        n = len(SPLIT.train)
        truncate_history(SPLIT, self.USERS, 1)
        assert len(SPLIT.train) == n

    def test_full_candidates_tinh_lai_theo_train_moi(self):
        """cached_property khong duoc mang lich su cu sang split da cat."""
        _ = SPLIT.full_candidates(self.USERS[0])
        s = truncate_history(SPLIT, self.USERS, 1)
        assert len(s.full_candidates(self.USERS[0])) > len(SPLIT.full_candidates(self.USERS[0]))

    def test_user_knn_mot_tuong_tac_khong_co_lang_gieng(self):
        """Pearson cua user chi co 1 rating: vector sau khi tru trung binh = 0
        => khong lang gieng => moi diem -inf. Day la can cu that cho B7."""
        s = truncate_history(SPLIT, self.USERS, 1)
        model = UserKNN(k=10).fit(s.train)
        for u in self.USERS:
            assert np.isneginf(model.score(u, s.candidates(u))).all()

    def test_item_knn_van_cham_duoc_voi_mot_tuong_tac(self):
        s = truncate_history(SPLIT, self.USERS, 1)
        model = ItemKNN(k=10).fit(s.train)
        assert any(np.isfinite(model.score(u, s.candidates(u))).any() for u in self.USERS)
