"""Chot hai thay doi giao thuc chung cua dot 2 (ca nhom da dong y).

  2.1  tap validation = tuong tac ap chot   -- split.validation_split
  2.3  pha hoa timestamp ngau nhien theo seed -- split.sort_with_random_ties
"""

import numpy as np
import pandas as pd
import pytest

from recsys.data.split import leave_one_out, sort_with_random_ties, validation_split
from tests.test_runner import make_ratings


def ratings_trung_timestamp(n_users: int = 40, n_hoa: int = 6) -> pd.DataFrame:
    """Moi user: 10 tuong tac timestamp tang dan, roi n_hoa phim CUNG timestamp cuoi."""
    rows = []
    for u in range(1, n_users + 1):
        for j in range(10):
            rows.append((u, 1000 + u * 50 + j, 4.0, j))
        for j in range(n_hoa):
            rows.append((u, 5000 + j, 4.0, 99))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


# ---------------------------------------------------------------- 2.3 ----


class TestPhaHoa:
    def test_khong_con_luon_chon_item_id_lon_nhat(self):
        """Ban truoc: trung timestamp thi item kiem tra LUON la item_id lon nhat
        (5005), lech ve phim moi. Gio phai trai deu tren ca nhom bi hoa."""
        s = leave_one_out(ratings_trung_timestamp(), n_negatives=5)
        chon = list(s.test.values())
        assert set(chon) <= set(range(5000, 5006))
        assert len(set(chon)) >= 4

    def test_van_lay_trong_nhom_timestamp_CUOI(self):
        s = leave_one_out(ratings_trung_timestamp(), n_negatives=5)
        assert all(5000 <= i <= 5005 for i in s.test.values())

    def test_cung_seed_cung_ket_qua_ke_ca_khi_xao_dong(self):
        r = ratings_trung_timestamp()
        a = leave_one_out(r, n_negatives=5, seed=42)
        b = leave_one_out(r.sample(frac=1.0, random_state=3), n_negatives=5, seed=42)
        assert a.test == b.test

    def test_khac_seed_co_the_chon_khac(self):
        r = ratings_trung_timestamp()
        a = leave_one_out(r, n_negatives=5, seed=1)
        b = leave_one_out(r, n_negatives=5, seed=2)
        assert a.test != b.test

    def test_khong_hoa_thi_khong_doi_gi(self):
        r = make_ratings()  # timestamp khong trung
        s = sort_with_random_ties(r, seed=42)
        cu = r.sort_values(["user_id", "timestamp"], kind="mergesort")
        assert s.index.equals(cu.index)

    def test_pha_hoa_khong_doi_item_am(self):
        """Luong ngau nhien rieng: item am phai giong het khi KHONG co hoa nao
        -- va item am khong phu thuoc phim nao bi chon lam test."""
        r = ratings_trung_timestamp()
        a = leave_one_out(r, n_negatives=5, seed=42)
        rng = np.random.default_rng(42)
        all_items = np.sort(r["item_id"].unique())
        for u in a.test:
            seen = set(r.loc[r["user_id"] == u, "item_id"])
            pool = all_items[~np.isin(all_items, list(seen))]
            assert np.array_equal(a.negatives[u], rng.choice(pool, size=5, replace=False))


# ---------------------------------------------------------------- 2.1 ----

SPLIT = leave_one_out(make_ratings(), n_negatives=20)
VAL = validation_split(SPLIT)


class TestValidation:
    def test_item_validation_la_tuong_tac_AP_CHOT(self):
        r = make_ratings()
        for u in VAL.users:
            cua_user = r[r["user_id"] == u].sort_values("timestamp")
            assert VAL.test[u] == cua_user["item_id"].iloc[-2]

    def test_train_validation_bang_train_tru_item_validation(self):
        assert len(VAL.train) == len(SPLIT.train) - len(VAL.test)
        cap = set(zip(VAL.train["user_id"], VAL.train["item_id"]))
        for u, i in VAL.test.items():
            assert (u, i) not in cap

    def test_KHONG_nhin_thay_item_kiem_tra(self):
        """Item kiem tra khong duoc o trong train cua validation, va khong duoc
        lam item am cua validation (user DA xem no)."""
        cap = set(zip(VAL.train["user_id"], VAL.train["item_id"]))
        for u in VAL.users:
            t = SPLIT.test[u]
            assert (u, t) not in cap
            assert t not in set(VAL.negatives[u].tolist())

    def test_item_am_khong_trung_item_da_xem(self):
        r = make_ratings()
        for u, negs in VAL.negatives.items():
            seen = set(r.loc[r["user_id"] == u, "item_id"])
            assert not (set(negs.tolist()) & seen)
            assert len(negs) == 20

    def test_lap_lai_duoc_va_dung_seed_cua_split(self):
        again = validation_split(leave_one_out(make_ratings(), n_negatives=20))
        assert again.test == VAL.test
        for u in VAL.users:
            assert np.array_equal(again.negatives[u], VAL.negatives[u])
        khac = validation_split(leave_one_out(make_ratings(), n_negatives=20, seed=7))
        assert any(
            not np.array_equal(khac.negatives[u], VAL.negatives[u]) for u in VAL.users
        )

    def test_khop_cach_pha_hoa_cua_tap_kiem_tra(self):
        """Hai phim trung timestamp cuoi: mot phim lam test, phim con lai phai
        lam validation (khong phai mot phim som hon)."""
        r = ratings_trung_timestamp(n_hoa=2)
        s = leave_one_out(r, n_negatives=5)
        v = s.validation()
        for u in v.users:
            assert {s.test[u], v.test[u]} == {5000, 5001}

    def test_user_qua_it_lich_su_bi_bo(self):
        r = pd.DataFrame(
            {"user_id": [1, 1, 2, 2, 2, 2], "item_id": [1, 2, 1, 2, 3, 4],
             "rating": 4.0, "timestamp": [1, 2, 1, 2, 3, 4]}
        )
        r = pd.concat([r, make_ratings(n_users=3, n_items=200)
                       .assign(user_id=lambda d: d["user_id"] + 10)])
        s = leave_one_out(r, n_negatives=5, min_interactions=2)
        v = s.validation()
        assert 1 not in v.test  # con dung 1 tuong tac train
        assert 2 in v.test

    def test_chay_duoc_voi_evaluate(self):
        from recsys.eval.runner import evaluate
        from recsys.models.baselines import Popularity

        res = evaluate(Popularity().fit(VAL.train), VAL)
        assert res["n_users"] == len(VAL.users)
        assert 0.0 <= res["ndcg@10"] <= 1.0
