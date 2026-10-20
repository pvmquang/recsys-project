"""Chot phep chia du lieu.

Day la module NGUY HIEM NHAT trong ca du an. Chia sai khong nem loi: no chi
lam moi con so trong bao cao lech di, va lech CANG NHIEU o dung nhung mo hinh
manh nhat -- nen bang xep hang cuoi cung sai ma van "trong hop ly".

Ba loi cu the ma cac bai duoi chan:
  1. Item kiem tra ro ri vao tap huan luyen  -> mo hinh doan duoc dieu no da thay
  2. Item am trung voi item user da xem      -> "item am" that ra la item duong
  3. Lay tuong tac ngau nhien thay vi cuoi cung -> mo hinh nhin thay tuong lai
"""

import numpy as np
import pandas as pd
import pytest

from recsys.data.split import Split, leave_one_out, user_activity_groups


def make_ratings(n_users: int = 6, n_items: int = 40, per_user: int = 10):
    """Du lieu gia: moi user danh gia `per_user` item, timestamp tang dan."""
    rows = []
    for u in range(1, n_users + 1):
        for j in range(per_user):
            rows.append(
                {
                    "user_id": u,
                    "item_id": (u * 3 + j) % n_items + 1,
                    "rating": 4.0,
                    # Timestamp tang dan theo j => item cuoi cung xac dinh duoc
                    "timestamp": 1000 + j * 10,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["user_id", "item_id"])


class TestLeaveOneOut:
    def test_moi_user_giu_lai_dung_mot_item(self):
        s = leave_one_out(make_ratings(), n_negatives=5, min_interactions=3)
        assert len(s.test) == 6
        assert all(isinstance(v, (int, np.integer)) for v in s.test.values())

    def test_item_kiem_tra_KHONG_nam_trong_tap_huan_luyen(self):
        """Loi ro ri so mot: mo hinh duoc cham diem tren thu no da hoc."""
        s = leave_one_out(make_ratings(), n_negatives=5, min_interactions=3)
        for user_id, item_id in s.test.items():
            seen_in_train = set(
                s.train.loc[s.train["user_id"] == user_id, "item_id"]
            )
            assert item_id not in seen_in_train, (
                f"User {user_id}: item kiem tra {item_id} van nam trong train"
            )

    def test_lay_tuong_tac_CUOI_CUNG_theo_thoi_gian(self):
        """Lay ngau nhien thay vi cuoi cung = cho mo hinh nhin thay tuong lai."""
        ratings = make_ratings()
        s = leave_one_out(ratings, n_negatives=5, min_interactions=3)
        for user_id in s.users:
            cua_user = ratings[ratings["user_id"] == user_id]
            muon_nhat = cua_user.sort_values("timestamp").iloc[-1]["item_id"]
            assert s.test[user_id] == muon_nhat

    def test_thu_tu_dong_trong_file_KHONG_anh_huong_ket_qua(self):
        """Xao tron dong roi chia lai phai ra dung mot ket qua. Khong dam bao
        duoc dieu nay thi lan chia doi khi doi theo thu tu doc file, va hai
        thanh vien chay cung lenh se ra hai bo so khac nhau."""
        ratings = make_ratings()
        a = leave_one_out(ratings, n_negatives=5, min_interactions=3)
        xao = ratings.sample(frac=1.0, random_state=7).reset_index(drop=True)
        b = leave_one_out(xao, n_negatives=5, min_interactions=3)
        assert a.test == b.test

    def test_item_am_KHONG_trung_item_user_da_xem(self):
        """Loi ro ri so hai: 'item am' that ra la item user da thich."""
        ratings = make_ratings()
        s = leave_one_out(ratings, n_negatives=5, min_interactions=3)
        for user_id, negs in s.negatives.items():
            seen = set(ratings.loc[ratings["user_id"] == user_id, "item_id"])
            assert not (set(negs.tolist()) & seen), (
                f"User {user_id}: item am trung voi item da xem"
            )

    def test_dung_so_luong_item_am(self):
        s = leave_one_out(make_ratings(), n_negatives=7, min_interactions=3)
        assert all(len(v) == 7 for v in s.negatives.values())

    def test_item_am_khong_trung_lap_nhau(self):
        s = leave_one_out(make_ratings(), n_negatives=7, min_interactions=3)
        for negs in s.negatives.values():
            assert len(set(negs.tolist())) == len(negs)

    def test_khong_du_item_am_thi_NEM_chu_khong_im_lang_tra_it_hon(self):
        """Tra ve it item am hon la lam mau so cua moi do do khac nhau giua
        cac user, va khong gi tren bang ket qua noi ra dieu do."""
        ratings = make_ratings(n_users=2, n_items=12, per_user=10)
        with pytest.raises(ValueError, match="item am"):
            leave_one_out(ratings, n_negatives=99, min_interactions=3)


class TestMinInteractions:
    def test_user_qua_it_tuong_tac_bi_loai_khoi_TAP_KIEM_TRA(self):
        ratings = pd.concat(
            [
                make_ratings(n_users=3, per_user=10),
                # User 99 chi co 2 tuong tac
                pd.DataFrame(
                    {
                        "user_id": [99, 99],
                        "item_id": [31, 32],
                        "rating": [5.0, 5.0],
                        "timestamp": [1, 2],
                    }
                ),
            ]
        )
        s = leave_one_out(ratings, n_negatives=5, min_interactions=5)
        assert 99 not in s.test

    def test_nhung_tuong_tac_cua_ho_VAN_o_trong_tap_huan_luyen(self):
        """Loai khoi tap kiem tra khong co nghia la vut du lieu: tuong tac cua
        ho van giup cac mo hinh loc cong tac hoc ve item."""
        ratings = pd.concat(
            [
                make_ratings(n_users=3, per_user=10),
                pd.DataFrame(
                    {
                        "user_id": [99, 99],
                        "item_id": [31, 32],
                        "rating": [5.0, 5.0],
                        "timestamp": [1, 2],
                    }
                ),
            ]
        )
        s = leave_one_out(ratings, n_negatives=5, min_interactions=5)
        assert (s.train["user_id"] == 99).sum() == 2


class TestLapLaiDuoc:
    """Ba thanh vien chay tren ba may phai ra CUNG mot lan chia.

    Khong thi so cua ba nguoi khong so sanh duoc voi nhau, va khong co gi
    tren bang ket qua noi ra dieu do.
    """

    def test_cung_seed_ra_cung_ket_qua(self):
        r = make_ratings()
        a = leave_one_out(r, n_negatives=5, min_interactions=3, seed=42)
        b = leave_one_out(r, n_negatives=5, min_interactions=3, seed=42)
        assert a.test == b.test
        for u in a.negatives:
            assert np.array_equal(a.negatives[u], b.negatives[u])

    def test_khac_seed_ra_item_am_khac(self):
        r = make_ratings()
        a = leave_one_out(r, n_negatives=5, min_interactions=3, seed=1)
        b = leave_one_out(r, n_negatives=5, min_interactions=3, seed=2)
        khac = any(
            not np.array_equal(a.negatives[u], b.negatives[u]) for u in a.negatives
        )
        assert khac

    def test_item_kiem_tra_KHONG_doi_theo_seed(self):
        """Item kiem tra do THOI GIAN quyet dinh, khong do ngau nhien.
        Doi theo seed nghia la phep chia dang lay ngau nhien o dau do."""
        r = make_ratings()
        a = leave_one_out(r, n_negatives=5, min_interactions=3, seed=1)
        b = leave_one_out(r, n_negatives=5, min_interactions=3, seed=2)
        assert a.test == b.test


class TestCandidates:
    def test_dung_mot_cong_chin_muoi_chin(self):
        s = leave_one_out(make_ratings(), n_negatives=9, min_interactions=3)
        for u in s.users:
            assert len(s.candidates(u)) == 10

    def test_item_dung_luon_o_vi_tri_0(self):
        """Runner PHAI xao truoc khi xep hang. Bai nay ghi lai tinh chat do
        de ai doc runner hieu vi sao co dong permutation o day."""
        s = leave_one_out(make_ratings(), n_negatives=5, min_interactions=3)
        for u in s.users:
            assert s.candidates(u)[0] == s.test[u]


class TestUserActivityGroups:
    def test_xep_dung_nhom(self):
        train = pd.DataFrame(
            {
                "user_id": [1] * 3 + [2] * 10 + [3] * 50 + [4] * 200,
                "item_id": range(263),
            }
        )
        g = user_activity_groups(train)
        assert g[1] == "<5"
        assert g[2] == "5-20"
        assert g[3] == "20-100"
        assert g[4] == ">100"

    def test_MOVIELENS_KHONG_CO_USER_NHOM_DUOI_5(self):
        """Ghi lai mot su that ve du lieu, khong phai ve ma nguon.

        MovieLens chi nhan user da danh gia IT NHAT 20 phim. Do that tren
        ml-latest-small: min = 20 danh gia, 0/610 user duoi nguong do.

        => Nhom '<5' LUON RONG, va bang theo_nhom KHONG tra loi duoc cau hoi
        cold-start that su. Muon do cold-start thi phai TAO RA no: cat bot
        lich su huan luyen cua mot phan user xuong con 1, 3, 5 tuong tac.

        Bai nay khong goi mang. No chi chot rang phep xep nhom PHAN BIET duoc
        hai ca do, de khi co du lieu that thi nhom '<5' rong la mot ket qua
        doc duoc chu khong phai mot loi im lang.
        """
        train = pd.DataFrame({"user_id": [1] * 20 + [2] * 30, "item_id": range(50)})
        g = user_activity_groups(train)
        assert "<5" not in set(g.values())


def test_summary_khong_nem():
    s = leave_one_out(make_ratings(), n_negatives=5, min_interactions=3)
    assert "train" in s.summary()
    assert isinstance(s, Split)
