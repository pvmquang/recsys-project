"""Chot bo chay danh gia.

Bang ket qua la thu di thang vao slide va bao cao. Mot cot ghi sai k, hay mot
nhom user dat nhan sai, khong nem loi -- no chi lam nguoi doc ket luan sai.
"""

import numpy as np
import pandas as pd

from recsys.data.split import leave_one_out
from recsys.eval.runner import MAIN_K, _by_activity, evaluate, format_table
from recsys.models.baselines import Popularity, RandomRec


def make_ratings(n_users: int = 30, n_items: int = 200, seed: int = 0) -> pd.DataFrame:
    """User i co 20 + 3*i tuong tac -- du cho nhom phan vi, user nao cung >= 20
    giong MovieLens."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        items = rng.choice(n_items, size=20 + 3 * u, replace=False) + 1
        for t, it in enumerate(items):
            rows.append((u, int(it), 4.0, t))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


SPLIT = leave_one_out(make_ratings(), n_negatives=20)


class TestCoverage:
    def test_co_coverage_cho_TUNG_k(self):
        """Mot cot 'coverage' khong ghi k da bi dat canh cot @10 trong khi no
        la @20. Moi k phai co cot rieng, ten ghi ro k."""
        res = evaluate(Popularity().fit(SPLIT.train), SPLIT, ks=(5, 10, 20))
        assert "coverage" not in res
        assert {"coverage@5", "coverage@10", "coverage@20"} <= set(res)

    def test_coverage_tang_theo_k(self):
        res = evaluate(RandomRec().fit(SPLIT.train), SPLIT, ks=(5, 10, 20))
        assert res["coverage@5"] <= res["coverage@10"] <= res["coverage@20"]

    def test_popularity_coverage_at_k_dung_gia_tri(self):
        """Popularity goi y gan nhu cung mot danh sach cho moi user (chi khac phan
        da xem bi loai), nen coverage@k phai rat nho so voi danh muc."""
        m = Popularity().fit(SPLIT.train)
        res = evaluate(m, SPLIT, ks=(5, 10))
        tung_goi_y = {i for u in SPLIT.users for i in m.recommend(u, k=10)}
        assert res["coverage@10"] == len(tung_goi_y) / len(m.all_items_)

    def test_bang_in_ra_ghi_ro_k_cua_coverage(self):
        res = evaluate(Popularity().fit(SPLIT.train), SPLIT)
        txt = format_table(pd.DataFrame([res]).set_index("model"))
        assert f"coverage@{MAIN_K}" in txt


class TestPerUser:
    def test_mac_dinh_KHONG_tra_per_user(self):
        """Ket qua mac dinh la dict phang -- de vao thang DataFrame."""
        res = evaluate(Popularity().fit(SPLIT.train), SPLIT)
        assert "per_user" not in res

    def test_per_user_danh_chi_so_user_id(self):
        res = evaluate(Popularity().fit(SPLIT.train), SPLIT, keep_per_user=True)
        pu = res["per_user"]
        assert set(pu.index) == set(SPLIT.users)
        assert pu[f"ndcg@{MAIN_K}"].mean() == res[f"ndcg@{MAIN_K}"]


class TestTheoNhom:
    def test_chia_PHAN_VI_nhom_nao_cung_du_dong(self):
        """Nguong co dinh (5, 20, 100) tren du lieu user nao cung >= 20 tuong tac
        cho mot nhom 29 user ma nhan doc nham thanh 'user it lich su'."""
        res = evaluate(
            Popularity().fit(SPLIT.train), SPLIT, by_activity=True, keep_per_user=True
        )
        bang = res["theo_nhom"]
        assert len(bang) == 3
        assert bang["so_user"].sum() == len(SPLIT.users)
        assert bang["so_user"].min() >= len(SPLIT.users) // 3 - 1

    def test_nhan_ghi_KHOANG_THAT_va_xep_tang_dan(self):
        res = evaluate(Popularity().fit(SPLIT.train), SPLIT, by_activity=True)
        nhan = list(res["theo_nhom"].index)
        dau = [int(s.split("-")[0]) for s in nhan]
        assert dau == sorted(dau)
        counts = SPLIT.train.groupby("user_id").size()
        assert dau[0] == counts.min()
        assert int(nhan[-1].split("-")[1]) == counts.max()

    def test_dung_k_duoc_chon_khong_phai_ks_1(self):
        """Ban truoc lay cung `ks[1]` -- truyen ks=(10,) la vo."""
        res = evaluate(Popularity().fit(SPLIT.train), SPLIT, ks=(10,), keep_per_user=True)
        bang = _by_activity(res["per_user"].reset_index(), SPLIT, k=10)
        assert "ndcg@10" in bang.columns
