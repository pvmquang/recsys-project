"""Bac B0 -- hai baseline. Muc dich khong phai de thang, ma de:

  1. Kiem tra chinh khung danh gia co dung khong. Random PHAI ra ndcg@10
     thap va gan mot gia tri tinh duoc truoc; ra cao la khung do sai.
  2. Lam moc cho moi bac sau. Mot mo hinh phuc tap ma khong thang noi
     Popularity la mot ket qua DANG BAO CAO, khong phai mot that bai can giau.

Popularity thuong manh hon nguoi ta tuong tren MovieLens. Neu B3 (MF) chi
hon Popularity vai phan tram thi do la phat hien that, hay viet no vao bao cao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.base import Recommender


class Popularity(Recommender):
    """Goi y phim duoc nhieu nguoi tuong tac nhat, giong nhau cho moi user.

    Khong ca nhan hoa mot chut nao -- day chinh la dieu lam no thanh baseline
    tot: moi bac sau phai chung minh viec ca nhan hoa co dang cong suc hay khong.
    """

    name = "B0-Popularity"

    def _fit(self, train: pd.DataFrame) -> None:
        counts = train.groupby("item_id").size()
        self.pop_ = counts.to_dict()
        self.top_ = counts.sort_values(ascending=False).index.tolist()

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        # Item khong co trong train -> -inf, khong phai 0 (xem Recommender.score).
        return np.array([self.pop_.get(int(i), -np.inf) for i in item_ids], dtype=float)

    def explain(self, user_id: int, item_id: int) -> str:
        n = self.pop_.get(int(item_id), 0)
        return f"Phim pho bien -- {n} nguoi da danh gia"


class RandomRec(Recommender):
    """Xep hang ngau nhien. San cua moi do do.

    Voi 1 item dung trong 100 ung vien, ky vong ly thuyet:
        recall@10 = 10/100 = 0.10
        auc       = 0.50
    Chay thuc te lech xa hai so nay la KHUNG DANH GIA dang sai, khong phai
    mo hinh sai. Day la cong dung chinh cua lop nay.
    """

    name = "B0-Random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _fit(self, train: pd.DataFrame) -> None:
        pass

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        # Gieo theo user_id de ket qua lap lai duoc giua cac lan chay.
        rng = np.random.default_rng(self.seed + int(user_id))
        return rng.random(len(item_ids))

    def explain(self, user_id: int, item_id: int) -> str:
        return "Chon ngau nhien (baseline)"
