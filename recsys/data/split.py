"""Chia du lieu: leave-one-out theo THOI GIAN + lay mau item am.

Giao thuc (chot o de xuat de tai, moi mo hinh B0-B8 deu do bang dung cai nay):

  1. Moi user: tuong tac CUOI CUNG theo timestamp -> tap kiem tra.
     Phan con lai -> tap huan luyen.
  2. Voi moi user, lay ngau nhien 99 item ho CHUA he tuong tac.
  3. Mo hinh xep hang 100 ung vien do. Item dung nam cang cao cang tot.

VI SAO KHONG CHIA NGAU NHIEN
Chia ngau nhien cho mo hinh nhin thay TUONG LAI cua chinh user do -- no hoc
tu phim user xem thang 12 roi duoc cham diem tren phim user xem thang 6.
Moi chi so se cao len mot cach gia tao, va cang cao o dung nhung mo hinh
manh nhat, nen bang so sanh cuoi cung se xep hang SAI.

Ma nguon tham khao cua do an da dinh dung lop loi nay o phan phan tich cam xuc
(mot binh luan lap 169 lan, chia ngau nhien truoc khi loai trung => cung mot
binh luan nam o ca train lan test => accuracy 99%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    """Ket qua chia du lieu."""

    train: pd.DataFrame
    #: user_id -> item_id duoc giu lai de kiem tra (dung MOT item moi user)
    test: dict[int, int]
    #: user_id -> 99 item am
    negatives: dict[int, np.ndarray]

    @property
    def users(self) -> list[int]:
        return list(self.test.keys())

    def candidates(self, user_id: int) -> np.ndarray:
        """100 ung vien cua mot user: 1 item dung + 99 item am.

        Item dung LUON o vi tri 0. Moi phep danh gia deu phai tu tim lai vi tri
        cua no sau khi xep hang, khong duoc gia dinh gi tu thu tu nay.
        """
        return np.concatenate([[self.test[user_id]], self.negatives[user_id]])

    def summary(self) -> str:
        return (
            f"train {len(self.train):,} tuong tac | "
            f"test {len(self.test):,} user | "
            f"{len(next(iter(self.negatives.values())))} item am moi user"
        )


def leave_one_out(
    ratings: pd.DataFrame,
    n_negatives: int = 99,
    min_interactions: int = 5,
    seed: int = 42,
) -> Split:
    """Chia du lieu theo giao thuc tren.

    min_interactions: user co it hon ngan nay tuong tac bi loai khoi TAP KIEM TRA
        (nhung tuong tac cua ho VAN nam trong tap huan luyen). Giu ho lai thi
        ket qua bi nhieu boi nhung user ma khong mo hinh nao doan noi.

    seed: BAT BUOC co. Ba thanh vien chay tren ba may phai ra cung mot lan chia,
        neu khong thi so cua ba nguoi khong so sanh duoc voi nhau, va khong co
        gi tren man hinh noi ra dieu do.
    """
    rng = np.random.default_rng(seed)

    # Sap xep on dinh: cung timestamp thi lay item_id lon hon lam item kiem tra.
    # Khong co tieu chi phu nay thi thu tu phu thuoc vao thu tu dong trong file,
    # va lan chia se doi khi doi phien ban pandas.
    df = ratings.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")

    counts = df.groupby("user_id").size()
    eligible = set(counts[counts >= min_interactions].index)

    all_items = np.sort(ratings["item_id"].unique())
    seen_by_user = df.groupby("user_id")["item_id"].apply(set).to_dict()

    # Tuong tac cuoi cung cua moi user hop le
    last_idx = df.groupby("user_id").tail(1)
    last_idx = last_idx[last_idx["user_id"].isin(eligible)]

    test = dict(zip(last_idx["user_id"], last_idx["item_id"]))
    train = df.drop(index=last_idx.index)

    negatives: dict[int, np.ndarray] = {}
    for user_id in test:
        seen = seen_by_user[user_id]
        pool = all_items[~np.isin(all_items, list(seen))]
        if len(pool) < n_negatives:
            raise ValueError(
                f"User {user_id} chi con {len(pool)} item chua xem, "
                f"khong du {n_negatives} item am."
            )
        negatives[user_id] = rng.choice(pool, size=n_negatives, replace=False)

    return Split(train=train, test=test, negatives=negatives)


def user_activity_groups(
    train: pd.DataFrame, bins: tuple[int, ...] = (0, 5, 20, 100)
) -> dict[int, str]:
    """Xep user vao nhom theo so tuong tac -- phuc vu phan tich cold-start.

    Tra ve user_id -> ten nhom ('<5', '5-20', '20-100', '>100').

    Day la mot trong hai phan tich bat buoc cua de tai: gia thuyet la nhanh noi
    dung thang o nhom thua, loc cong tac thang o nhom day, va lai ghep kieu
    chuyen dich bam duoc duong bao tren cua ca hai. Khong tach nhom thi ca ba
    ket luan do deu khong kiem chung duoc.

    Nguong CO DINH chi co nghia tren du lieu co user thua that, hoac sau khi cat
    bot lich su de tao cold-start. Tren ml-latest-small (moi user >= 20 rating),
    nhom '5-20' chi la 29 user co 19-20 tuong tac -- runner vi vay chia theo
    PHAN VI (xem `runner._by_activity`).
    """
    counts = train.groupby("user_id").size()
    labels = []
    for i, lo in enumerate(bins):
        hi = bins[i + 1] if i + 1 < len(bins) else None
        labels.append(f"<{bins[1]}" if i == 0 else (f"{lo}-{hi}" if hi else f">{lo}"))

    def label_of(n: int) -> str:
        for i in range(len(bins) - 1, 0, -1):
            if n > bins[i]:
                return labels[i]
        return labels[0]

    return {int(u): label_of(int(n)) for u, n in counts.items()}
