"""Hop dong chung cho MOI mo hinh goi y trong do an.

Tam bac mo hinh B0-B8 deu cai dat dung lop nay. Khung danh gia
(recsys/eval/) chi biet den lop nay, khong biet ben trong mo hinh la gi.

=> Ba thanh vien lam song song tu tuan 3 ma khong giam chan nhau,
   va them mot mo hinh moi la do duoc ngay trong ngay.

DUNG DOI hop dong nay sau tuan 2. Moi lan doi la phai sua lai tat ca
cac bac da viet xong, va bac viet sau se khong khop bac viet truoc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Recommender(ABC):
    """Lop cha cua moi mo hinh goi y.

    Lop con PHAI cai dat hai ham:
        _fit(train)            -- hoc tu du lieu huan luyen
        score(user_id, items)  -- cham diem mot danh sach item cho mot user

    Lop con NEN cai dat them:
        explain(user_id, item) -- cau giai thich vi sao goi y item nay

    `recommend()` duoc suy ra tu `score()` nen khong can viet lai.
    """

    #: Ten hien trong bang ket qua danh gia. Lop con ghi de.
    name: str = "unnamed"

    def fit(self, train: pd.DataFrame) -> "Recommender":
        """Huan luyen mo hinh.

        `train` co bon cot: user_id, item_id, rating, timestamp.

        Ham nay KHONG duoc ghi de. No ghi nho hai thu moi mo hinh deu can
        (danh muc item, va item tung user da xem) roi goi `_fit()`.
        Ghi de no la mo hinh cua ban thieu `all_items_` / `seen_`, va loi
        chi no ra o tan luc danh gia.
        """
        self.all_items_ = np.sort(train["item_id"].unique())
        self.seen_ = train.groupby("user_id")["item_id"].apply(set).to_dict()
        self._fit(train)
        return self

    @abstractmethod
    def _fit(self, train: pd.DataFrame) -> None:
        """Phan hoc that su cua tung mo hinh."""

    @abstractmethod
    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Cham diem `item_ids` cho `user_id`. Diem cao = nen goi y truoc.

        Tra ve mang cung do dai `item_ids`. Thang diem khong quan trong
        va khong can so sanh duoc giua cac mo hinh -- moi phep danh gia
        deu chi dung THU TU.

        Item khong biet (khong co trong train) phai tra ve -inf chu khong
        phai 0: diem 0 co the cao hon diem thuc cua item khac, va item la
        se leo len dau danh sach ma khong co gi bao.
        """

    def recommend(
        self, user_id: int, k: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Tra ve top-k item cho `user_id`, da xep hang giam dan."""
        items = self.all_items_
        if exclude_seen:
            seen = self.seen_.get(user_id, set())
            if seen:
                items = items[~np.isin(items, list(seen))]
        if len(items) == 0:
            return []
        scores = self.score(user_id, items)
        k = min(k, len(items))
        # argpartition lay k phan tu lon nhat trong O(n), roi moi sap xep k cai do.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return items[top].tolist()

    def explain(self, user_id: int, item_id: int) -> str:
        """Cau giai thich vi sao `item_id` duoc goi y cho `user_id`.

        Mac dinh tra ve chuoi rong. Lop con nen ghi de -- day la chuan dau
        ra M2 cua de cuong, va du kien de viet cau nay deu da co san trong
        luc tinh diem.
        """
        return ""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
