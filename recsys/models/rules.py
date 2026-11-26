"""Bac B6 -- LUAT KET HOP theo PHIEN (Agrawal & Srikant, VLDB 1994; FP-growth: Han,
Pei & Yin, SIGMOD 2000). Khai pha bang mlxtend.

PHIEN LA GI TREN MOVIELENS
MovieLens chi co thoi diem CHAM DIEM, khong co thoi diem xem. Mot phien la chuoi
rating cua cung mot user, hai rating lien tiep cach nhau khong qua `session_gap`
giay. ml-latest-small, gap 30 phut: 7.091 phien trong train, 97% tuong tac nam
trong phien co tu 2 phim. Co phien dai 1.019 phim -- user cham hang loat luc moi
dang ky. Phien kieu do la "mot buoi cham diem", khong phai "mot buoi xem phim".

=========================================================================
KHAI PHA: moi phien la mot giao dich (tap phim). fpgrowth tim tap phim xuat hien
trong it nhat `min_support` phan giao dich, toi da `max_len` phim. Tu do sinh luat
X -> {j}: "phien co du cac phim X thi thuong co ca j".
    confidence = P(j | X)            -- uu tien phim pho bien
    lift       = P(j | X) / P(j)     -- uu tien phim it nguoi xem ma di kem X
CHAM DIEM: ngu canh cua user = phien cuoi cung trong train (context="session")
hoac n phim gan nhat (context=n). Diem cua j = max (hay tong) metric cua cac luat
X -> {j} ma X nam tron trong ngu canh. Khong luat nao khop thi -inf.
=========================================================================

QUAN HE VOI ItemKNN (B1)
Voi max_len=2, moi luat la {a} -> {j}, va confidence(a -> j) la ti le dong xuat
hien cua a va j tren so phien co a -- cung loai tin hieu voi tuong dong cua
ItemKNN. Hai khac biet: (1) dem trong PHIEN, khong tren ca lich su user, (2) chi
giu cap vuot nguong min_support, nen phim it phien khong co luat (ml-latest-small,
min_support 0,0005: 5.328/9.701 phim lam ve phai). B6 vi vay tra loi cau hoi "chi nhin
phien gan nhat co du khong", va ItemKNN la doi chung tu nhien.

GIAI THICH: luat la cau giai thich san co. explain() neu luat manh nhat cho phim
do, kem so phien lam bang chung.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.base import Recommender

#: Sieu tham so CHON TREN VALIDATION (checklist 2.1). Xem scripts/exp_rules.py --
#: script chay lai phep chon va canh bao neu ket qua khac so ghi o day.
#: ml-latest-small, seed 42, ndcg@10 tren validation:
#:   gap 1800 / ms 0.005 0.3169, 0.002 0.4138, 0.001 0.4666, 0.0005 0.4867
#:   gap 86400 / ms 0.005 0.3526, 0.002 0.4326, 0.001 0.4556, 0.0005 0.4669
#:   (ms giam thi tang deu; 0.0005 ~ "cap co trong >= 2 phien", la san co nghia)
#:   ngu canh (confidence, max): 1 phim 0.4315, 3 phim 0.4970, 10 phim 0.4995 (chon),
#:   phien cuoi 0.4867; lift tot nhat 0.3718 (10 phim, sum)
#:   max_len 3 o ms 0.005: 0.3081 so voi 0.3050, gap 9 lan so luat -> giu 2
#: Tham chieu: BPR 0.4928, ItemKNN 0.4768 cung giao thuc.
SESSION_GAP = 1800
MIN_SUPPORT = 0.0005
MAX_LEN = 2
CONTEXT: str | int = 10
METRIC = "confidence"
AGG = "max"


def session_ids(train: pd.DataFrame, gap: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Sap theo (user, timestamp) roi danh so phien 0, 1, 2...

    Phien moi bat dau o rating dau tien cua moi user, va moi khi hai rating lien
    tiep cach nhau HON `gap` giay. Tra ve (train da sap, ma phien tung dong).
    Sap on dinh: cung timestamp thi giu thu tu dau vao (split da pha hoa theo seed).
    """
    df = train.sort_values(["user_id", "timestamp"], kind="mergesort")
    cach = df.groupby("user_id")["timestamp"].diff()
    moi = cach.isna() | (cach > gap)
    return df, moi.to_numpy().cumsum() - 1


def _kiem_tra_cham_diem(context, metric: str, agg: str) -> None:
    if metric not in ("confidence", "lift"):
        raise ValueError("metric phai la 'confidence' hoac 'lift'")
    if agg not in ("max", "sum"):
        raise ValueError("agg phai la 'max' hoac 'sum'")
    if context != "session" and not (isinstance(context, int) and context >= 1):
        raise ValueError("context phai la 'session' hoac so nguyen >= 1")


class AssociationRules(Recommender):
    """B6 -- luat ket hop theo phien. Xem docstring cua module."""

    name = "B6-Rules"

    def __init__(
        self,
        session_gap: float = SESSION_GAP,
        min_support: float = MIN_SUPPORT,
        max_len: int = MAX_LEN,
        context: str | int = CONTEXT,
        metric: str = METRIC,
        agg: str = AGG,
        titles: Mapping[int, str] | None = None,
    ):
        _kiem_tra_cham_diem(context, metric, agg)
        if max_len < 2:
            raise ValueError("max_len >= 2: luat can it nhat mot phim moi ve")
        self.session_gap = session_gap
        self.min_support = min_support
        self.max_len = max_len
        self.context = context
        self.metric = metric
        self.agg = agg
        #: item_id -> ten phim, chi cho explain(). Truyen vao, khong tu doc file.
        self.titles = titles

    # ------------------------------------------------------------------ fit --

    def _fit(self, train: pd.DataFrame) -> None:
        df, sid = session_ids(train, self.session_gap)
        self.items_, col = np.unique(df["item_id"].to_numpy(), return_inverse=True)
        self.iidx_ = {int(v): k for k, v in enumerate(self.items_)}
        n_items = len(self.items_)

        X = sparse.csr_matrix(
            (np.ones(len(df), dtype=bool), (sid, col)), shape=(sid.max() + 1, n_items)
        )
        # Phien 1 phim khong sinh duoc luat nao, chi lam loang support.
        X = X[np.diff(X.indptr) >= 2]
        self.n_sessions_ = X.shape[0]
        self._mine(X, n_items)
        # Giu lai de rescore() dung lai ngu canh ma khong khai pha lai.
        self._lich_su = (df["user_id"].to_numpy(), sid, col)
        self._set_scoring()

    def rescore(self, context=None, metric=None, agg=None) -> "AssociationRules":
        """Doi cach CHAM DIEM tren luat da khai pha, khong khai pha lai.

        Cho ket qua DUNG nhu fit() voi tham so moi, vi context/metric/agg khong
        anh huong buoc khai pha. Script chon sieu tham so dua vao dieu nay: khai
        pha o min_support 0,002 mat vai phut, cham diem thi vai giay.
        """
        context = self.context if context is None else context
        metric = self.metric if metric is None else metric
        agg = self.agg if agg is None else agg
        _kiem_tra_cham_diem(context, metric, agg)
        self.context, self.metric, self.agg = context, metric, agg
        self._set_scoring()
        return self

    def fold_in(self, history: pd.DataFrame) -> "AssociationRules":
        """Thay lich su dung lam NGU CANH bang `history`, giu nguyen luat da khai pha.

        Cach phuc vu user moi ma khong khai pha lai: luat hoc offline tu phien cua
        moi nguoi, phien cua user moi chi la ngu canh. User co trong `history` duoc
        dat lai ngu canh va danh sach phim da xem; user khac giu nguyen. Phim khong
        co trong luc khai pha bi bo qua (khong co luat nao ve chung).

        scripts/exp_rules.py dung ham nay cho thi nghiem user moi: khai pha mot lan
        tren train da BO HAN cac user cua fold, roi fold_in lich su da cat cua ho.
        """
        df, sid = session_ids(history, self.session_gap)
        df = df.assign(_sid=sid)
        df = df[df["item_id"].isin(self.iidx_.keys())]
        users_cu, sid_cu, col_cu = self._lich_su
        giu = ~np.isin(users_cu, df["user_id"].unique())
        col_moi = df["item_id"].map(self.iidx_).to_numpy()
        # Ma phien moi tach han khoi ma phien cu de khong user nao bi gop phien.
        sid_moi = df["_sid"].to_numpy() + (sid_cu.max() + 1 if len(sid_cu) else 0)
        self._lich_su = (
            np.concatenate([users_cu[giu], df["user_id"].to_numpy()]),
            np.concatenate([sid_cu[giu], sid_moi]),
            np.concatenate([col_cu[giu], col_moi]),
        )
        for u, g in history.groupby("user_id")["item_id"]:
            self.seen_[int(u)] = set(g)
        self._set_scoring()
        return self

    def _set_scoring(self) -> None:
        users, sid, col = self._lich_su
        # Ngu canh: phien cuoi, hoac n phim gan nhat, cua tung user.
        self.context_: dict[int, np.ndarray] = {}
        for u, idx in pd.Series(np.arange(len(users))).groupby(users):
            idx = idx.to_numpy()
            if self.context == "session":
                idx = idx[sid[idx] == sid[idx[-1]]]
            else:
                idx = idx[-self.context :]
            self.context_[int(u)] = np.unique(col[idx])
        self.value_ = self.rules_[self.metric].to_numpy(dtype=float)
        self._cache: dict[int, np.ndarray] = {}

    def _mine(self, X: sparse.csr_matrix, n_items: int) -> None:
        from mlxtend.frequent_patterns import association_rules, fpgrowth

        self.ante_ = sparse.csr_matrix((0, n_items))
        self.ante_size_ = np.zeros(0, dtype=int)
        self.cons_ = np.zeros(0, dtype=int)
        self.rules_ = pd.DataFrame(columns=["antecedents", "consequents", "support",
                                            "confidence", "lift"])
        if self.n_sessions_ == 0:
            return
        # Cot dat ten 0..n-1: mlxtend doi cot so cua DataFrame thua phai bat dau tu 0.
        tx = pd.DataFrame.sparse.from_spmatrix(X, columns=np.arange(n_items))
        with warnings.catch_warnings():
            # pandas canh bao ve DataFrame thua ben trong mlxtend; khong lien quan ket qua.
            warnings.simplefilter("ignore")
            fi = fpgrowth(tx, min_support=self.min_support, max_len=self.max_len)
            if (fi["itemsets"].apply(len) >= 2).sum() == 0:
                return
            r = association_rules(
                fi, num_itemsets=self.n_sessions_, metric="confidence", min_threshold=0.0,
                return_metrics=["support", "confidence", "lift"],
            )
        # Chi giu luat ve phai MOT phim: diem cham cho tung phim.
        r = r[r["consequents"].apply(len) == 1].reset_index(drop=True)
        self.rules_ = r[["antecedents", "consequents", "support", "confidence", "lift"]]
        rows = np.repeat(np.arange(len(r)), r["antecedents"].apply(len).to_numpy())
        cols = np.fromiter((i for a in r["antecedents"] for i in a), dtype=int,
                           count=len(rows))
        #: ante_[r, i] = 1 neu phim i nam o ve trai luat r.
        self.ante_ = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(r), n_items)
        )
        self.ante_size_ = np.diff(self.ante_.indptr)
        self.cons_ = np.fromiter((next(iter(c)) for c in r["consequents"]), dtype=int,
                                 count=len(r))

    # ---------------------------------------------------------------- score --

    def _matched(self, u: int) -> np.ndarray:
        """Chi so cac luat co ve trai nam tron trong ngu canh cua user."""
        ctx = self.context_.get(u)
        if ctx is None or len(self.cons_) == 0:
            return np.zeros(0, dtype=int)
        c = np.zeros(self.ante_.shape[1])
        c[ctx] = 1.0
        hit = self.ante_ @ c
        ok = hit == self.ante_size_
        # Luat goi y lai phim ngay trong ngu canh khong co nghia; recommend() se loai
        # phim da xem, nhung score() tren ung vien thi khong, nen bo o day.
        ok &= c[self.cons_] == 0
        return np.flatnonzero(ok)

    def _user_scores(self, u: int) -> np.ndarray:
        if u in self._cache:
            return self._cache[u]
        out = np.full(self.ante_.shape[1], -np.inf)
        m = self._matched(u)
        if len(m):
            cons, val = self.cons_[m], self.value_[m]
            if self.agg == "max":
                np.maximum.at(out, cons, val)
            else:
                tong = np.bincount(cons, weights=val, minlength=len(out))
                out[np.unique(cons)] = tong[np.unique(cons)]
        self._cache[u] = out
        return out

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Metric cua luat manh nhat (hay tong cac luat) khop ngu canh, ve phai la
        phim do. -inf khi user la, phim la, hoac khong luat nao khop."""
        out = np.full(len(item_ids), -np.inf)
        s = self._user_scores(int(user_id))
        for p, it in enumerate(item_ids):
            j = self.iidx_.get(int(it))
            if j is not None:
                out[p] = s[j]
        return out

    # -------------------------------------------------------------- explain --

    def best_rule(self, user_id: int, item_id: int) -> dict | None:
        """Luat khop ngu canh co metric cao nhat cho `item_id`.

        Tra ve {"ve_trai": [item_id...], "confidence", "lift", "so_phien"} hoac None.
        """
        j = self.iidx_.get(int(item_id))
        if j is None:
            return None
        m = self._matched(int(user_id))
        m = m[self.cons_[m] == j]
        if len(m) == 0:
            return None
        r = m[np.argmax(self.value_[m])]
        row = self.rules_.iloc[r]
        return {
            "ve_trai": [int(self.items_[i]) for i in sorted(row["antecedents"])],
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "so_phien": int(round(row["support"] * self.n_sessions_)),
        }

    def explain(self, user_id: int, item_id: int) -> str:
        b = self.best_rule(user_id, item_id)
        if b is None:
            return ""
        ten = " va ".join(self._title(i) for i in b["ve_trai"])
        return (
            f"Nguoi cham {ten} trong cung mot phien thuong cham ca phim nay "
            f"(do tin cay {b['confidence']:.0%}, {b['so_phien']} phien)"
        )

    def _title(self, item_id: int) -> str:
        if self.titles is None:
            return str(item_id)
        return self.titles.get(item_id, str(item_id))
