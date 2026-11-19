"""Bac B5 -- goi y theo NOI DUNG (content-based), ho so user bang TF-IDF.

Moi phim la mot vector TF-IDF tu metadata: the loai, thap nien, va neu co cache
TMDB thi them dao dien, dien vien, tu khoa (xem tmdb.content_soup). Ho so user la
tong co trong so cac vector phim user da xem. Diem = cosine(ho so, phim).

=========================================================================
KHAC LOC CONG TAC (B1-B4) O MOT DIEM DO DUOC: CHAM DUOC PHIM CHUA AI XEM
B1-B4 chi biet phim qua NGUOI XEM no. Phim khong co trong train thi khong co
vector, khong co lang gieng => -inf. B5 biet phim qua METADATA, nen phim moi
toanh van co diem neu co the loai. 23/610 phim test chua tung xuat hien trong
train (checklist 2.6) -- khong bac B0-B4 nao doan duoc chung. Day la can cu
cho lai ghep B7 ma B4 noi con thieu (docs/so-lieu-b4.md, muc cold-start).

Doi lai, B5 khong biet phim nao DUOC XEM NHIEU. Hai phim cung the loai, cung
dao dien thi bang diem, du mot phim co 300 luot xem va phim kia co 1.
=========================================================================

HO SO USER = ITEM-KNN THEO NOI DUNG VOI k = VO CUNG
Vi x_i da chuan hoa L2, diem truoc khi chia ||p_u|| la
    p_u . x_j = sum_{i da xem} w_i * cos(x_i, x_j)
tuc dung cong thuc "sum" cua ItemKNN (B1), chi khac do tuong dong lay tu
METADATA thay vi tu HANH VI, va khong cat k lang gieng. Chia ||p_u|| khong doi
thu tu trong mot user nen khong anh huong do do xep hang.

Dang thuc tren cung cho explain() CHINH XAC nhu B3: diem tach dung thanh dong gop
cua tung phim da xem (`contributions`), va thanh dong gop cua tung dac trung
(`shared_features`, tich tung thanh phan cua p_u va x_j).

TRONG SO w_i (sieu tham so, chon tren validation)
    weighting="binary"  w_i = 1: da xem la mot phieu, khop bai toan du doan
                        tuong tac (checklist 2.2).
    weighting="rating"  w_i = r_ui - trung binh cua u: phim cham thap keo ho so
                        RA XA (Rocchio co phan hoi am). Khop du doan so thich.
    half_life=h         nhan them 0.5 ** (tuoi / h), tuoi = so tuong tac xay ra
                        SAU phim do (phim gan nhat tuoi 0). None = khong giam.

KHONG DUNG tags.csv CUA MOVIELENS
Tag do chinh user viet, co timestamp, nhieu tag viet SAU tuong tac kiem tra. Dung
thang thi ho so phim chua thong tin tu tuong lai cua user. Muon dung phai loc
theo thoi diem cat cua tung user -- chua lam.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys.base import Recommender
from recsys.data.tmdb import _token, content_soup

#: Cac nhom dac trung co the bat. "year" la thap nien rut tu ten phim MovieLens
#: ("Alien (1979)" -> decade_1970), co san ca khi chua co TMDB.
ALL_FIELDS = ("genres", "year", "directors", "cast", "keywords", "overview")
_LIST_FIELDS = ("genres", "directors", "cast", "keywords")
_YEAR = re.compile(r"\((\d{4})\)\s*$")

#: Sieu tham so CHON TREN VALIDATION (checklist 2.1). Xem scripts/exp_content.py --
#: script chay lai phep chon va canh bao neu ket qua khac so ghi o day.
#: ml-latest-small, seed 42, cache TMDB phu 98,8% phim, ndcg@10 tren validation
#: (docs/so-lieu-b5.md):
#:   the loai + nam + dao dien + dien vien + tu khoa, binary, half_life 3,
#:   trong so the loai 1, dao dien 4                      -> 0.2571
#:   cung cau hinh, trong so the loai 2, dao dien 2         -> 0.2245
#:   chi the loai + nam, half_life 3                        -> 0.2035
#:   them tom tat (overview)                                -> 0.1796
#:   weighting="rating" tot nhat                            -> 0.1543
#:   tham chieu: BPR 0.4928, ItemKNN k=80 0.4768
#: Trong so the loai 1 nam o BIEN cua luoi (1, 2, 4) -- chua thu 0.
#: Chua co cache TMDB thi cac truong TMDB rong, mo hinh chi con the loai + nam.
FIELDS: tuple[str, ...] = ("genres", "year", "directors", "cast", "keywords")
WEIGHTING = "binary"
HALF_LIFE: float | None = 3.0
GENRE_WEIGHT = 1
DIRECTOR_WEIGHT = 4


def decade_token(title: str) -> str | None:
    """'Alien (1979)' -> 'decade_1970'. Ten khong co nam thi None."""
    m = _YEAR.search(str(title).strip())
    if m is None:
        return None
    return f"decade_{int(m.group(1)) // 10 * 10}"


class ContentBased(Recommender):
    """B5 -- goi y theo noi dung. Xem docstring cua module.

    metadata: bang phim, it nhat co item_id va genres (list). Thuong la output
        cua tmdb.load_metadata(). Truyen vao, mo hinh khong tu doc file.
    """

    name = "B5-Content"

    def __init__(
        self,
        metadata: pd.DataFrame,
        fields: Sequence[str] = FIELDS,
        weighting: str = WEIGHTING,
        half_life: float | None = HALF_LIFE,
        genre_weight: int = GENRE_WEIGHT,
        director_weight: int = DIRECTOR_WEIGHT,
        join_names: bool = True,
        titles: Mapping[int, str] | None = None,
    ):
        bad = set(fields) - set(ALL_FIELDS)
        if bad:
            raise ValueError(f"Khong biet nhom dac trung {sorted(bad)}. Chon trong {ALL_FIELDS}")
        if weighting not in ("binary", "rating"):
            raise ValueError("weighting phai la 'binary' hoac 'rating'")
        self.metadata = metadata
        self.fields = tuple(fields)
        self.weighting = weighting
        self.half_life = half_life
        self.genre_weight = genre_weight
        self.director_weight = director_weight
        #: False = tach ten nguoi thanh tung tu nhu ma nguon tham khao
        #: ('james_cameron' -> 'james cameron'). Chi de do chenh lech, xem tmdb._token.
        self.join_names = join_names
        #: item_id -> ten phim, chi cho explain(). Khong co thi dung cot title cua metadata.
        self.titles = titles

    # ------------------------------------------------------------ dac trung --

    def _soup(self, row: pd.Series) -> str:
        data = {f: (row.get(f) if f in self.fields else []) for f in _LIST_FIELDS}
        data["overview"] = row.get("overview", "")
        text = content_soup(
            data,
            genre_weight=self.genre_weight,
            director_weight=self.director_weight,
            use_overview="overview" in self.fields,
        )
        if "year" in self.fields:
            dec = decade_token(row.get("title", ""))
            if dec:
                text = f"{text} {dec}".strip()
        if not self.join_names:
            text = text.replace("_", " ")
        return text

    def _labels(self) -> dict[str, str]:
        """token -> nhan doc duoc cho explain(): 'james_cameron' -> 'James Cameron'."""
        out: dict[str, str] = {}
        for _, row in self.metadata.iterrows():
            for f in _LIST_FIELDS:
                if f in self.fields:
                    for name in row.get(f) or []:
                        out.setdefault(_token(name), str(name))
            if "year" in self.fields:
                dec = decade_token(row.get("title", ""))
                if dec:
                    out.setdefault(dec, f"thap nien {dec.split('_')[1]}")
        return out

    def _build_items(self) -> None:
        meta = self.metadata.drop_duplicates("item_id")
        soups = [self._soup(r) for _, r in meta.iterrows()]
        # token_pattern \S+: moi token cua content_soup da la mot don vi (ten noi
        # bang '_'). Mac dinh cua sklearn bo token 1 ky tu va cat tai dau '-'
        # ('sci-fi' -> 'sci', 'fi').
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, norm="l2")
        if not any(soups):
            raise ValueError("Khong phim nao co dac trung voi fields da chon")
        self.X_ = vec.fit_transform(soups).tocsr()
        self.vocab_ = np.asarray(vec.get_feature_names_out())
        self.meta_items_ = meta["item_id"].to_numpy()
        self.row_ = {int(i): r for r, i in enumerate(self.meta_items_)}
        #: Phim co it nhat mot dac trung. Phim khong co gi (vi du "(no genres
        #: listed)" ma khong co TMDB) luon -inf.
        self.has_feat_ = np.diff(self.X_.indptr) > 0

    # ------------------------------------------------------------------ fit --

    def _weights(self, train: pd.DataFrame) -> pd.Series:
        df = train.sort_values(["user_id", "timestamp"], kind="mergesort")
        if self.weighting == "binary":
            w = pd.Series(1.0, index=df.index)
        else:
            w = df["rating"] - df.groupby("user_id")["rating"].transform("mean")
            # User cham moi phim cung mot muc thi moi trong so bang 0 va ho so rong.
            # Lui ve binary cho user do thay vi tra -inf cho moi phim.
            phang = w.abs().groupby(df["user_id"]).transform("sum") == 0
            w = w.where(~phang, 1.0)
        if self.half_life is not None:
            tuoi = df.groupby("user_id").cumcount(ascending=False)
            w = w * 0.5 ** (tuoi / self.half_life)
        return w

    def _fit(self, train: pd.DataFrame) -> None:
        self._build_items()
        self.users_ = np.sort(train["user_id"].unique())
        self.uidx_ = {int(u): k for k, u in enumerate(self.users_)}

        rows_item = train["item_id"].map(self.row_)
        ok = rows_item.notna().to_numpy()
        w = self._weights(train).reindex(train.index).to_numpy()
        r = train["user_id"].map(self.uidx_).to_numpy()[ok]
        c = rows_item.to_numpy()[ok].astype(int)
        #: W[u, i] = trong so cua phim i trong ho so u.
        self.W_ = sparse.csr_matrix(
            (w[ok], (r, c)), shape=(len(self.users_), self.X_.shape[0])
        )
        P = (self.W_ @ self.X_).tocsr()
        norm = np.sqrt(np.asarray(P.multiply(P).sum(axis=1)).ravel())
        self.norm_ = norm
        inv = np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)
        self.P_ = sparse.csr_matrix(sparse.diags(inv) @ P)
        # Ma tran diem phim x user, dac: 9.742 x 610 x 8 byte ~ 48 MB. Tinh mot lan,
        # score() chi con tra bang.
        self.S_ = (self.X_ @ self.P_.T).toarray()
        # "Co chung it nhat mot dac trung voi ho so" -- tinh bang mau thua, khong
        # suy tu S_ > 0: voi weighting="rating" diem co the am hoac triet tieu ve 0.
        Xb = self.X_.copy()
        Xb.data[:] = 1.0
        Pb = self.P_.copy().tocsr()
        Pb.data = (Pb.data != 0).astype(float)
        self.overlap_ = (Xb @ Pb.T).toarray() > 0

    # ---------------------------------------------------------------- score --

    def score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Cosine(ho so user, phim). -inf khi: user la, phim khong co trong
        metadata, phim khong co dac trung, hoac khong chung dac trung nao voi ho so.

        Phim CHUA AI XEM trong train van co diem neu co metadata -- xem module.
        """
        out = np.full(len(item_ids), -np.inf)
        u = self.uidx_.get(int(user_id))
        if u is None:
            return out
        rows = np.array([self.row_.get(int(i), -1) for i in item_ids])
        known = rows >= 0
        idx = rows[known]
        ok = self.overlap_[idx, u]
        vals = np.where(ok, self.S_[idx, u], -np.inf)
        out[known] = vals
        return out

    # -------------------------------------------------------------- explain --

    def contributions(self, user_id: int, item_id: int) -> dict[int, float]:
        """Dong gop cua tung phim da xem vao diem cua `item_id`.

        w_i * cos(x_i, x_j) / ||p_u||. Tong cac gia tri BANG DUNG score() --
        phan ra chinh xac, khong phai hau kiem (xem docstring cua module).
        """
        u = self.uidx_.get(int(user_id))
        j = self.row_.get(int(item_id))
        if u is None or j is None or self.norm_[u] == 0:
            return {}
        w = self.W_.getrow(u)
        if w.nnz == 0:
            return {}
        cos = (self.X_[w.indices] @ self.X_[j].T).toarray().ravel()
        vals = w.data * cos / self.norm_[u]
        return {int(self.meta_items_[i]): float(v) for i, v in zip(w.indices, vals)}

    def shared_features(self, user_id: int, item_id: int) -> dict[str, float]:
        """Dong gop cua tung dac trung (token) vao diem: p_u[t] * x_j[t].

        Tong cac gia tri cung BANG DUNG score().
        """
        u = self.uidx_.get(int(user_id))
        j = self.row_.get(int(item_id))
        if u is None or j is None:
            return {}
        prod = self.P_.getrow(u).multiply(self.X_.getrow(j)).tocoo()
        return {str(self.vocab_[t]): float(v) for t, v in zip(prod.col, prod.data)}

    def explain(self, user_id: int, item_id: int) -> str:
        feats = self.shared_features(user_id, item_id)
        top_f = [t for t, v in sorted(feats.items(), key=lambda kv: -kv[1]) if v > 0][:3]
        if not top_f:
            return ""
        if not hasattr(self, "labels_"):
            self.labels_ = self._labels()
        ten_f = ", ".join(self.labels_.get(t, t) for t in top_f)
        contrib = self.contributions(user_id, item_id)
        top_i = [i for i, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0][:2]
        txt = f"Vi ban hay xem phim co: {ten_f}"
        if top_i:
            txt += f" (giong nhat: {', '.join(self._title(i) for i in top_i)})"
        return txt

    def _title(self, item_id: int) -> str:
        if self.titles is not None:
            return self.titles.get(item_id, str(item_id))
        if "title" in self.metadata.columns:
            hit = self.metadata.loc[self.metadata["item_id"] == item_id, "title"]
            if len(hit):
                return str(hit.iloc[0])
        return str(item_id)
