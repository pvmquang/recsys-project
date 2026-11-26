"""Phan xu ly cua web demo: huan luyen hoac nap mo hinh, tra goi y kem giai thich.

File nay KHONG import Flask. Test goi thang DemoService voi du lieu do choi, va
mot script khac (in goi y ra terminal chang han) dung lai duoc ma khong can web.

VI SAO HUAN LUYEN TREN split.train CHU KHONG PHAI TOAN BO DU LIEU
Demo dung dung tap train cua leave_one_out(seed=42), giong moi con so trong README.
Nho vay trang cua mot user hien duoc phim user do THUC SU xem tiep theo (phim test)
va hang cua phim do trong tung mo hinh. Nguoi xem thay mot dong cua bang ket qua,
khong phai mot danh sach goi y khong ai kiem duoc.

VI SAO CACHE BANG PICKLE
Huan luyen du 9 mo hinh mat khoang 16 phut tren ml-latest-small. Do bang chinh
load_or_fit (fit tren split.train, 25/11/2026):
    B0 0,1 s | B1 3,1 s | B2 0,1 s | B3 8,5 s | B4 34,7 s | B5 1,8 s
    B6 474,8 s | B7 48,1 s | B8 409,6 s
Nap lai 9 mo hinh tu pickle (~390 MB) va khoi dong web mat 8 giay. B6 khai pha
luat mat 218 giay trong docs/so-lieu-b6.md; lan do nay cham hon, chua ro vi sao
(may ban hon?). Cache gan voi dau van tay cua ma nguon mo hinh
(fingerprint): sua mot file trong recsys/models/ la cache cu bi bo va huan luyen lai.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.base import Recommender
from recsys.data.split import Split
from recsys.models.baselines import Popularity
from recsys.models.bpr import BPR
from recsys.models.content import ContentBased
from recsys.models.hybrid import NOI_DUNG, PHO_BIEN, TRON, Hybrid
from recsys.models.knn import K_ITEM, K_USER, ItemKNN, UserKNN
from recsys.models.lightgcn import LightGCN
from recsys.models.mf import MF
from recsys.models.rules import AssociationRules

_SRC = Path(__file__).resolve().parents[1]
CACHE_DIR = _SRC.parent / "data" / "cache" / "web"

#: Cach phuc vu NGUOI DUNG MOI (chua co trong train) cua tung mo hinh:
#:   refit    huan luyen lai tren train + vai phim nguoi do chon. Chi cho mo hinh
#:            fit duoi ~5 giay.
#:   fold_in  giu nguyen phan da hoc, chi nap lich su moi (AssociationRules.fold_in).
#:   cham     huan luyen lai duoc nhung mat gan mot phut; chi chay khi nguoi dung chon.
#:   khong    khong ho tro. MF/BPR/LightGCN can vector rieng cho nguoi moi, ma lay
#:            vector do phai huan luyen lai ca mo hinh.
REFIT, FOLD_IN, CHAM, KHONG = "refit", "fold_in", "cham", "khong"

Builder = Callable[[pd.DataFrame | None, Mapping[int, str]], Recommender]


@dataclass(frozen=True)
class ModelSpec:
    """Mot mo hinh tren giao dien: ma ngan, cach tao, cach phuc vu nguoi moi."""

    key: str
    label: str
    mo_ta: str
    build: Builder
    guest: str = KHONG


#: Sieu tham so lay tu hang so cua tung mo hinh (da chon tren validation).
#: B0-Random khong co o day: mot danh sach ngau nhien khong co gi de xem.
SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("b0", "Phổ biến", "Phim nhiều người xem nhất, giống nhau cho mọi người.",
              lambda meta, t: Popularity(), REFIT),
    ModelSpec("b1", "ItemKNN", "Phim giống những phim bạn đã xem, tính từ ma trận đánh giá.",
              lambda meta, t: ItemKNN(k=K_ITEM, titles=t), REFIT),
    ModelSpec("b2", "UserKNN", "Phim mà những người có gu giống bạn đã xem (Pearson).",
              lambda meta, t: UserKNN(k=K_USER), REFIT),
    ModelSpec("b3", "ALS", "Phân rã ma trận tương tác ẩn bằng bình phương tối thiểu luân phiên.",
              lambda meta, t: MF(titles=t), KHONG),
    ModelSpec("b4", "BPR-MF", "Phân rã ma trận, học thứ tự giữa phim đã xem và chưa xem.",
              lambda meta, t: BPR(titles=t), KHONG),
    ModelSpec("b5", "Nội dung", "Thể loại, đạo diễn, diễn viên, từ khóa từ TMDB.",
              lambda meta, t: ContentBased(meta, titles=t), REFIT),
    ModelSpec("b6", "Luật kết hợp", "Luật \"xem A thì xem B\" khai phá từ các phiên xem phim.",
              lambda meta, t: AssociationRules(titles=t), FOLD_IN),
    ModelSpec("b7", "Lai ghép", "BPR trộn nội dung; phim chưa ai xem dùng nội dung, "
              "người chưa có lịch sử dùng Phổ biến.",
              lambda meta, t: Hybrid(meta, titles=t), CHAM),
    ModelSpec("b8", "LightGCN", "Lan truyền embedding trên đồ thị người dùng và phim.",
              lambda meta, t: LightGCN(titles=t), KHONG),
)

#: Ten nhanh cua B7 (Hybrid.branch) -> nhan tren giao dien.
NHANH = {PHO_BIEN: "nhánh Phổ biến", NOI_DUNG: "nhánh nội dung", TRON: "trộn BPR + nội dung"}

_YEAR = re.compile(r"\s*\((\d{4})\)\s*$")
#: MovieLens ghi "Matrix, The" de sap xep theo chu cai. Tren giao dien dao lai.
_ARTICLE = re.compile(r"^(.*), (The|A|An)$")


def display_title(title: str) -> str:
    """'Matrix, The (1999)' -> 'The Matrix'. Bo nam, dao mao tu ve dau."""
    t = _YEAR.sub("", title)
    m = _ARTICLE.match(t)
    return f"{m.group(2)} {m.group(1)}" if m else t


# ---------------------------------------------------------------- cache --


def fingerprint(dataset: str, seed: int, extra: str = "") -> str:
    """Dau van tay cua (du lieu, seed, ma nguon mo hinh). Doi la cache cu bi bo.

    Bam ca thu muc recsys/models/ chu khong rieng file cua tung mo hinh: Hybrid
    dung BPR va ContentBased, LightGCN ke thua BPR. Bam rieng thi sua bpr.py ma
    B7, B8 van nap ban cu.
    """
    h = hashlib.sha256(f"{dataset}|{seed}|{extra}".encode())
    files = sorted((_SRC / "models").glob("*.py")) + [
        _SRC / "base.py",
        _SRC / "data" / "split.py",
        _SRC / "data" / "tmdb.py",
    ]
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def load_or_fit(
    spec: ModelSpec,
    train: pd.DataFrame,
    meta: pd.DataFrame | None,
    titles: Mapping[int, str],
    path: Path,
    fp: str,
    rebuild: bool = False,
    log: Callable[[str], None] = print,
) -> Recommender:
    """Nap mo hinh tu `path` neu dau van tay khop, nguoc lai huan luyen roi ghi lai."""
    if path.exists() and not rebuild:
        try:
            with path.open("rb") as fh:
                saved = pickle.load(fh)
            if saved.get("fp") == fp:
                log(f"  {spec.key} nạp từ cache")
                return saved["model"]
            log(f"  {spec.key} cache cũ (mã nguồn hoặc dữ liệu đã đổi), huấn luyện lại")
        except Exception as exc:  # file hong, hoac pickle tu phien ban thu vien khac
            log(f"  {spec.key} không đọc được cache ({exc}), huấn luyện lại")
    t0 = time.perf_counter()
    model = spec.build(meta, titles)
    model.fit(train)
    log(f"  {spec.key} {model.name} huấn luyện xong sau {time.perf_counter() - t0:.1f} giây")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"fp": fp, "model": model}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return model


# -------------------------------------------------------------- dich vu --


def _rank_of_target(model: Recommender, user_id: int, candidates: np.ndarray) -> int | None:
    """Hang (tu 1) cua item dung, xep DUNG nhu recsys/eval/runner.evaluate.

    Xao theo rng(user_id) roi argsort on dinh, de con so hien tren web la con so
    ma do do da thay. None khi item dung co diem -inf: mo hinh khong co y kien,
    hang cua no chi do thu tu xao quyet dinh.
    """
    scores = np.asarray(model.score(user_id, candidates), dtype=float)
    if not np.isfinite(scores[0]):
        return None
    order = np.random.default_rng(user_id).permutation(len(candidates))
    cand, sc = candidates[order], scores[order]
    ranked = cand[np.argsort(-sc, kind="stable")]
    return int(np.flatnonzero(ranked == candidates[0])[0]) + 1


class DemoService:
    """Moi thu trang web can, tren mot Split va cac mo hinh DA huan luyen.

    movies: bang phim, can item_id, title, genres (list hoac chuoi "A|B");
        poster_path tuy chon. Thuong la tmdb.load_metadata().
    models: ma -> mo hinh da fit tren split.train. Thu tu dict la thu tu tren web.
    meta: metadata truyen cho mo hinh noi dung khi phuc vu nguoi dung moi.
    """

    def __init__(
        self,
        split: Split,
        movies: pd.DataFrame,
        models: Mapping[str, Recommender],
        specs: Sequence[ModelSpec] = SPECS,
        meta: pd.DataFrame | None = None,
        dataset: str = "ml-latest-small",
    ):
        self.split = split
        self.models = dict(models)
        self.specs = {s.key: s for s in specs if s.key in self.models}
        self.meta = meta
        self.dataset = dataset
        self.train = split.train
        self.movies = self._movie_table(movies)
        self.titles = {i: m["title_full"] for i, m in self.movies.items()}
        self.n_views = self.train["item_id"].value_counts().to_dict()
        self.history_ = {
            int(u): g.sort_values("timestamp", ascending=False, kind="mergesort")
            for u, g in self.train.groupby("user_id")
        }
        self._guest_lock = threading.Lock()
        self._guest_cache: dict[tuple, list[dict]] = {}

    # ----------------------------------------------------------- phim --

    @staticmethod
    def _movie_table(movies: pd.DataFrame) -> dict[int, dict]:
        out: dict[int, dict] = {}
        has_poster = "poster_path" in movies.columns
        for row in movies.drop_duplicates("item_id").itertuples(index=False):
            full = str(row.title)
            m = _YEAR.search(full)
            genres = row.genres
            if isinstance(genres, str):
                genres = [] if genres == "(no genres listed)" else genres.split("|")
            out[int(row.item_id)] = {
                "item_id": int(row.item_id),
                "title_full": full,
                "title": display_title(full),
                "year": m.group(1) if m else "",
                "genres": list(genres or []),
                "poster": (getattr(row, "poster_path") or "") if has_poster else "",
            }
        return out

    def movie(self, item_id: int) -> dict:
        item_id = int(item_id)
        m = self.movies.get(item_id)
        if m is None:
            m = {"item_id": item_id, "title_full": str(item_id), "title": f"Phim {item_id}",
                 "year": "", "genres": [], "poster": ""}
        return {**m, "views": int(self.n_views.get(item_id, 0))}

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Phim co ten chua `query` (khong phan biet hoa thuong), nhieu nguoi xem truoc."""
        q = query.strip().lower()
        if not q:
            return []
        hits = [i for i, m in self.movies.items() if q in m["title_full"].lower()]
        hits.sort(key=lambda i: (-self.n_views.get(i, 0), i))
        return [self.movie(i) for i in hits[:limit]]

    # ------------------------------------------------------- nguoi dung --

    @property
    def users(self) -> list[int]:
        return sorted(int(u) for u in self.split.users)

    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self.split.test

    def sample_users(self) -> list[dict]:
        """5 user o phan vi 0, 25, 50, 75, 100% cua so tuong tac trong train."""
        counts = sorted((len(self.history_.get(u, ())), u) for u in self.users)
        if not counts:
            return []
        picks, seen = [], set()
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            n, u = counts[round(q * (len(counts) - 1))]
            if u not in seen:
                seen.add(u)
                picks.append({"user_id": u, "n": n})
        return picks

    def history(self, user_id: int) -> list[dict]:
        """Lich su trong train, moi nhat truoc."""
        g = self.history_.get(int(user_id))
        if g is None:
            return []
        out = []
        for item, rating, ts in zip(g["item_id"], g["rating"], g["timestamp"]):
            day = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%d/%m/%Y")
            out.append({**self.movie(item), "rating": float(rating), "date": day})
        return out

    def target(self, user_id: int) -> dict:
        """Phim user xem tiep theo, bi giau khoi train de kiem tra."""
        return self.movie(self.split.test[int(user_id)])

    # ---------------------------------------------------------- goi y --

    def _item_rows(self, model: Recommender, user_id: int, items: Sequence[int]) -> list[dict]:
        target = self.split.test.get(int(user_id))
        rows = []
        for rank, item in enumerate(items, 1):
            row = {
                **self.movie(item),
                "rank": rank,
                "explain": model.explain(user_id, item),
                "hit": target is not None and int(item) == int(target),
                "new": int(item) not in self.n_views,
                "branch": "",
            }
            if isinstance(model, Hybrid):
                row["branch"] = NHANH.get(model.branch(user_id, item), "")
            rows.append(row)
        return rows

    def recommend(self, key: str, user_id: int, k: int = 10) -> list[dict]:
        model = self.models[key]
        return self._item_rows(model, user_id, model.recommend(user_id, k=k))

    def target_rank(self, key: str, user_id: int) -> dict:
        """Hang cua phim test theo hai giao thuc trong README.

        mau: giua 100 ung vien (1 dung + 99 am lay mau), giao thuc cua bang chinh.
        toan: giua moi phim user chua xem.
        """
        model = self.models[key]
        u = int(user_id)
        full = self.split.full_candidates(u)
        return {
            "mau": _rank_of_target(model, u, self.split.candidates(u)),
            "n_mau": len(self.split.candidates(u)),
            "toan": _rank_of_target(model, u, full),
            "n_toan": len(full),
        }

    def compare(self, user_id: int, k: int = 10) -> list[dict]:
        """Top-k va hang phim test cua MOI mo hinh dang nap, cho trang so sanh."""
        out = []
        for key, model in self.models.items():
            out.append({
                "key": key,
                "spec": self.specs.get(key),
                "name": model.name,
                "recs": self.recommend(key, user_id, k),
                "rank": self.target_rank(key, user_id),
            })
        return out

    # --------------------------------------------------- nguoi dung moi --

    def guest_keys(self, heavy: bool = False) -> list[str]:
        ok = {REFIT, FOLD_IN} | ({CHAM} if heavy else set())
        return [k for k, s in self.specs.items() if s.guest in ok]

    @property
    def guest_id(self) -> int:
        """Ma cho nguoi dung moi: lon hon moi user that, khong bao gio trung."""
        return int(self.train["user_id"].max()) + 1

    def guest_rows(self, picks: Mapping[int, float]) -> pd.DataFrame:
        """Phim nguoi moi chon thanh cac dong tuong tac, cach nhau 60 giay.

        Cung mot phien (B6 cat phien o khoang nghi 1800 giay), va sau moi tuong
        tac cua train de lich su moi la lich su GAN NHAT.
        """
        t0 = int(self.train["timestamp"].max()) + 60
        items = [int(i) for i in picks]
        return pd.DataFrame({
            "user_id": self.guest_id,
            "item_id": items,
            "rating": [float(picks[i]) for i in items],
            "timestamp": [t0 + 60 * n for n in range(len(items))],
        })

    def guest(self, picks: Mapping[int, float], keys: Sequence[str], k: int = 10) -> list[dict]:
        """Goi y cho mot nguoi chua co trong train, tu cac phim ho chon.

        Mo hinh da huan luyen KHONG bi doi: refit tao mo hinh moi; fold_in chi doi
        ngu canh cua guest_id, va lan goi sau dat lai ngu canh do.
        """
        picks = {int(i): float(r) for i, r in picks.items() if int(i) in self.movies}
        if not picks:
            return []
        cache_key = (tuple(sorted(picks.items())), tuple(keys), k)
        with self._guest_lock:
            if cache_key in self._guest_cache:
                return self._guest_cache[cache_key]
            rows = self.guest_rows(picks)
            train = pd.concat([self.train, rows], ignore_index=True)
            gid = self.guest_id
            out = []
            for key in keys:
                spec = self.specs[key]
                t0 = time.perf_counter()
                if spec.guest == FOLD_IN:
                    model = self.models[key].fold_in(rows)
                elif spec.guest in (REFIT, CHAM):
                    model = spec.build(self.meta, self.titles).fit(train)
                else:
                    continue
                items = model.recommend(gid, k=k)
                out.append({
                    "key": key,
                    "spec": spec,
                    "name": model.name,
                    "recs": self._item_rows(model, gid, items),
                    "seconds": time.perf_counter() - t0,
                })
            if len(self._guest_cache) > 32:
                self._guest_cache.clear()
            self._guest_cache[cache_key] = out
            return out
