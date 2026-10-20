"""Nap metadata phim tu TMDB, co CACHE tren dia.

VI SAO PHAI CO CACHE
9.724 phim x 1 request ~ 40 phut goi mang. Goi lai moi lan huan luyen thi mot
buoi thu nghiem 20 lan chay mat 13 tieng cho rieng phan tai du lieu, va het
quota API. Cache la dieu kien de bac B5 lap thi nghiem duoc.

Cache ghi kieu JSONL (moi dong mot phim) nen NOI TIEP DUOC: dut mang o phim
thu 5.000 thi chay lai chi tai 4.724 phim con lai. Mot file JSON lon thi mat
trang neu dut giua chung.

CHAY DUOC KHI CHUA CO TMDB KEY
`load_metadata()` luon tra ve du lieu: movies.csv cua MovieLens da co san
TITLE va GENRES. Co cache TMDB thi bo sung them dien vien, dao dien, tu khoa,
tom tat. Nghia la bac B5 bat dau duoc ngay, khong phai cho dang ky khoa.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from recsys.data import movielens

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_FILE = CACHE_DIR / "tmdb_movies.jsonl"

API_BASE = "https://api.themoviedb.org/3"

#: So dien vien dau tien lay lam dac trung. Lay het thi vai phu va dien vien
#: quan chung lan at vai chinh, ma ho khong mang thong tin ve "phim nay giong
#: phim nao".
TOP_CAST = 5

NO_GENRE = "(no genres listed)"


class TmdbError(RuntimeError):
    pass


class TmdbClient:
    """Goi TMDB API, co thu lai va gian nhip."""

    def __init__(self, api_key: str | None = None, delay: float = 0.25) -> None:
        if api_key is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            api_key = os.environ.get("TMDB_API_KEY", "").strip()

        if not api_key:
            raise TmdbError(
                "Chua co TMDB_API_KEY.\n"
                "  1. Dang ky: https://www.themoviedb.org/settings/api\n"
                "  2. Chep .env.example thanh .env roi dien khoa vao.\n"
                "Chua co khoa van chay duoc bac B5 voi genres cua MovieLens -- "
                "xem load_metadata()."
            )
        self.api_key = api_key
        self.delay = delay
        self.session = requests.Session()

    def fetch_movie(self, tmdb_id: int, max_retries: int = 3) -> dict | None:
        """Lay chi tiet mot phim. Tra ve None neu TMDB khong co phim do.

        Dung `append_to_response` de lay chi tiet + credits + keywords trong
        MOT request thay vi ba. Goi ba lan la gap ba thoi gian va gap ba quota.
        """
        url = f"{API_BASE}/movie/{tmdb_id}"
        params = {"api_key": self.api_key, "append_to_response": "credits,keywords"}

        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=20)
            except requests.RequestException as exc:
                if attempt == max_retries - 1:
                    raise TmdbError(f"Loi mang khi lay phim {tmdb_id}: {exc}") from exc
                time.sleep(2**attempt)
                continue

            if resp.status_code == 404:
                # Mot so tmdb_id trong links.csv da cu, phim khong con tren TMDB.
                # Day la chuyen BINH THUONG, khong phai loi -- ghi nhan roi di tiep.
                return None
            if resp.status_code == 429:
                # Vuot nhip cho phep. TMDB noi phai cho bao lau trong header.
                wait = int(resp.headers.get("Retry-After", 5))
                time.sleep(wait + 1)
                continue
            if resp.status_code != 200:
                if attempt == max_retries - 1:
                    raise TmdbError(f"TMDB tra ve {resp.status_code} cho {tmdb_id}")
                time.sleep(2**attempt)
                continue

            time.sleep(self.delay)
            return resp.json()

        raise TmdbError(f"Khong lay duoc phim {tmdb_id} sau {max_retries} lan thu")


def _read_cached_ids() -> set[int]:
    """Doc cac tmdb_id da nam trong cache -- de chay tiep, khong tai lai."""
    if not CACHE_FILE.exists():
        return set()
    ids: set[int] = set()
    with CACHE_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(int(json.loads(line)["_tmdb_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                # Dong hong (dut giua chung lan truoc) -- bo qua, se tai lai.
                continue
    return ids


def build_cache(
    dataset: str = "ml-latest-small",
    limit: int | None = None,
    api_key: str | None = None,
) -> Path:
    """Tai metadata cho moi phim trong bo du lieu, ghi vao cache.

    Chay lai duoc bao nhieu lan cung duoc: phim da co trong cache thi bo qua.
    Dut mang giua chung thi chay lai, no di tiep tu cho do.

    limit: chi tai N phim dau -- de thu nhanh truoc khi chay ca 9.724 phim.
    """
    links = movielens.load_links(dataset).dropna(subset=["tmdb_id"])
    links["tmdb_id"] = links["tmdb_id"].astype(int)

    done = _read_cached_ids()
    todo = [t for t in links["tmdb_id"].tolist() if t not in done]
    if limit is not None:
        todo = todo[:limit]

    print(f"Cache: da co {len(done):,} phim | can tai them {len(todo):,} phim")
    if not todo:
        return CACHE_FILE

    client = TmdbClient(api_key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = n_missing = 0
    # Mo o che do append va flush sau MOI dong: dut giua chung van giu duoc
    # phan da tai. Ghi mot lan o cuoi la mat sach neu Ctrl+C.
    with CACHE_FILE.open("a", encoding="utf-8") as fh:
        for i, tmdb_id in enumerate(todo, start=1):
            payload = client.fetch_movie(tmdb_id)
            if payload is None:
                payload = {"_missing": True}
                n_missing += 1
            else:
                n_ok += 1
            payload["_tmdb_id"] = tmdb_id
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh.flush()

            if i % 200 == 0 or i == len(todo):
                print(f"  {i:,}/{len(todo):,}  (ok {n_ok:,} | khong co {n_missing:,})")

    print(f"Xong. Cache: {CACHE_FILE}")
    return CACHE_FILE


def _parse(payload: dict) -> dict:
    """Rut cac truong can dung tu mot ban ghi TMDB tho."""
    if payload.get("_missing"):
        return {}

    credits = payload.get("credits") or {}
    cast = [c.get("name", "") for c in (credits.get("cast") or [])[:TOP_CAST]]
    directors = [
        c.get("name", "")
        for c in (credits.get("crew") or [])
        if c.get("job") == "Director"
    ]
    kw_block = payload.get("keywords") or {}
    keywords = [k.get("name", "") for k in (kw_block.get("keywords") or [])]
    genres = [g.get("name", "") for g in (payload.get("genres") or [])]

    return {
        "tmdb_id": payload.get("_tmdb_id"),
        "tmdb_genres": genres,
        "cast": [c for c in cast if c],
        "directors": [d for d in directors if d],
        "keywords": [k for k in keywords if k],
        "overview": payload.get("overview") or "",
        "poster_path": payload.get("poster_path") or "",
        "vote_average": payload.get("vote_average"),
        "popularity": payload.get("popularity"),
    }


def load_cache() -> pd.DataFrame:
    """Doc cache thanh DataFrame. Cache rong thi tra ve DataFrame rong."""
    if not CACHE_FILE.exists():
        return pd.DataFrame()

    rows = []
    with CACHE_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = _parse(json.loads(line))
            except json.JSONDecodeError:
                continue
            if parsed:
                rows.append(parsed)

    return pd.DataFrame(rows)


def _empty_metadata(movies: pd.DataFrame) -> pd.DataFrame:
    """Bang metadata khi chua co cache TMDB -- chi co title va genres."""
    n = len(movies)
    movies = movies.copy()
    movies["cast"] = [[] for _ in range(n)]
    movies["directors"] = [[] for _ in range(n)]
    movies["keywords"] = [[] for _ in range(n)]
    movies["overview"] = ""
    movies["poster_path"] = ""
    movies["has_tmdb"] = False
    return movies


def load_metadata(dataset: str = "ml-latest-small") -> pd.DataFrame:
    """Bang metadata phim de bac B5 dung.

    LUON tra ve item_id / title / genres tu MovieLens.
    Neu cache TMDB co san thi bo sung cast / directors / keywords / overview.

    Cot `has_tmdb` noi ro tung phim co metadata TMDB hay khong. Khong co cot
    nay thi phim THIEU metadata trong y het phim co metadata RONG, va bac B5
    se bao "phim nay khong giong phim nao" cho mot thu that ra la thieu du lieu.
    """
    movies = movielens.load_movies(dataset)
    # MovieLens ghi the loai kieu "Action|Adventure|Sci-Fi".
    # NO_GENRE la gia tri THAT trong du lieu, khong phai loi doc file.
    movies["genres"] = movies["genres"].apply(
        lambda s: [] if s == NO_GENRE else str(s).split("|")
    )

    cache = load_cache()
    if cache.empty:
        return _empty_metadata(movies)

    links = movielens.load_links(dataset).dropna(subset=["tmdb_id"])
    links["tmdb_id"] = links["tmdb_id"].astype(int)

    df = movies.merge(links[["item_id", "tmdb_id"]], on="item_id", how="left")
    df = df.merge(cache, on="tmdb_id", how="left")

    df["has_tmdb"] = df["overview"].notna()
    for col in ("cast", "directors", "keywords"):
        df[col] = df[col].apply(lambda v: v if isinstance(v, list) else [])
    df["overview"] = df["overview"].fillna("")
    df["poster_path"] = df["poster_path"].fillna("")
    return df


def _token(name: str) -> str:
    """Bien mot TEN NGUOI thanh MOT token: 'James Cameron' -> 'james_cameron'.

    DAY LA MOT SUA LOI CO CHU Y so voi ma nguon tham khao cua do an.

    Repo goc dua thang ten vao CountVectorizer, nen 'James Cameron' bi tach
    thanh 'james' + 'cameron'. Hau qua: moi phim co dien vien ten Chris, John
    hay Lee deu tuong dong GIA voi nhau, va cai ten cang pho bien thi tuong
    dong gia cang manh. Noi ten lai lam mot token thi 'chris_evans' va
    'chris_pratt' la hai token khac han nhau.

    Bao cao nen do CA HAI cach -- day la mot cai tien nho, do duoc bang so, va
    giai thich duoc bang mot cau.
    """
    return "_".join(str(name).lower().split())


def content_soup(
    row: pd.Series,
    genre_weight: int = 2,
    director_weight: int = 2,
    use_overview: bool = False,
) -> str:
    """Ghep metadata mot phim thanh chuoi van ban de TF-IDF vector hoa.

    genre_weight / director_weight: lap lai token bao nhieu lan. Lap N lan la
    nhan tan suat (TF) cua token do len N. Repo goc de dien vien, dao dien va
    the loai TRONG SO NGANG NHAU, trong khi dao dien mang nhieu thong tin ve
    phong cach phim hon mot vai phu. Hai tham so nay de bao cao DO duoc anh
    huong cua viec danh trong so, thay vi chot cung mot lua chon.

    use_overview: them ca tom tat noi dung. Mac dinh TAT vi tom tat dai gap
    nhieu lan phan con lai, nen no se lan at dien vien/dao dien/the loai. Bat
    len de do thu la mot thi nghiem dang lam, nhung phai DO chu dung doan.
    """
    parts: list[str] = []

    for g in row.get("genres") or []:
        parts.extend([_token(g)] * genre_weight)

    for d in row.get("directors") or []:
        parts.extend([_token(d)] * director_weight)

    for c in row.get("cast") or []:
        parts.append(_token(c))

    for k in row.get("keywords") or []:
        parts.append(_token(k))

    if use_overview:
        parts.extend(str(row.get("overview") or "").lower().split())

    return " ".join(parts)


def build_soup_column(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Ap `content_soup` cho ca bang."""
    return df.apply(lambda r: content_soup(r, **kwargs), axis=1)


def coverage_report(df: pd.DataFrame) -> str:
    """Mot doan ngan noi cache phu duoc bao nhieu phim, thieu nhung gi.

    In no ra TRUOC khi huan luyen bac B5. Thieu metadata la nguyen nhan pho
    bien nhat lam nhanh noi dung ra ket qua kem, va no khong nem loi -- phim
    thieu du lieu chi lang le khong giong phim nao.
    """
    n = len(df)
    has = int(df["has_tmdb"].sum())
    no_genre = int(df["genres"].apply(len).eq(0).sum())
    no_cast = int(df["cast"].apply(len).eq(0).sum())
    no_dir = int(df["directors"].apply(len).eq(0).sum())
    return (
        f"{n:,} phim | co metadata TMDB: {has:,} ({has / n:.1%})\n"
        f"  khong co the loai : {no_genre:,}\n"
        f"  khong co dien vien: {no_cast:,}\n"
        f"  khong co dao dien : {no_dir:,}"
    )
