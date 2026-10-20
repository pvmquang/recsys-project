"""Tai va nap bo du lieu MovieLens.

Hai bo duoc dung trong do an:
  ml-latest-small  100.836 danh gia, 610 user, 9.742 phim  -- phat trien, lap thi nghiem
  ml-1m            1.000.209 danh gia, 6.040 user          -- chay ban cuoi, co nhan khau hoc

Moi thu tra ve deu dung BON TEN COT chuan: user_id, item_id, rating, timestamp.
MovieLens goi chung la userId/movieId, ml-1m thi dung dau :: -- quy ve mot moi
o day de khong mo hinh nao phai biet no dang doc bo nao.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DATASETS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}


def download(name: str = "ml-latest-small", force: bool = False) -> Path:
    """Tai va giai nen bo du lieu. Tra ve thu muc chua file."""
    if name not in DATASETS:
        raise ValueError(f"Khong biet bo '{name}'. Chon mot trong: {list(DATASETS)}")

    target = RAW_DIR / name
    if target.exists() and not force:
        return target

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dang tai {name} ...")
    resp = requests.get(DATASETS[name], timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(RAW_DIR)
    print(f"Xong: {target}")
    return target


def load_ratings(name: str = "ml-latest-small") -> pd.DataFrame:
    """Nap bang danh gia, chuan hoa ve user_id / item_id / rating / timestamp."""
    folder = download(name)

    if name == "ml-1m":
        df = pd.read_csv(
            folder / "ratings.dat",
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
        )
    else:
        df = pd.read_csv(folder / "ratings.csv").rename(
            columns={"userId": "user_id", "movieId": "item_id"}
        )

    return df[["user_id", "item_id", "rating", "timestamp"]]


def load_movies(name: str = "ml-latest-small") -> pd.DataFrame:
    """Nap bang phim: item_id, title, genres."""
    folder = download(name)

    if name == "ml-1m":
        df = pd.read_csv(
            folder / "movies.dat",
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1",
        )
    else:
        df = pd.read_csv(folder / "movies.csv").rename(columns={"movieId": "item_id"})

    return df[["item_id", "title", "genres"]]


def load_links(name: str = "ml-latest-small") -> pd.DataFrame:
    """Nap bang noi sang TMDB: item_id, imdb_id, tmdb_id.

    Day la duong noi MovieLens <-> TMDB cua ca do an. Khop bang KHOA CHINH
    `tmdb_id`, tuyet doi khong khop bang TEN PHIM: MovieLens ghi
    'Avatar (2009)' con TMDB ghi 'Avatar', va khop gan dung se roi rung mot
    so phim ma khong co dong nao bao.

    ml-1m KHONG co links.csv -- nhanh noi dung vi vay chi chay tren
    ml-latest-small, hoac phai tu khop sang ml-25m truoc.
    """
    folder = download(name)
    path = folder / "links.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Bo '{name}' khong co links.csv nen khong noi sang TMDB duoc. "
            "Dung 'ml-latest-small' cho nhanh noi dung."
        )
    return pd.read_csv(path).rename(
        columns={"movieId": "item_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"}
    )


def describe(df: pd.DataFrame) -> str:
    """Mot dong thong ke -- de in ra khi bat dau moi lan chay."""
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    density = len(df) / (n_users * n_items) * 100
    return (
        f"{len(df):,} danh gia | {n_users:,} user | {n_items:,} item "
        f"| do day {density:.2f}%"
    )
