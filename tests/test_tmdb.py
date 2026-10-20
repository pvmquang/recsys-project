"""Chot phan doc va ghep metadata TMDB.

Khong bai nao goi mang. Phan goi API khong test tu dong duoc, nhung phan
PARSE va phan GHEP CHUOI thi test duoc -- va do moi la cho sai am tham:
goi API hong thi nem loi, con parse sai thi chi lam moi phim giong nhau hon
mot chut, khong dong nao bao.
"""

import pandas as pd
import pytest

from recsys.data import tmdb


# Mot ban ghi TMDB tho, rut gon tu phan hoi that cua /movie/{id}?append_to_response=credits,keywords
PAYLOAD = {
    "_tmdb_id": 27205,
    "genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}],
    "overview": "Cobb steals secrets from dreams.",
    "poster_path": "/abc.jpg",
    "vote_average": 8.4,
    "popularity": 123.4,
    "credits": {
        "cast": [
            {"name": "Leonardo DiCaprio"},
            {"name": "Joseph Gordon-Levitt"},
            {"name": "Elliot Page"},
            {"name": "Tom Hardy"},
            {"name": "Ken Watanabe"},
            {"name": "Dien Vien Thu Sau"},
        ],
        "crew": [
            {"name": "Christopher Nolan", "job": "Director"},
            {"name": "Hans Zimmer", "job": "Original Music Composer"},
        ],
    },
    "keywords": {"keywords": [{"name": "dream"}, {"name": "heist"}]},
}


class TestParse:
    def test_lay_dung_cac_truong(self):
        out = tmdb._parse(PAYLOAD)
        assert out["tmdb_id"] == 27205
        assert out["tmdb_genres"] == ["Action", "Science Fiction"]
        assert out["keywords"] == ["dream", "heist"]
        assert out["overview"].startswith("Cobb")

    def test_chi_lay_TOP_CAST_dien_vien_dau(self):
        # Payload co 6 dien vien, TOP_CAST = 5 => dien vien thu 6 bi bo.
        out = tmdb._parse(PAYLOAD)
        assert len(out["cast"]) == tmdb.TOP_CAST
        assert "Dien Vien Thu Sau" not in out["cast"]

    def test_chi_lay_nguoi_co_job_Director(self):
        # Hans Zimmer la nhac si, khong phai dao dien. Lay nham ca crew thi
        # moi phim Hollywood deu "cung dao dien" voi nhau qua cac vai ky thuat.
        out = tmdb._parse(PAYLOAD)
        assert out["directors"] == ["Christopher Nolan"]

    def test_phim_khong_co_tren_tmdb_tra_ve_rong(self):
        assert tmdb._parse({"_missing": True, "_tmdb_id": 999}) == {}

    def test_thieu_truong_khong_nem_loi(self):
        # TMDB khong phai luc nao cung tra du truong. Thieu thi phai ra list
        # rong, khong duoc nem -- nem la mot phim hong lam chet ca lan tai.
        out = tmdb._parse({"_tmdb_id": 1})
        assert out["cast"] == []
        assert out["directors"] == []
        assert out["keywords"] == []
        assert out["overview"] == ""


class TestToken:
    def test_ten_nguoi_thanh_mot_token(self):
        assert tmdb._token("James Cameron") == "james_cameron"
        assert tmdb._token("Science Fiction") == "science_fiction"

    def test_hai_nguoi_cung_ten_rieng_ra_hai_token_KHAC_nhau(self):
        # Day la loi cua ma nguon tham khao: tach ten ra thi ca hai deu sinh
        # token 'chris', tao tuong dong gia giua hai phim khong lien quan.
        a = tmdb._token("Chris Evans")
        b = tmdb._token("Chris Pratt")
        assert a != b
        assert "chris" not in (a.split() + b.split())

    def test_khoang_trang_thua_khong_tao_token_khac(self):
        assert tmdb._token("  Tom   Hardy ") == tmdb._token("Tom Hardy")


class TestContentSoup:
    row = pd.Series(
        {
            "genres": ["Action", "Sci-Fi"],
            "directors": ["Christopher Nolan"],
            "cast": ["Tom Hardy", "Cillian Murphy"],
            "keywords": ["dream"],
            "overview": "A very long overview text here",
        }
    )

    def test_trong_so_lap_lai_token(self):
        soup = tmdb.content_soup(self.row, genre_weight=3, director_weight=1)
        assert soup.split().count("action") == 3
        assert soup.split().count("christopher_nolan") == 1

    def test_trong_so_1_la_khong_lap(self):
        soup = tmdb.content_soup(self.row, genre_weight=1, director_weight=1)
        assert soup.split().count("action") == 1

    def test_mac_dinh_KHONG_lay_overview(self):
        # Tom tat dai gap nhieu lan phan con lai nen no se lan at dien vien,
        # dao dien va the loai. Bat len phai la mot quyet dinh co y thuc.
        soup = tmdb.content_soup(self.row)
        assert "overview" not in soup
        assert "very" not in soup

    def test_bat_overview_thi_co_trong_soup(self):
        soup = tmdb.content_soup(self.row, use_overview=True)
        assert "overview" in soup

    def test_phim_khong_co_metadata_ra_chuoi_rong(self):
        empty = pd.Series({"genres": [], "directors": [], "cast": [], "keywords": []})
        assert tmdb.content_soup(empty) == ""

    def test_thieu_han_cot_cung_khong_nem(self):
        # Series khong co cot 'keywords' -- .get() phai do duoc chuyen do.
        partial = pd.Series({"genres": ["Drama"]})
        assert tmdb.content_soup(partial) == "drama drama"


class TestLoadMetadata:
    def test_cache_rong_van_ra_bang_day_du_cot(self, monkeypatch):
        """Chua co TMDB key van phai dung duoc -- genres cua MovieLens la du
        de bat dau bac B5."""
        movies = pd.DataFrame(
            {
                "item_id": [1, 2],
                "title": ["A (1999)", "B (2001)"],
                "genres": ["Action|Sci-Fi", tmdb.NO_GENRE],
            }
        )
        monkeypatch.setattr(tmdb.movielens, "load_movies", lambda *a, **k: movies)
        monkeypatch.setattr(tmdb, "load_cache", lambda: pd.DataFrame())

        df = tmdb.load_metadata()
        assert list(df["genres"]) == [["Action", "Sci-Fi"], []]
        assert not df["has_tmdb"].any()
        for col in ("cast", "directors", "keywords", "overview", "poster_path"):
            assert col in df.columns

    def test_no_genres_listed_ra_list_rong_chu_khong_phai_mot_the_loai(self, monkeypatch):
        # '(no genres listed)' la gia tri THAT trong MovieLens. De nguyen thi
        # no thanh mot "the loai" ma hang tram phim cung chia se, va chung se
        # tuong dong gia voi nhau.
        movies = pd.DataFrame(
            {"item_id": [1], "title": ["X"], "genres": [tmdb.NO_GENRE]}
        )
        monkeypatch.setattr(tmdb.movielens, "load_movies", lambda *a, **k: movies)
        monkeypatch.setattr(tmdb, "load_cache", lambda: pd.DataFrame())

        df = tmdb.load_metadata()
        assert df.loc[0, "genres"] == []


def test_client_thieu_khoa_thi_bao_ro_cach_lay(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.setattr(tmdb, "load_cache", lambda: pd.DataFrame())
    with pytest.raises(tmdb.TmdbError) as exc:
        tmdb.TmdbClient(api_key="")
    msg = str(exc.value)
    assert "themoviedb.org" in msg  # co duong dan di dang ky
    assert ".env" in msg  # co noi chep file nao
