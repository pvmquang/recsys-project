"""Chot web demo (recsys/web/). Bo qua neu chua cai Flask.

Du lieu do choi: user 1-4 xem phien sang (1, 2, 3) roi phien chieu gom 4, 5, 6 theo
thu tu xoay vong; user 5-8 xem (1, 2, 7) roi 8, 9 xen ke thu tu. Hai phien cach nhau
1 ngay. Phim test (phim cuoi) cua user nay van co trong train cua user khac, nen
Popularity cham duoc. Phim 10 khong ai xem.
Phim test: user 1 -> 6, user 2 -> 4, user 5 -> 9, user 6 -> 8.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("flask")

from recsys.data.split import leave_one_out  # noqa: E402
from recsys.eval.runner import evaluate  # noqa: E402
from recsys.models.baselines import Popularity  # noqa: E402
from recsys.models.knn import ItemKNN  # noqa: E402
from recsys.models.rules import AssociationRules  # noqa: E402
from recsys.web.app import create_app, parse_picks  # noqa: E402
from recsys.web.service import (  # noqa: E402
    FOLD_IN,
    KHONG,
    REFIT,
    DemoService,
    ModelSpec,
    display_title,
    fingerprint,
    load_or_fit,
)

NGAY = 86_400

MOVIES = pd.DataFrame(
    {
        "item_id": list(range(1, 11)),
        "title": ["Matrix, The (1999)"] + [f"Phim {i} ({1990 + i})" for i in range(2, 11)],
        "genres": [["Action"]] * 6 + [["Romance"]] * 3 + [["Action"]],
        "poster_path": [""] * 10,
    }
)


def lam_ratings() -> pd.DataFrame:
    rows = []
    for u in range(1, 9):
        if u <= 4:
            sang, chieu = [1, 2, 3], [[4, 5, 6], [5, 6, 4], [6, 4, 5], [4, 5, 6]][u - 1]
        else:
            sang, chieu = [1, 2, 7], ([8, 9] if u % 2 else [9, 8])
        rows += [(u, i, 4.0, t * 60) for t, i in enumerate(sang)]
        rows += [(u, i, 4.0, NGAY + t * 60) for t, i in enumerate(chieu)]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


SPECS = (
    ModelSpec("b0", "Pho bien", "", lambda meta, t: Popularity(), REFIT),
    ModelSpec("b1", "ItemKNN", "", lambda meta, t: ItemKNN(k=5, titles=t), REFIT),
    ModelSpec("b6", "Luat", "",
              lambda meta, t: AssociationRules(min_support=0.2, context="session", titles=t),
              FOLD_IN),
    ModelSpec("b9", "Khong ho tro", "", lambda meta, t: Popularity(), KHONG),
)


@pytest.fixture
def dich_vu() -> DemoService:
    split = leave_one_out(lam_ratings(), n_negatives=2, seed=42)
    titles = dict(zip(MOVIES["item_id"], MOVIES["title"]))
    models = {s.key: s.build(None, titles).fit(split.train) for s in SPECS}
    return DemoService(split, MOVIES, models, specs=SPECS, dataset="do-choi")


@pytest.fixture
def client(dich_vu):
    return create_app(dich_vu, posters=False).test_client()


class TestTienIch:
    def test_ten_phim_dao_mao_tu_va_bo_nam(self):
        assert display_title("Matrix, The (1999)") == "The Matrix"
        assert display_title("Phim 2 (1992)") == "Phim 2"
        assert display_title("Whatever, Inc. (2001)") == "Whatever, Inc."

    def test_doc_phim_da_chon_bo_gia_tri_hong_va_kep_diem(self):
        assert parse_picks(["1:4", "2", "x:5", "3:9", "4:abc"]) == {1: 4.0, 2: 5.0, 3: 5.0}

    def test_tim_phim_khong_phan_biet_hoa_thuong(self, dich_vu):
        assert [m["item_id"] for m in dich_vu.search("MATRIX")] == [1]
        assert dich_vu.search("   ") == []

    def test_tim_phim_xep_nhieu_luot_xem_truoc(self, dich_vu):
        ids = [m["item_id"] for m in dich_vu.search("phim")]
        luot = [dich_vu.n_views.get(i, 0) for i in ids]
        assert luot == sorted(luot, reverse=True)


class TestNguoiDungCo:
    def test_goi_y_khong_chua_phim_da_xem(self, dich_vu):
        for key in ("b0", "b1"):
            for u in dich_vu.users:
                da_xem = set(dich_vu.split.train.query("user_id == @u")["item_id"])
                assert not da_xem & {m["item_id"] for m in dich_vu.recommend(key, u)}

    def test_danh_dau_trung_phim_test(self, dich_vu):
        recs = dich_vu.recommend("b0", 1)  # user 1: phim test la 6, chua xem 6-9
        assert [m["item_id"] for m in recs if m["hit"]] == [6]
        assert all(m["hit"] == (m["item_id"] == 6) for m in recs)

    def test_hang_phim_test_khop_voi_khung_danh_gia(self, dich_vu):
        """1 / hang tren 100 ung vien phai bang MRR tung user cua runner.evaluate."""
        per_user = evaluate(dich_vu.models["b0"], dich_vu.split, keep_per_user=True)["per_user"]
        for u in dich_vu.users:
            hang = dich_vu.target_rank("b0", u)["mau"]
            assert hang is not None
            assert 1.0 / hang == pytest.approx(per_user.loc[u, "mrr"])

    def test_hang_rong_khi_phim_test_khong_co_diem(self, dich_vu):
        """Diem -inf: hang chi do thu tu xao quyet dinh, nen khong hien so."""
        class KhongBiet(Popularity):
            def score(self, user_id, item_ids):
                return np.full(len(item_ids), -np.inf)

        dich_vu.models["b9"] = KhongBiet().fit(dich_vu.split.train)
        assert dich_vu.target_rank("b9", 1) == {"mau": None, "n_mau": 3, "toan": None, "n_toan": 4}


class TestNguoiDungMoi:
    def test_chi_mo_hinh_ho_tro_moi_chay(self, dich_vu):
        assert dich_vu.guest_keys() == ["b0", "b1", "b6"]

    def test_khong_goi_y_lai_phim_da_chon(self, dich_vu):
        picks = {1: 5.0, 2: 4.0}
        out = dich_vu.guest(picks, dich_vu.guest_keys())
        assert [c["key"] for c in out] == ["b0", "b1", "b6"]
        for col in out:
            assert col["recs"], col["key"]
            assert not {m["item_id"] for m in col["recs"]} & set(picks)

    def test_luat_goi_y_phim_cung_phien(self, dich_vu):
        """1, 2 -> 3 (user 1-4) va 1, 2 -> 7 (user 5-8): ca hai phai len dau."""
        col = dich_vu.guest({1: 5.0, 2: 5.0}, ["b6"])[0]
        assert {m["item_id"] for m in col["recs"][:2]} == {3, 7}

    def test_khong_lam_doi_mo_hinh_da_huan_luyen(self, dich_vu):
        truoc = {k: dich_vu.recommend(k, 5) for k in ("b0", "b1", "b6")}
        knn = dich_vu.models["b1"]
        dich_vu.guest({4: 5.0, 8: 3.0}, dich_vu.guest_keys())
        assert dich_vu.models["b1"] is knn
        assert dich_vu.guest_id not in knn.seen_
        assert {k: dich_vu.recommend(k, 5) for k in truoc} == truoc

    def test_bo_phim_khong_co_trong_danh_muc(self, dich_vu):
        assert dich_vu.guest({999: 5.0}, ["b0"]) == []

    def test_dong_tuong_tac_cung_mot_phien_sau_train(self, dich_vu):
        rows = dich_vu.guest_rows({3: 5.0, 4: 2.0})
        assert rows["user_id"].nunique() == 1
        assert rows["user_id"].iloc[0] > dich_vu.split.train["user_id"].max()
        assert rows["timestamp"].min() > dich_vu.split.train["timestamp"].max()
        assert rows["timestamp"].diff().max() < 1800
        assert rows["rating"].tolist() == [5.0, 2.0]


class TestCache:
    def test_nap_lai_khi_dau_van_tay_khop_hoc_lai_khi_lech(self, tmp_path, dich_vu):
        goi = []

        def build(meta, titles):
            goi.append(1)
            return Popularity()

        spec = ModelSpec("b0", "x", "", build, REFIT)
        path = tmp_path / "b0.pkl"
        train = dich_vu.split.train
        log = lambda s: None  # noqa: E731
        m1 = load_or_fit(spec, train, None, {}, path, "fp1", log=log)
        m2 = load_or_fit(spec, train, None, {}, path, "fp1", log=log)
        assert len(goi) == 1
        assert m2.recommend(1) == m1.recommend(1)
        load_or_fit(spec, train, None, {}, path, "fp2", log=log)
        load_or_fit(spec, train, None, {}, path, "fp2", rebuild=True, log=log)
        assert len(goi) == 3

    def test_cache_hong_thi_hoc_lai(self, tmp_path, dich_vu):
        spec = ModelSpec("b0", "x", "", lambda meta, t: Popularity(), REFIT)
        path = tmp_path / "b0.pkl"
        path.write_bytes(b"khong phai pickle")
        m = load_or_fit(spec, dich_vu.split.train, None, {}, path, "fp", log=lambda s: None)
        assert m.recommend(1)

    def test_dau_van_tay_doi_theo_seed_va_du_lieu(self):
        assert fingerprint("a", 42) == fingerprint("a", 42)
        assert fingerprint("a", 42) != fingerprint("a", 7)
        assert fingerprint("a", 42) != fingerprint("b", 42)


class TestTrang:
    def test_trang_chu(self, client):
        r = client.get("/")
        assert r.status_code == 200
        html = r.get_data(as_text=True)
        assert "ItemKNN" in html and "/user/1" in html

    def test_trang_user_hien_phim_test_va_lich_su(self, client):
        html = client.get("/user/5?model=b1").get_data(as_text=True)
        assert "Phim 9" in html  # phim test
        assert "Phim 8" in html  # lich su
        assert "Phim xem tiếp theo" in html

    def test_user_hoac_mo_hinh_la_tra_404(self, client):
        assert client.get("/user/999").status_code == 404
        assert client.get("/user/1?model=zz").status_code == 404
        assert client.get("/user/999/compare").status_code == 404
        assert client.get("/go?uid=abc").status_code == 404

    def test_go_chuyen_toi_trang_user(self, client):
        r = client.get("/go?uid=3")
        assert r.status_code == 302 and r.headers["Location"].endswith("/user/3")

    def test_trang_so_sanh_co_moi_mo_hinh(self, client):
        html = client.get("/user/1/compare").get_data(as_text=True)
        for key in ("b0", "b1", "b6", "b9"):
            assert f"/user/1?model={key}" in html

    def test_trang_nguoi_moi_tim_va_goi_y(self, client):
        html = client.get("/guest?q=matrix").get_data(as_text=True)
        assert "The Matrix" in html and "p=1:5" in html.replace("%3A", ":")
        html = client.get("/guest?p=1:5&p=2:4&go=1").get_data(as_text=True)
        assert "B6-Rules" in html and "B1-ItemKNN" in html
