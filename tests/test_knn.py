"""Chot B1 Item-KNN va B2 User-KNN.

Ma tran do choi dung de tinh duoc BANG TAY, va moi ca duoi day deu ghi phep
tinh ngay trong comment. Mot cong thuc tuong dong sai khong nem loi -- no chi
lam moi con so trong bao cao lech di mot cach nhat quan.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.models.knn import ItemKNN, UserKNN, build_matrix, topk_similarity

# u1, u2 cham 10/20 cao va 30 thap. u3, u4 thi nguoc lai.
# Chi u2 va u4 danh gia item 40 -- do la item de thu phep cham diem.
#
# DIEM PHAI BIEN THIEN TRONG TUNG USER. Fixture dau tien cho u1 cham 5.0 cho ca
# hai phim; phuong sai cua u1 bang 0 nen sau khi tru trung binh vector cua u1
# TOAN 0, va Pearson voi bat ky ai cung khong xac dinh => bon test UserKNN do vi
# FIXTURE, khong vi code. Xem `TestPhuongSaiKhong` o cuoi file.
TRAIN = pd.DataFrame(
    {
        "user_id": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        "item_id": [10, 20, 30, 10, 20, 30, 40, 10, 20, 30, 10, 20, 30, 40],
        "rating": [5.0, 4.0, 1.0, 5.0, 3.0, 1.0, 5.0,
                   1.0, 2.0, 5.0, 2.0, 1.0, 4.0, 1.0],
        "timestamp": range(14),
    }
)


class TestBuildMatrix:
    def test_hinh_dang_va_gia_tri(self):
        R, users, items, uidx, iidx = build_matrix(TRAIN)
        assert R.shape == (4, 4)
        assert R.nnz == 14
        assert list(users) == [1, 2, 3, 4]
        assert list(items) == [10, 20, 30, 40]
        assert R[uidx[1], iidx[10]] == 5.0

    def test_cap_user_item_TRUNG_thi_NEM(self):
        """csr_matrix CONG don o trung nhau, nen hai dong (1, 10) thanh rating
        9.0 -- khong nem, chi lam lech mo hinh. Phai bat tu day."""
        dup = pd.concat([TRAIN, TRAIN.head(1)])
        with pytest.raises(ValueError, match="trung nhau"):
            build_matrix(dup)


class TestTopkSimilarity:
    def test_khong_bao_gio_giu_chinh_no(self):
        M = sparse.csr_matrix(np.array([[1.0, 0], [1.0, 0], [0, 1.0]]))
        S = topk_similarity(M, k=3)
        assert S[0, 0] == 0.0
        assert S[1, 1] == 0.0

    def test_cosine_dung_gia_tri(self):
        # [1,0] va [1,1]: cosine = 1/(1*sqrt2) = 0.7071
        M = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 1.0]]))
        S = topk_similarity(M, k=2)
        assert S[0, 1] == pytest.approx(1 / np.sqrt(2))

    def test_cat_dung_k_lang_gieng(self):
        M = sparse.csr_matrix(np.ones((6, 3)))
        S = topk_similarity(M, k=2)
        assert all(S[i].nnz == 2 for i in range(6))

    def test_tuong_dong_AM_bi_bo(self):
        # [1,-1] va [-1,1] co cosine = -1. Lang gieng tuong quan NGHICH la
        # nhieu, khong phai bang chung -- min_sim=0 bo chung.
        M = sparse.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        S = topk_similarity(M, k=1)
        assert S.nnz == 0

    def test_significance_ha_trong_so_cap_it_item_chung(self):
        """Slide L2 tr.19: hai user chi cung danh gia 2 phim thi tuong quan
        1.0 cua ho khong dang tin bang hai user cung danh gia 50 phim."""
        M = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]))
        B = M.copy()
        B.data = np.ones_like(B.data)
        khong = topk_similarity(M, k=1)[0, 1]
        co = topk_similarity(M, k=1, significance=10, binary=B)[0, 1]
        # cung 2 item chung, gamma=10 => nhan 2/10
        assert co == pytest.approx(khong * 0.2)
        assert co < khong

    def test_significance_bat_ma_thieu_binary_thi_NEM(self):
        M = sparse.csr_matrix(np.ones((2, 2)))
        with pytest.raises(ValueError, match="binary"):
            topk_similarity(M, k=1, significance=5)


class TestItemKNN:
    def test_tuong_dong_tinh_tu_HANH_VI_khong_tu_metadata(self):
        """u1/u2 cham 10 va 20 cao roi cham 30 thap; u3/u4 nguoc lai. Nen 10 va
        20 phai tuong dong hon 10 va 30 -- may khong biet gi ve NOI DUNG phim,
        no chi thay CUNG MOT NHOM NGUOI doi xu voi chung giong nhau.

        Day la khac biet cot loi so voi bac B5 va so voi ma nguon tham khao."""
        m = ItemKNN(k=3).fit(TRAIN)
        i10, i20, i30 = m.iidx_[10], m.iidx_[20], m.iidx_[30]
        assert m.S_[i10, i20] > m.S_[i10, i30]

    def test_khong_bang_chung_thi_am_vo_cung_KHONG_phai_0(self):
        """Ca nay tung la BUG THAT: che do sum tra ve 0 cho item khong co lang
        gieng nao tung duoc user danh gia. 0 co the CAO HON diem that cua item
        khac (xem Recommender.score), va no lam "khong biet gi" doc y het
        "biet, va bang 0"."""
        rieng = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [10, 20, 30, 40],
                "rating": [5.0, 3.0, 5.0, 3.0],
                "timestamp": range(4),
            }
        )
        m = ItemKNN(k=3).fit(rieng)
        # Lang gieng cua 30 la 40; u1 chua danh gia ca hai => khong bang chung
        assert np.isneginf(m.score(1, np.array([30]))[0])

    def test_item_LA_tra_ve_am_vo_cung(self):
        m = ItemKNN(k=3).fit(TRAIN)
        assert np.isneginf(m.score(1, np.array([999]))[0])

    def test_user_LA_khong_nem_va_tra_am_vo_cung(self):
        m = ItemKNN(k=3).fit(TRAIN)
        s = m.score(999, np.array([10, 20]))
        assert len(s) == 2 and np.all(np.isneginf(s))

    def test_che_do_weighted_tra_ve_thang_diem_RATING(self):
        """weighted la cong thuc du doan rating, nen ket qua phai nam trong
        dai rating CUA DU LIEU (1..5), khong phai tong do tuong dong."""
        m = ItemKNN(k=3, score_mode="weighted").fit(TRAIN)
        s = m.score(1, np.array([40]))  # u1 chua xem 40
        assert 1.0 <= s[0] <= 5.0

    def test_hai_che_do_cham_diem_KHAC_nhau(self):
        a = ItemKNN(k=3, score_mode="sum").fit(TRAIN).score(1, np.array([40]))
        b = ItemKNN(k=3, score_mode="weighted").fit(TRAIN).score(1, np.array([40]))
        assert not np.isclose(a[0], b[0])

    def test_explain_dan_ten_phim_lam_cham_diem(self):
        m = ItemKNN(k=3).fit(TRAIN)
        assert "tuong tu" in m.explain(1, 40)

    def test_explain_in_TEN_phim_khi_co_titles(self):
        """Cau giai thich hien len giao dien va slide -- 'gan nhat: 10, 20' khong
        noi gi voi nguoi dung."""
        titles = {10: "Toy Story (1995)", 20: "Heat (1995)", 30: "Up (2009)"}
        txt = ItemKNN(k=3, titles=titles).fit(TRAIN).explain(1, 40)
        assert any(t in txt for t in titles.values())

    def test_explain_thieu_ten_thi_in_ma_phim_khong_nem(self):
        txt = ItemKNN(k=3, titles={}).fit(TRAIN).explain(1, 40)
        assert "gan nhat" in txt

    def test_adjusted_cosine_KHONG_bat_mac_dinh(self):
        assert ItemKNN().adjusted is False


class TestUserKNN:
    def test_lang_gieng_cua_u1_la_u2(self):
        """u1 va u2 danh gia cung hai phim => sau khi tru trung binh, hai vector
        cua ho cung huong. u3/u4 danh gia phim khac han."""
        m = UserKNN(k=3).fit(TRAIN)
        srow = m.S_[m.uidx_[1]]
        assert m.uidx_[2] in srow.indices

    def test_che_do_weighted_tra_ve_thang_diem_RATING(self):
        m = UserKNN(k=3, score_mode="weighted").fit(TRAIN)
        s = m.score(1, np.array([40]))  # lang gieng u2 da cham 40 = 5.0
        assert 1.0 <= s[0] <= 6.0

    def test_user_LA_tra_ve_am_vo_cung(self):
        m = UserKNN(k=3).fit(TRAIN)
        assert np.all(np.isneginf(m.score(999, np.array([10, 20]))))

    def test_lang_gieng_tuong_quan_NGHICH_bi_bo(self):
        """u3/u4 cham nguoc han u1 => Pearson am => khong phai lang gieng.
        Giu lang gieng am la lay "nguoi gu trai nguoc" lam bang chung."""
        m = UserKNN(k=3).fit(TRAIN)
        srow = m.S_[m.uidx_[1]]
        assert m.uidx_[3] not in srow.indices
        assert m.uidx_[4] not in srow.indices

    def test_explain_dem_so_nguoi(self):
        m = UserKNN(k=3).fit(TRAIN)
        assert "nguoi" in m.explain(1, 40)


class TestHopDong:
    """Hai bac moi phai thoa dung hop dong `Recommender` nhu B0."""

    @pytest.mark.parametrize("model", [ItemKNN(k=3), UserKNN(k=3)])
    def test_fit_ghi_nho_danh_muc_va_item_da_xem(self, model):
        model.fit(TRAIN)
        assert set(model.all_items_) == {10, 20, 30, 40}
        assert model.seen_[1] == {10, 20, 30}

    @pytest.mark.parametrize("model", [ItemKNN(k=3), UserKNN(k=3)])
    def test_recommend_loai_item_da_xem(self, model):
        model.fit(TRAIN)
        assert set(model.recommend(1, k=4)) == {40}  # u1 da xem 10, 20, 30

    @pytest.mark.parametrize("model", [ItemKNN(k=3), UserKNN(k=3)])
    def test_recommend_user_LA_khong_nem_va_tra_ve_RONG(self, model):
        """KNN khong co bang chung nao cho user chua tung thay => moi diem -inf.

        Truoc day bai nay doi du 2 phim -- tuc la doi hai phim chon tuy y tu
        cac diem bang nhau, trong y het goi y that. Rong moi la cau tra loi
        dung: B7 dua vao do de chuyen sang Popularity hay noi dung.
        """
        model.fit(TRAIN)
        assert model.recommend(999, k=2) == []

    def test_recommend_KHONG_tra_item_khong_bang_chung(self):
        """Item -inf khong duoc lap vao cho du k, du k lon hon so item co diem."""
        m = UserKNN(k=3).fit(TestPhuongSaiKhong.PHANG)
        # u1 khong co lang gieng nao => khong item nao co bang chung
        assert m.recommend(1, k=5, exclude_seen=False) == []

    @pytest.mark.parametrize("model", [ItemKNN(k=3), UserKNN(k=3)])
    def test_score_dung_do_dai(self, model):
        model.fit(TRAIN)
        items = np.array([10, 20, 30, 40, 999])
        assert len(model.score(1, items)) == 5

    @pytest.mark.parametrize("model", [ItemKNN(k=3), UserKNN(k=3)])
    def test_explain_luon_tra_ve_chuoi(self, model):
        model.fit(TRAIN)
        assert isinstance(model.explain(1, 20), str)

    def test_score_mode_la_thi_NEM(self):
        with pytest.raises(ValueError, match="score_mode"):
            ItemKNN(score_mode="trung_binh")


class TestPhuongSaiKhong:
    """User cham MOI phim cung mot diem thi Pearson khong xac dinh voi bat ky ai.

    Tinh chat cua Pearson, khong phai loi cai dat -- va la mot ly do THAT de bac
    B7 phai co nhanh lai ghep kieu chuyen dich: nhom user nay khong bao gio nhan
    duoc goi y tu B2.

    Fixture dau tien cua file nay vo tinh cho MOI user phuong sai 0, nen bon test
    UserKNN do cung luc va rat de doc thanh "code sai".
    """

    PHANG = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 20, 10, 20],
            "rating": [5.0, 5.0, 4.0, 4.0],  # moi user mot muc diem duy nhat
            "timestamp": range(4),
        }
    )

    def test_UserKNN_khong_tim_duoc_lang_gieng_nao(self):
        m = UserKNN(k=3).fit(self.PHANG)
        assert m.S_.nnz == 0
        assert np.all(np.isneginf(m.score(1, np.array([20]))))

    def test_ItemKNN_VAN_chay_duoc_tren_cung_du_lieu(self):
        """ItemKNN dung cosine tren rating THO (khong tru trung binh) nen no
        khong dinh. Hai bac phan ung khac nhau tren CUNG mot du lieu -- day la
        phat hien dang ghi vao bao cao, khong phai mot su co."""
        m = ItemKNN(k=3).fit(self.PHANG)
        assert m.S_.nnz > 0
