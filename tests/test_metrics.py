"""Chot bo do do bang so TINH TAY.

Vi sao phai co file nay: mot cong thuc do do sai khong bao gio nem loi. No
chi lam moi con so trong bao cao lech di mot cach nhat quan, nen bang ket qua
van "trong hop ly" va khong ai phat hien ra.

Moi ca duoi day deu co phep tinh tay ghi ngay trong comment. Doi mot ham
trong metrics.py ma test do la cong thuc da doi -- kiem lai bang tay truoc
khi sua test.
"""

import math

import pytest

from recsys.eval import metrics as m

# Danh sach xep hang mau: item dung la 30 (vi tri 2) va 50 (vi tri 4)
RANKED = [10, 30, 20, 50, 40]
RELEVANT = {30, 50}


def test_precision_at_k():
    # top-3 = [10, 30, 20], trung 1 => 1/3
    assert m.precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    # top-5 trung ca 2 => 2/5
    assert m.precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.4)


def test_recall_at_k():
    # top-3 trung 1 trong tong so 2 item dung => 0.5
    assert m.recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.5)
    assert m.recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(1.0)


def test_reciprocal_rank():
    # item dung dau tien (30) o vi tri 2 => 1/2
    assert m.reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)
    # khong trung gi => 0
    assert m.reciprocal_rank([1, 2, 3], RELEVANT) == 0.0


def test_average_precision():
    # P@2 = 1/2 (tai item 30), P@4 = 2/4 (tai item 50)
    # AP@5 = (1/2 + 1/2) / min(2, 5) = 0.5
    assert m.average_precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.5)
    # Cat o k=3: chi con hit tai vi tri 2 => (1/2) / min(2,3) = 0.25
    assert m.average_precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.25)


def test_dcg():
    # hit tai vi tri 2 va 4 => 1/log2(3) + 1/log2(5)
    expected = 1 / math.log2(3) + 1 / math.log2(5)
    assert m.dcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(expected)


def test_ndcg():
    # IDCG voi 2 item dung o dinh = 1/log2(2) + 1/log2(3) = 1 + 0.6309
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert m.ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)


def test_ndcg_hoan_hao_bang_1():
    # Hai item dung nam dung hai vi tri dau => nDCG phai bang chan 1.0
    assert m.ndcg_at_k([30, 50, 10, 20], RELEVANT, 4) == pytest.approx(1.0)


def test_auc():
    # 2 item dung, 3 item sai => 6 cap.
    # item 30 (vi tri 2) xep tren 20, 50, 40 -> nhung 50 la item dung, bo qua
    #   => 30 tren duoc 20 va 40 = 2 cap dung
    # item 50 (vi tri 4) xep tren 40 = 1 cap dung; thua 10 va 20
    # Tong cap dung = 2 + 1 = 3. Khong: dem lai theo cach cua ham --
    #   duyet: 10(sai, pos_so_far=0 -> +0), 30(dung -> pos=1),
    #          20(sai -> +1), 50(dung -> pos=2), 40(sai -> +2)
    #   => 0 + 1 + 2 = 3 cap dung tren 2*3 = 6 => 0.5
    assert m.auc(RANKED, RELEVANT) == pytest.approx(3 / 6)


def test_auc_xep_hang_hoan_hao():
    # Ca hai item dung deu tren moi item sai => AUC = 1.0
    assert m.auc([30, 50, 10, 20, 40], RELEVANT) == pytest.approx(1.0)


def test_auc_xep_hang_te_nhat():
    assert m.auc([10, 20, 40, 30, 50], RELEVANT) == pytest.approx(0.0)


def test_coverage():
    # 2 user, goi y cham vao 3 item khac nhau trong danh muc 10 item
    assert m.coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)


class TestTruongHopMotItemDung:
    """Giao thuc that cua do an: DUNG MOT item dung moi user.

    Ba bai duoi chot dieu ma docstring cua metrics.py canh bao -- Recall,
    HitRate va MAP/MRR trung nhau o thiet ke nay. Chung la TAI LIEU chay
    duoc, de nguoi viet bao cao khong bay ba cot giong het nhau nhu ba bang
    chung doc lap.
    """

    ranked = [10, 20, 30, 40, 50]
    relevant = {30}  # item dung o vi tri 3

    def test_recall_bang_hit_rate(self):
        for k in (1, 3, 5):
            assert m.recall_at_k(self.ranked, self.relevant, k) == pytest.approx(
                m.hit_rate_at_k(self.ranked, self.relevant, k)
            )

    def test_map_bang_mrr_khi_khong_cat(self):
        assert m.average_precision_at_k(
            self.ranked, self.relevant, 5
        ) == pytest.approx(m.reciprocal_rank(self.ranked, self.relevant))

    def test_gia_tri_cu_the(self):
        # item dung o vi tri 3 => MRR = 1/3, Recall@5 = 1, Precision@5 = 1/5
        assert m.reciprocal_rank(self.ranked, self.relevant) == pytest.approx(1 / 3)
        assert m.recall_at_k(self.ranked, self.relevant, 5) == pytest.approx(1.0)
        assert m.precision_at_k(self.ranked, self.relevant, 5) == pytest.approx(0.2)
        # nDCG@5 = (1/log2(4)) / (1/log2(2)) = 0.5
        assert m.ndcg_at_k(self.ranked, self.relevant, 5) == pytest.approx(0.5)
