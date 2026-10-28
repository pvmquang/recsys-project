"""Chot khoang tin cay bootstrap.

Sai o day khong nem loi: no chi lam mot chenh lech nhieu trong nhu "co y nghia",
va cau ket luan sai do di thang vao slide.
"""

import numpy as np
import pandas as pd
import pytest

from recsys.eval.stats import bootstrap_ci, is_significant, paired_diff_ci


class TestBootstrapCI:
    def test_CI_bao_quanh_trung_binh(self):
        x = np.random.default_rng(0).normal(0.4, 0.1, size=500)
        mean, lo, hi = bootstrap_ci(x)
        assert lo < mean < hi
        assert mean == pytest.approx(x.mean())

    def test_do_rong_khop_sai_so_chuan(self):
        """CI 95% ~ trung binh +- 1.96 * SE. Lech xa la cong thuc sai."""
        x = np.random.default_rng(1).normal(0, 1, size=1000)
        _, lo, hi = bootstrap_ci(x, n_boot=4000)
        se = x.std(ddof=1) / np.sqrt(len(x))
        assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.1)

    def test_lap_lai_duoc_theo_seed(self):
        x = np.arange(100, dtype=float)
        assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3)

    def test_rong_thi_NEM(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]))


class TestPairedDiff:
    def test_ghep_theo_USER_ID_khong_theo_vi_tri(self):
        """Hai lan danh gia duyet user theo thu tu khac nhau thi ghep theo vi tri
        se tru diem cua hai nguoi khac nhau ma khong nem loi."""
        a = pd.Series([0.9, 0.1, 0.5], index=[1, 2, 3])
        b = a.iloc[::-1]  # cung gia tri, thu tu nguoc
        diff, lo, hi = paired_diff_ci(a, b)
        assert diff == 0 and lo == 0 and hi == 0

    def test_ghep_cap_HEP_hon_hai_CI_rieng(self):
        """User kho thi kho voi moi mo hinh: lay hieu theo cap triet tieu phan do.
        Day la ly do dung CI cap, khong so hai CI rieng."""
        rng = np.random.default_rng(0)
        do_kho = rng.normal(0, 1, size=600)  # bien thien giua user, rat lon
        idx = np.arange(600)
        a = pd.Series(do_kho + 0.05 + rng.normal(0, 0.1, 600), index=idx)
        b = pd.Series(do_kho + rng.normal(0, 0.1, 600), index=idx)

        _, lo, hi = paired_diff_ci(a, b)
        _, alo, ahi = bootstrap_ci(a)
        _, blo, bhi = bootstrap_ci(b)
        assert is_significant((0, lo, hi))  # cap: thay ro chenh 0.05
        assert alo < bhi and blo < ahi  # hai CI rieng chong nhau
        assert (hi - lo) < (ahi - alo) / 5

    def test_khac_tap_user_thi_NEM(self):
        a = pd.Series([1.0, 2.0], index=[1, 2])
        b = pd.Series([1.0, 2.0], index=[1, 3])
        with pytest.raises(ValueError, match="CUNG"):
            paired_diff_ci(a, b)


def test_is_significant():
    assert is_significant((0.05, 0.03, 0.07))
    assert is_significant((-0.05, -0.07, -0.03))
    assert not is_significant((0.01, -0.01, 0.03))
