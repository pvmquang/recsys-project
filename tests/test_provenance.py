"""Chot phan ghi nguon goc ket qua (checklist 3.5)."""

import json

import pandas as pd

from recsys.eval.provenance import run_info, save_result


def test_csv_doc_lai_duoc_nhu_cu_va_co_file_meta(tmp_path):
    df = pd.DataFrame({"ndcg@10": [0.47]}, index=pd.Index(["B1-ItemKNN"], name="model"))
    path = save_result(df, tmp_path / "sub" / "kq.csv", seed=42, dataset="ml-latest-small")

    back = pd.read_csv(path, index_col="model")
    pd.testing.assert_frame_equal(back, df)

    meta = json.loads((tmp_path / "sub" / "kq.meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 42
    assert meta["dataset"] == "ml-latest-small"
    for key in ("git_commit", "git_dirty", "thoi_gian", "python", "thu_vien"):
        assert key in meta


def test_run_info_co_commit_khi_chay_trong_repo():
    info = run_info()
    # Chay trong repo git (ke ca tren CI) thi phai co hash 40 ky tu.
    assert info["git_commit"] is None or len(info["git_commit"]) == 40
    assert info["thu_vien"]["pandas"] == pd.__version__
