"""Chay web demo.

    .venv/Scripts/python -m recsys.web                       # 9 mo hinh, lan dau ~16 phut
    .venv/Scripts/python -m recsys.web --models b0,b1,b4,b5  # it mo hinh, lan dau ~1 phut
    .venv/Scripts/python -m recsys.web --rebuild             # bo cache, huan luyen lai
    .venv/Scripts/python -m recsys.web --no-posters          # khong tai anh tu TMDB

Lan dau huan luyen roi ghi cache vao data/cache/web/. Lan sau nap trong vai giay.
Mo http://127.0.0.1:5000 sau khi dong "Running on" hien ra.
"""

from __future__ import annotations

import argparse
import time

from recsys.data import movielens, tmdb
from recsys.data.split import leave_one_out
from recsys.web.app import create_app
from recsys.web.service import CACHE_DIR, SPECS, DemoService, fingerprint, load_or_fit


def main() -> None:
    keys_all = [s.key for s in SPECS]
    p = argparse.ArgumentParser(description="Web demo hệ gợi ý phim")
    p.add_argument("--dataset", default="ml-latest-small")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default=",".join(keys_all),
                   help=f"mã mô hình, cách nhau dấu phẩy, trong {','.join(keys_all)}")
    p.add_argument("--rebuild", action="store_true", help="bỏ cache, huấn luyện lại")
    p.add_argument("--no-posters", action="store_true", help="không tải ảnh bìa từ TMDB")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    bad = sorted(set(keys) - set(keys_all))
    if bad:
        p.error(f"không biết mô hình {bad}, chọn trong {keys_all}")

    t0 = time.perf_counter()
    ratings = movielens.load_ratings(args.dataset)
    split = leave_one_out(ratings, n_negatives=99, seed=args.seed)
    meta = tmdb.load_metadata(args.dataset)
    print(f"{args.dataset}: {split.summary()}")
    print(f"  phim có metadata TMDB: {int(meta['has_tmdb'].sum()) if 'has_tmdb' in meta else 0}"
          f"/{len(meta)}")

    titles = dict(zip(meta["item_id"].astype(int), meta["title"].astype(str)))
    # Cache TMDB doi thi B5, B7 doi. Ghi kich thuoc file vao dau van tay.
    tmdb_size = tmdb.CACHE_FILE.stat().st_size if tmdb.CACHE_FILE.exists() else 0
    fp = fingerprint(args.dataset, args.seed, extra=f"tmdb={tmdb_size}")

    print(f"Mô hình ({len(keys)}), cache ở {CACHE_DIR}:")
    specs = {s.key: s for s in SPECS}
    models = {}
    for key in keys:
        path = CACHE_DIR / f"{args.dataset}-s{args.seed}-{key}.pkl"
        models[key] = load_or_fit(specs[key], split.train, meta, titles, path, fp,
                                  rebuild=args.rebuild)
    print(f"Sẵn sàng sau {time.perf_counter() - t0:.0f} giây.")

    service = DemoService(split, meta, models, meta=meta, dataset=args.dataset)
    app = create_app(service, posters=not args.no_posters)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
