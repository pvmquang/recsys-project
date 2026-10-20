"""Tai metadata TMDB ve cache.

    python scripts/fetch_tmdb.py --limit 20     # thu nhanh 20 phim truoc
    python scripts/fetch_tmdb.py                # tai het (~40 phut)
    python scripts/fetch_tmdb.py --report       # xem cache dang phu duoc bao nhieu

CHAY MOT LAN LA DU. Cache nam o data/cache/tmdb_movies.jsonl va khong bi xoa
khi chay lai. Dut giua chung (Ctrl+C, rot mang) thi chay lai lenh nay, no di
tiep tu cho do chu khong tai lai tu dau.

Can TMDB_API_KEY trong .env. Chua co khoa thi bac B5 van chay duoc bang
genres cua MovieLens -- xem recsys/data/tmdb.load_metadata().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.data import tmdb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-latest-small")
    p.add_argument("--limit", type=int, default=None, help="Chi tai N phim dau")
    p.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Giay cho giua hai request. Ha xuong la de bi TMDB chan (429).",
    )
    p.add_argument("--report", action="store_true", help="Chi xem cache, khong tai")
    args = p.parse_args()

    if args.report:
        df = tmdb.load_metadata(args.dataset)
        print(tmdb.coverage_report(df))
        return

    try:
        tmdb.build_cache(args.dataset, limit=args.limit)
    except tmdb.TmdbError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    df = tmdb.load_metadata(args.dataset)
    print()
    print(tmdb.coverage_report(df))


if __name__ == "__main__":
    main()
