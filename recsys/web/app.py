"""Cac trang cua web demo. Moi xu ly nam o service.DemoService, day chi la dinh tuyen.

Khong luu trang thai phia server: phim nguoi dung moi da chon nam tren URL
(?p=318:5&p=296:4). Tai lai trang, gui link cho nguoi khac deu ra cung ket qua.
"""

from __future__ import annotations

import random

from flask import Flask, abort, redirect, render_template, request, url_for

from recsys.web.service import DemoService

POSTER_URL = "https://image.tmdb.org/t/p/w185"


def parse_picks(values: list[str]) -> dict[int, float]:
    """['318:5', '296'] -> {318: 5.0, 296: 5.0}. Bo qua gia tri hong, rating ve [0.5, 5]."""
    picks: dict[int, float] = {}
    for v in values:
        item, _, rating = v.partition(":")
        try:
            i = int(item)
            r = float(rating) if rating else 5.0
        except ValueError:
            continue
        picks[i] = min(5.0, max(0.5, r))
    return picks


def create_app(service: DemoService, posters: bool = True) -> Flask:
    app = Flask(__name__)
    app.config["SERVICE"] = service

    @app.context_processor
    def _chung():
        def poster(m: dict) -> str:
            return f"{POSTER_URL}{m['poster']}" if posters and m.get("poster") else ""

        return {"svc": service, "poster": poster}

    def _model_key(default: str | None = None) -> str:
        key = request.args.get("model") or default
        if key is None or key not in service.models:
            key = "b7" if "b7" in service.models else next(iter(service.models))
        return key

    @app.get("/")
    def index():
        return render_template("index.html", samples=service.sample_users())

    @app.get("/random")
    def random_user():
        return redirect(url_for("user", uid=random.choice(service.users)))

    @app.get("/go")
    def go():
        uid = request.args.get("uid", type=int)
        if uid is None or not service.has_user(uid):
            return render_template("index.html", samples=service.sample_users(),
                                   loi=f"Không có người dùng {request.args.get('uid', '')} "
                                       "trong tập kiểm tra."), 404
        return redirect(url_for("user", uid=uid))

    @app.get("/user/<int:uid>")
    def user(uid: int):
        if not service.has_user(uid):
            abort(404)
        if request.args.get("model") and request.args["model"] not in service.models:
            abort(404)
        key = _model_key()
        return render_template(
            "user.html",
            uid=uid,
            key=key,
            model=service.models[key],
            history=service.history(uid),
            target=service.target(uid),
            recs=service.recommend(key, uid),
            rank=service.target_rank(key, uid),
        )

    @app.get("/user/<int:uid>/compare")
    def compare(uid: int):
        if not service.has_user(uid):
            abort(404)
        return render_template("compare.html", uid=uid, target=service.target(uid),
                               cols=service.compare(uid))

    @app.get("/guest")
    def guest():
        picks = parse_picks(request.args.getlist("p"))
        picks = {i: r for i, r in picks.items() if i in service.movies}
        heavy = request.args.get("nang") == "1"
        q = request.args.get("q", "")
        results = []
        if request.args.get("go") and picks:
            results = service.guest(picks, service.guest_keys(heavy))
        return render_template(
            "guest.html",
            q=q,
            found=service.search(q) if q else [],
            picks=[{**service.movie(i), "rating": r} for i, r in picks.items()],
            pick_args=[f"{i}:{r:g}" for i, r in picks.items()],
            heavy=heavy,
            results=results,
            keys_nhanh=service.guest_keys(False),
            keys_nang=[k for k in service.guest_keys(True) if k not in service.guest_keys(False)],
        )

    @app.errorhandler(404)
    def not_found(_):
        return render_template("404.html"), 404

    return app
