"""
Scrape Premier League lineups + missing players, so player availability can be
turned into model features.

    python scripts/scrape_lineups.py [--seasons 2025 2026] [--limit N]

Writes data/players/e0_lineups.json and is resumable: re-running skips matches
already in the cache. One match costs ~1.8 s (two requests), so a full season is
about 12 minutes.
"""
import os
import sys
import json
import time
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "flashscore_api"))

from flashscore import tournaments as T          # noqa: E402
from flashscore import matches as M              # noqa: E402

OUT = os.path.join(BASE, "data", "players", "e0_lineups.json")
REF = {"path": "/football/england/premier-league/"}
# tournament_id:stage_id, from tournaments.get_seasons()
SEASONS = {
    2026: ("SY30SsKF", "CfoA8Dmm"),
    2025: ("KKay4EE8", "OEEq9Yvp"),
}


def load_cache():
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    return {"matches": {}, "seasons": {}}


def save_cache(cache):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, OUT)


def season_results(year, pages=8):
    tid, sid = SEASONS[year]
    ref = dict(REF, tournament_id=tid, stage_id=sid)
    out = []
    for page in range(1, pages + 1):
        try:
            res = T.get_results(ref, page=page)
        except Exception as exc:
            print(f"  page {page} failed: {exc}")
            break
        rows = res.get("matches") or []
        if not rows:
            break
        out.extend(rows)
        if len(rows) < 60:      # last page
            break
    return out


def starters(side):
    """Starting XI only — the bench lives in a second group."""
    if not side:
        return []
    for group in side.get("groups") or []:
        if (group.get("name") or "").lower().startswith("starting"):
            return [{"id": p["player"]["id"], "name": p["player"]["name"]}
                    for p in group.get("players") or [] if p.get("player")]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--limit", type=int, default=0, help="stop after N new matches (0 = all)")
    args = ap.parse_args()

    cache = load_cache()
    added = 0

    for year in args.seasons:
        rows = season_results(year)
        print(f"season {year}/{str(year + 1)[-2:]}: {len(rows)} matches on the board")
        for row in rows:
            mid = row.get("id")
            if not mid or mid in cache["matches"]:
                continue
            if args.limit and added >= args.limit:
                break
            home = (row.get("home") or {}).get("name")
            away = (row.get("away") or {}).get("name")
            try:
                lu = M.get_lineups(mid)
                mp = M.get_missing_players(mid)
            except Exception as exc:
                print(f"  ! {mid} {home}-{away}: {exc}")
                continue

            entry = {
                "date": row.get("date"),
                "home": home,
                "away": away,
                "home_score": row.get("home_score"),
                "away_score": row.get("away_score"),
                "home_xi": starters(lu.get("home")),
                "away_xi": starters(lu.get("away")),
                "home_missing": [
                    {"name": (p.get("player") or {}).get("name"),
                     "reason": p.get("reason"),
                     "likelihood": p.get("likelihood")}
                    for p in (mp.get("home") or [])
                ],
                "away_missing": [
                    {"name": (p.get("player") or {}).get("name"),
                     "reason": p.get("reason"),
                     "likelihood": p.get("likelihood")}
                    for p in (mp.get("away") or [])
                ],
            }
            cache["matches"][mid] = entry
            cache["seasons"][mid] = year
            added += 1

            if added % 10 == 0:
                save_cache(cache)
                print(f"  {added} new matches cached ({year})")
            time.sleep(0.25)

        save_cache(cache)
        if args.limit and added >= args.limit:
            break

    print(f"\nDone: {added} new matches, {len(cache['matches'])} cached total -> {OUT}")


if __name__ == "__main__":
    main()
