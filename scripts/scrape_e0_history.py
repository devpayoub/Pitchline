"""
Backfill the 2025-26 Premier League matches into the player cache.

    python scripts/scrape_e0_history.py [--season 2025] [--limit N]

`tournaments.get_results` silently ignores the season id and always answers with
the current season, so the only route to a past season is per team: each club's
results page pages back through its own history. Twenty clubs, a few pages each,
then dedupe by match id.
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "flashscore_api"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from flashscore import tournaments as T, teams as TM, matches as M      # noqa: E402
import scrape_player_data as SP                                         # noqa: E402

REF = {"path": "/football/england/premier-league/"}
WINDOW = {2025: ("2025-07-01", "2026-06-30"), 2024: ("2024-07-01", "2025-06-30")}


def team_refs():
    st = T.get_standings(REF)
    return [row["team"] for row in (st.get("standings") or []) if row.get("team")]


def collect_ids(season, pages=4):
    lo, hi = WINDOW[season]
    ids = {}
    for team in team_refs():
        ref = {"id": team["id"], "path": REF["path"]}
        for page in range(1, pages + 1):
            try:
                res = TM.get_results(ref, page=page)
            except Exception as exc:
                print(f"  {team['name']} page {page}: {exc}")
                break
            rows = [m for comp in (res.get("competitions") or [])
                    for m in (comp.get("matches") or [])]
            if not rows:
                break
            fresh = 0
            for m in rows:
                d = str(m.get("date") or "")
                if lo <= d <= hi and m.get("id") not in ids:
                    ids[m["id"]] = {"date": d,
                                    "home": (m.get("home") or {}).get("name"),
                                    "away": (m.get("away") or {}).get("name"),
                                    "home_score": m.get("home_score"),
                                    "away_score": m.get("away_score")}
                    fresh += 1
            if fresh == 0 and page > 1:
                break
        print(f"  {team['name']}: {len(ids)} unique so far")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"Collecting {args.season}-{str(args.season + 1)[-2:]} match ids via team pages...")
    ids = collect_ids(args.season)
    print(f"{len(ids)} matches found for the season")
    if not ids:
        return

    cache = SP.load_cache()
    added = 0
    for mid, meta in sorted(ids.items(), key=lambda kv: kv[1]["date"]):
        if mid in cache["matches"]:
            continue
        if args.limit and added >= args.limit:
            break
        try:
            ps = M.get_player_statistics(mid)
            mp = M.get_missing_players(mid)
        except Exception as exc:
            print(f"  ! {mid}: {exc}")
            continue

        players = [{
            "id": (p.get("player") or {}).get("id"),
            "name": (p.get("player") or {}).get("name"),
            "side": p.get("side"),
            "starter": bool(p.get("is_starter")),
            "goals": SP.stat_value(p, "GOALS") or 0,
            "assists": SP.stat_value(p, "ASSISTS") or 0,
            "minutes": SP.stat_value(p, "MINUTES_PLAYED") or 0,
        } for p in (ps.get("players") or [])]

        cache["matches"][mid] = {
            "date": meta["date"], "home": meta["home"], "away": meta["away"],
            "home_score": meta["home_score"], "away_score": meta["away_score"],
            "players": players,
            "missing": {
                "home": [{"name": (x.get("player") or {}).get("name"),
                          "reason": x.get("reason"), "likelihood": x.get("likelihood")}
                         for x in (mp.get("home") or [])],
                "away": [{"name": (x.get("player") or {}).get("name"),
                          "reason": x.get("reason"), "likelihood": x.get("likelihood")}
                         for x in (mp.get("away") or [])],
            },
        }
        cache["seasons"][mid] = args.season
        added += 1
        if added % 15 == 0:
            SP.save_cache(cache)
            print(f"  {added}/{len(ids)} scraped")
        time.sleep(0.15)

    SP.save_cache(cache)
    print(f"Done: {added} new, {len(cache['matches'])} cached total")


if __name__ == "__main__":
    main()
