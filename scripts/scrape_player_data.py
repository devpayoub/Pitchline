"""
Scrape Premier League player data (who started, who scored, who is out).

    python scripts/scrape_player_data.py [--seasons 2025 2026] [--limit N]

Why player statistics and not the lineups feed: the `df_li_` feed emits one
global "Starting Lineups" header and then lists both squads, so a side can come
back with 22 players in one group. `get_player_statistics` carries an explicit
`is_starter` flag per player, which is the only reliable XI signal.

Per match it stores: the XI of each side, goals/assists/minutes per player, and
the missing-players list. Writes data/players/e0_players.json, resumable.
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

OUT = os.path.join(BASE, "data", "players", "e0_players.json")
REF = {"path": "/football/england/premier-league/"}
SEASONS = {
    2026: ("SY30SsKF", "CfoA8Dmm"),
    2025: ("KKay4EE8", "OEEq9Yvp"),
}
WANTED = {"GOALS", "ASSISTS", "MINUTES_PLAYED"}


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
        if len(rows) < 60:
            break
    return out


def stat_value(player, key):
    for s in player.get("statistics") or []:
        if s.get("id") == key:
            return s.get("numeric_value")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache = load_cache()
    added = 0

    for year in args.seasons:
        rows = season_results(year)
        print(f"season {year}/{str(year + 1)[-2:]}: {len(rows)} matches")
        for row in rows:
            mid = row.get("id")
            if not mid or mid in cache["matches"]:
                continue
            if args.limit and added >= args.limit:
                break
            home = (row.get("home") or {}).get("name")
            away = (row.get("away") or {}).get("name")
            try:
                ps = M.get_player_statistics(mid)
                mp = M.get_missing_players(mid)
            except Exception as exc:
                print(f"  ! {mid} {home}-{away}: {exc}")
                continue

            players = []
            for p in ps.get("players") or []:
                players.append({
                    "id": (p.get("player") or {}).get("id"),
                    "name": (p.get("player") or {}).get("name"),
                    "side": p.get("side"),
                    "starter": bool(p.get("is_starter")),
                    "goals": stat_value(p, "GOALS") or 0,
                    "assists": stat_value(p, "ASSISTS") or 0,
                    "minutes": stat_value(p, "MINUTES_PLAYED") or 0,
                })

            cache["matches"][mid] = {
                "date": row.get("date"),
                "home": home,
                "away": away,
                "home_score": row.get("home_score"),
                "away_score": row.get("away_score"),
                "players": players,
                "missing": {
                    "home": [{"name": (x.get("player") or {}).get("name"),
                              "reason": x.get("reason"),
                              "likelihood": x.get("likelihood")} for x in (mp.get("home") or [])],
                    "away": [{"name": (x.get("player") or {}).get("name"),
                              "reason": x.get("reason"),
                              "likelihood": x.get("likelihood")} for x in (mp.get("away") or [])],
                },
            }
            cache["seasons"][mid] = year
            added += 1

            if added % 10 == 0:
                save_cache(cache)
                print(f"  {added} new matches cached")
            time.sleep(0.2)

        save_cache(cache)
        if args.limit and added >= args.limit:
            break

    print(f"\nDone: {added} new, {len(cache['matches'])} cached -> {OUT}")


if __name__ == "__main__":
    main()
