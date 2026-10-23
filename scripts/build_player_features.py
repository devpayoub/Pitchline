"""
Turn the scraped player data into pre-match team features.

    python scripts/build_player_features.py

Five numbers per side, each computed only from matches that happened BEFORE the
one it is attached to:

  xi_share        mean "how often does this starter usually start" (0-1)
  rotation        starters who were not in the previous XI (0-11)
  xi_goal_share   share of the team's season goal involvement that is on the pitch
  missing_count   players listed as unavailable
  missing_key     unavailable players who usually start (start-share >= 0.5)

Output: data/players/e0_player_features.csv
"""
import os
import sys
import json
import csv
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from teams import normalize          # noqa: E402

CACHE = os.path.join(BASE, "data", "players", "e0_players.json")
OUT = os.path.join(BASE, "data", "players", "e0_player_features.csv")


def surname(name):
    """'Bentancur R.' and 'Bentancur Rodrigo' must land on the same key."""
    return normalize(str(name or "").split(" ")[0])


def main():
    with open(CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    matches = cache["matches"]

    def season_of(date_str):
        """A season runs Aug->May: 2026-03-01 belongs to 2025/26."""
        try:
            d = date_str.split("-")
            year, month = int(d[0]), int(d[1])
        except (AttributeError, IndexError, ValueError):
            return 0
        return year if month >= 7 else year - 1

    # chronological, and season-scoped so "usual starter" means this season.
    # The season is derived from the date: the scrape used to tag everything with
    # whichever --seasons flag was passed, which split one campaign in two.
    ordered = sorted(matches.items(), key=lambda kv: kv[1].get("date") or "")

    state = defaultdict(lambda: {
        "matches": 0, "starts": defaultdict(int), "goals": defaultdict(int),
        "assists": defaultdict(int), "team_goals": 0, "last_xi": set(),
    })
    rows = []

    for mid, m in ordered:
        if m.get("home_score") is None or m.get("away_score") is None:
            continue
        season = season_of(m.get("date"))
        feature = {"date": m["date"], "home": m["home"], "away": m["away"],
                   "home_score": m["home_score"], "away_score": m["away_score"],
                   "season": season}

        for side in ("home", "away"):
            team = m[side]
            st = state[(season, team)]
            xi = [p for p in m["players"] if p.get("starter") and p.get("side") == side]
            played = st["matches"]

            if played and xi:
                shares = [st["starts"][p["id"]] / played for p in xi]
                feature[f"xi_share_{side}"] = round(sum(shares) / len(shares), 3)
                feature[f"rotation_{side}"] = len({p["id"] for p in xi} - st["last_xi"]) if st["last_xi"] else 0
                involvement = sum(st["goals"][p["id"]] + st["assists"][p["id"]] for p in xi)
                feature[f"xi_goal_share_{side}"] = round(
                    involvement / st["team_goals"], 3) if st["team_goals"] else None
            else:
                feature[f"xi_share_{side}"] = None
                feature[f"rotation_{side}"] = None
                feature[f"xi_goal_share_{side}"] = None

            missing = m.get("missing", {}).get(side) or []
            feature[f"missing_count_{side}"] = len(missing)
            key_missing = 0
            for entry in missing:
                key = surname(entry.get("name"))
                if not played:
                    continue
                # find the player id behind that surname
                for pid, cnt in st["starts"].items():
                    if pid and surname(_player_name(m, pid)) == key and cnt / played >= 0.5:
                        key_missing += 1
                        break
            feature[f"missing_key_{side}"] = key_missing

            # ---- update the running state with THIS match ----
            st["matches"] += 1
            st["team_goals"] += m["home_score"] if side == "home" else m["away_score"]
            for p in m["players"]:
                if p.get("side") != side:
                    continue
                if p.get("starter"):
                    st["starts"][p["id"]] += 1
                st["goals"][p["id"]] += p.get("goals") or 0
                st["assists"][p["id"]] += p.get("assists") or 0
            st["last_xi"] = {p["id"] for p in xi}

        rows.append(feature)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["date", "home", "away", "home_score", "away_score", "season",
              "xi_share_home", "xi_share_away", "rotation_home", "rotation_away",
              "xi_goal_share_home", "xi_goal_share_away",
              "missing_count_home", "missing_count_away",
              "missing_key_home", "missing_key_away"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    print(f"{len(rows)} matches -> {OUT}")
    if rows:
        sample = rows[-1]
        print("last row:", {k: sample[k] for k in fields if k.startswith(("xi_", "rotation", "missing"))})


_NAME_INDEX = {}


def _player_name(match, pid):
    if pid in _NAME_INDEX:
        return _NAME_INDEX[pid]
    for p in match["players"]:
        if p.get("id") == pid:
            _NAME_INDEX[pid] = p.get("name")
            return p.get("name")
    return ""


if __name__ == "__main__":
    main()
