"""
Fetch live Flashscore odds for 1X2, O/U 2.5 and BTTS, then de-vig them into
market probabilities.

Usage:
    python scripts/fetch_odds_v4.py [--window_days 8] [--out data/odds.json]

Output:
    data/odds.json   structured, consumed by export_results.py
"""
import os
import re
import sys
import json
import argparse
from datetime import datetime
from difflib import SequenceMatcher

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "flashscore_api"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from flashscore.fetch import get_feed, get_graphql          # noqa: E402
from flashscore import parsers as P                        # noqa: E402

from teams import TEAM_ALIASES, normalize, side_similarity      # noqa: E402

# Both sides of a fixture must clear this, otherwise a shared club name is enough
# to bind the wrong match (Galatasaray vs Stuttgart once matched Paderborn vs
# Stuttgart because only the away side agreed).
MIN_SIDE_SIM = 0.72
MIN_AVG_SIM = 0.78
SWAP_SIM = 0.85


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def fetch_all_matches(days_back: int = 1, days_ahead: int = 8) -> list:
    out = []
    for offset in range(-days_back, days_ahead):
        text = get_feed(f"f_1_{offset}_5_en_1", optional=True)
        if not text:
            continue
        for r in P.match_list(text, sport_id=1, leaderboard=False):
            if r.get("status") == "finished":
                continue
            out.append({
                "id": r.get("id"),
                "home": (r.get("home") or {}).get("name", ""),
                "away": (r.get("away") or {}).get("name", ""),
                "home_id": (r.get("home") or {}).get("id"),
                "away_id": (r.get("away") or {}).get("id"),
                "competition": (r.get("competition") or {}).get("name", ""),
            })
    return out


def find_match(home: str, away: str, matches: list):
    """Return (event, swapped) or (None, False).

    Requires BOTH sides to agree, so a shared club name alone cannot bind the
    wrong fixture. Falls back to the reversed fixture (Flashscore occasionally
    lists the same game the other way round) when that match is near-exact.
    """
    best, best_score = None, 0.0
    for m in matches:
        s1, s2 = side_similarity(home, m["home"]), side_similarity(away, m["away"])
        if min(s1, s2) < MIN_SIDE_SIM:
            continue
        score = (s1 + s2) / 2
        if score > best_score:
            best_score, best = score, dict(m, _s1=s1, _s2=s2, _swapped=False)

    if best is None or best_score < MIN_AVG_SIM:
        for m in matches:
            s1, s2 = side_similarity(home, m["away"]), side_similarity(away, m["home"])
            score = (s1 + s2) / 2
            if min(s1, s2) >= SWAP_SIM and score > best_score:
                best_score, best = score, dict(m, _s1=s1, _s2=s2, _swapped=True)

    if best is None:
        return None, False
    best["similarity"] = round(best_score, 3)
    return best, best.pop("_swapped")


# --------------------------------------------------------------------------- #
# Odds
# --------------------------------------------------------------------------- #
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def devig(odds: dict) -> dict:
    """Proportional de-vig: implied probabilities normalised to 1."""
    inv = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in inv.items()}


def best_price(book_odds: list) -> dict:
    """Highest available price per outcome across bookmakers."""
    best = {}
    for vals in book_odds:
        for k, v in vals.items():
            if v and v > best.get(k, 0):
                best[k] = v
    return best


def market_prob(book_odds: list) -> dict:
    """Average de-vigged probability across all bookmakers (more stable than one book).

    Re-normalised at the end: a bookmaker that lists only two of three outcomes
    would otherwise leave the three-way average summing to less than 1.
    """
    acc, n = {}, 0
    for vals in book_odds:
        probs = devig(vals)
        if len(probs) < 2:
            continue
        n += 1
        for k, v in probs.items():
            acc[k] = acc.get(k, 0.0) + v
    if not n:
        return {}
    avg = {k: v / n for k, v in acc.items()}
    total = sum(avg.values())
    return {k: v / total for k, v in avg.items()} if total > 0 else {}


def participant_sides(event_id: str) -> dict:
    """{event_participant_id: 'home'|'away'} — odds rows name their pick by an
    event-participant id, which is NOT the team id from the day feed."""
    payload = get_graphql("dsos2", {"eventId": event_id, "projectId": 2})
    holder = payload.get("findEventById") or {}
    sides = {}
    for p in holder.get("eventParticipants") or []:
        side = ((p.get("type") or {}).get("side") or "").lower()
        if p.get("id") and side in ("home", "away"):
            sides[p["id"]] = side
    return sides


def fetch_markets(event: dict) -> dict:
    payload = get_graphql(
        "oce",
        {"eventId": event["id"], "projectId": 2, "geoIpCode": "US", "geoIpSubdivisionCode": ""},
        odds=True,
    )
    holder = payload.get("findOddsByEventId") or {}
    books = {}
    for entry in holder.get("settings", {}).get("bookmakers", []):
        b = entry.get("bookmaker") or {}
        books[b.get("id")] = P.clean(b.get("name"))

    try:
        sides = participant_sides(event["id"])
    except Exception:
        sides = {}

    hda, btts = [], []
    ou = {"ou15": [], "ou25": [], "ou35": []}

    for row in holder.get("odds", []):
        if row.get("bettingScope") != "FULL_TIME":
            continue
        btype = row.get("bettingType")
        items = row.get("odds", []) or []

        if btype == "HOME_DRAW_AWAY" and len(items) == 3:
            vals = {}
            for it in items:
                price = _num(it.get("value"))
                if price is None:
                    continue
                pid = it.get("eventParticipantId")
                if pid is None:
                    vals["draw"] = price
                elif pid in sides:
                    vals[sides[pid]] = price
            if len(vals) == 3:
                hda.append(vals)

        elif btype == "OVER_UNDER":
            # One row per line: 1.5, 2.5 and 3.5 are the three the models predict.
            per_line = {}
            for it in items:
                h = it.get("handicap") or {}
                line = h.get("value") if isinstance(h, dict) else None
                key = {"1.5": "ou15", "2.5": "ou25", "3.5": "ou35"}.get(str(line))
                if not key:
                    continue
                sel = (it.get("selection") or "").upper()
                price = _num(it.get("value"))
                if sel in ("OVER", "UNDER") and price:
                    per_line.setdefault(key, {})[sel.lower()] = price
            for key, vals in per_line.items():
                if len(vals) == 2:
                    ou[key].append(vals)

        elif btype == "BOTH_TEAMS_TO_SCORE":
            vals = {}
            for i, it in enumerate(items):
                price = _num(it.get("value"))
                if not price:
                    continue
                sel = (it.get("selection") or "").upper()
                if sel in ("YES", "NO"):
                    vals[sel.lower()] = price
                elif len(items) == 2:
                    vals["yes" if i == 0 else "no"] = price
            if len(vals) == 2:
                btts.append(vals)

    markets = {}
    for name, rows in (("1x2", hda), ("ou15", ou["ou15"]), ("ou25", ou["ou25"]),
                       ("ou35", ou["ou35"]), ("btts", btts)):
        if not rows:
            continue
        markets[name] = {
            "prob": market_prob(rows),
            "price": best_price(rows),
            "books": len(rows),
        }
    return markets


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Fetch 1X2 / O-U 2.5 / BTTS odds from Flashscore")
    ap.add_argument("--window_days", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(BASE, "data", "odds.json"))
    ap.add_argument("--predictions", default=os.path.join(BASE, "data", "results_model.json"),
                    help="optional model predictions to restrict the scan to")
    args = ap.parse_args()

    print("Scanning Flashscore day feeds...")
    pool = fetch_all_matches(days_ahead=args.window_days)
    print(f"{len(pool)} upcoming matches on Flashscore\n")

    wanted = []
    if os.path.exists(args.predictions):
        with open(args.predictions, encoding="utf-8") as f:
            wanted = json.load(f).get("matches", [])
        print(f"Restricting to {len(wanted)} predicted fixtures")

    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "matches": {}}
    matched = with_odds = 0

    targets = wanted if wanted else [{"home": m["home"], "away": m["away"]} for m in pool]
    for pred in targets:
        home, away = pred["home"], pred["away"]
        event, swapped = find_match(home, away, pool)
        if not event:
            print(f"  x  {home} vs {away}: not on Flashscore")
            continue
        matched += 1
        try:
            markets = fetch_markets(event)
        except Exception as exc:                     # network hiccup: keep going
            print(f"  !  {home} vs {away}: {exc}")
            continue
        if not markets:
            print(f"  -  {home} vs {away}: no odds posted yet")
            continue
        if swapped and "1x2" in markets:
            mk = markets["1x2"]
            for field in ("prob", "price"):
                d = mk.get(field) or {}
                mk[field] = {"home": d.get("away"), "draw": d.get("draw"), "away": d.get("home")}
            print(f"  ~  {home} vs {away}: fixture listed reversed, 1X2 flipped")
        with_odds += 1
        out["matches"][f"{home} vs {away}"] = {
            "flashscore": {"home": event["home"], "away": event["away"],
                           "competition": event["competition"], "id": event["id"],
                           "swapped": swapped},
            "similarity": event.get("similarity"),
            "markets": markets,
        }
        bits = []
        if "1x2" in markets:
            p = markets["1x2"]["prob"]
            bits.append(f"1X2 {p.get('home', 0):.0%}/{p.get('draw', 0):.0%}/{p.get('away', 0):.0%}")
        for line in ("ou15", "ou25", "ou35"):
            if line in markets:
                bits.append(f"{line[2:]}: {markets[line]['prob'].get('over', 0):.0%}")
        if "btts" in markets:
            bits.append(f"BTTS {markets['btts']['prob'].get('yes', 0):.0%}")
        print(f"  +  {home} vs {away}: " + " | ".join(bits))

    out["summary"] = {"scanned": len(pool), "matched": matched, "with_odds": with_odds}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nMatched {matched}/{len(targets)} · odds for {with_odds}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
