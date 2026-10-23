"""
Live grading: read the newest archived predictions and score them against
matches that are still being played.

    python scripts/live_predictions.py

A goals market does not have to wait for the final whistle. Once the total goes
past the line the Over is won and the Under is lost, so the tick appears while
the match is still running. 1X2 and "no" sides of BTTS/Under only settle at full
time, and stay pending until then.

Output: data/history/live.json
"""
import os
import sys
import json
import argparse
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(BASE, "data", "history")
sys.path.insert(0, os.path.join(BASE, "flashscore_api"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from grade_predictions import parse_date                      # noqa: E402
from teams import side_similarity as similar                  # noqa: E402

LINES = {"ou15": 1.5, "ou25": 2.5, "ou35": 3.5}


def newest_archive():
    index_path = os.path.join(HIST, "index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    if not index:
        return None
    path = os.path.join(HIST, index[0]["file"])
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_feed_matches(days=1):
    """Today's and yesterday's fixtures, with live scores where available."""
    from flashscore.fetch import get_feed
    from flashscore import parsers as P

    out = []
    for offset in range(-days, 1):
        text = get_feed(f"f_1_{offset}_5_en_1", optional=True)
        if not text:
            continue
        for r in P.match_list(text, sport_id=1, leaderboard=False):
            status = r.get("status")
            if status not in ("live", "finished", "scheduled"):
                continue
            hg, ag = r.get("home_score"), r.get("away_score")
            if status == "scheduled":
                hg, ag = 0, 0          # not kicked off yet: no score, no ticks
            elif hg is None or ag is None:
                continue
            out.append({
                "id": r.get("id"),
                "home": (r.get("home") or {}).get("name", ""),
                "away": (r.get("away") or {}).get("name", ""),
                "date": str(r.get("date") or "")[:10],
                "status": status,
                "stage": r.get("stage") or "",
                "hg": int(hg), "ag": int(ag),
                "competition": (r.get("competition") or {}).get("name", ""),
            })
    return out


def find_live(match, pool):
    """Same club-pair rule as the grader, but the match may still be running."""
    want = parse_date(match.get("date"))
    best, best_score = None, 0.0
    for r in pool:
        played = parse_date(r["date"])
        if played is None or want is None:
            continue
        if (played - want).days not in (-1, 0, 1):
            continue
        s1, s2 = similar(match["home"], r["home"]), similar(match["away"], r["away"])
        if min(s1, s2) < 0.75:
            continue
        score = (s1 + s2) / 2
        if score > best_score:
            best, best_score = r, score
    return best


def market_state(market, pick, hg, ag, finished):
    """won / lost / pending for one selection, given the current score."""
    total = hg + ag

    if market == "1x2":
        if not finished:
            return "pending"
        truth = "home" if hg > ag else "draw" if hg == ag else "away"
        return "won" if pick == truth else "lost"

    if market == "btts":
        both = hg > 0 and ag > 0
        if both:
            return "won" if pick == "yes" else "lost"
        if finished:
            return "won" if pick == "no" else "lost"
        return "pending"

    line = LINES[market]
    if total > line:
        return "won" if pick == "over" else "lost"
    if finished:
        return "won" if pick == "under" else "lost"
    return "pending"


def pick_of(probs):
    return max(probs, key=lambda k: probs[k] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HIST, "live.json"))
    args = ap.parse_args()

    archive = newest_archive()
    if not archive:
        print("No archived predictions — run Predict & compare first.")
        return

    print("Fetching live scores...")
    pool = fetch_feed_matches()
    live_n = sum(1 for m in pool if m["status"] == "live")
    print(f"  {len(pool)} matches on the board ({live_n} in play)")

    totals = {m: {"decided": 0, "won": 0} for m in ("1x2", "ou15", "ou25", "ou35", "btts")}
    totals["value"] = {"decided": 0, "won": 0}
    rows = []

    for match in archive.get("matches", []):
        found = find_live(match, pool)
        if not found:
            continue
        finished = found["status"] == "finished"
        started = found["status"] != "scheduled"
        prices = match.get("market_prices") or {}
        markets = {}
        for market, probs in (match.get("markets") or {}).items():
            if not probs:
                continue
            pick = pick_of(probs)
            state = market_state(market, pick, found["hg"], found["ag"], finished) if started else "pending"
            price = (prices.get(market) or {}).get(pick)
            markets[market] = {
                "pick": pick, "prob": round(probs[pick] or 0, 4), "state": state,
                "price": round(price, 2) if price else None,
            }
            if state != "pending":
                totals[market]["decided"] += 1
                totals[market]["won"] += int(state == "won")

        # value bets are graded per selection
        for vb in archive.get("value_bets", []):
            if vb.get("match") != f"{match['home']} vs {match['away']}":
                continue
            state = (market_state(vb["market"], vb["outcome"], found["hg"], found["ag"], finished)
                     if started else "pending")
            markets.setdefault("value", {"picks": []})
            markets["value"]["picks"].append({
                "market": vb["market"], "outcome": vb["outcome"],
                "edge": vb["edge"], "price": vb.get("price"), "state": state,
            })
            if state != "pending":
                totals["value"]["decided"] += 1
                totals["value"]["won"] += int(state == "won")

        rows.append({
            "league": match.get("league"),
            "home": match["home"],
            "away": match["away"],
            "source": match.get("source", "model"),
            "status": found["status"],
            "stage": found["stage"],
            "hg": found["hg"], "ag": found["ag"],
            "markets": markets,
        })

    # live matches first, then the ones that have just finished
    order = {"live": 0, "finished": 1, "scheduled": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), r["league"], r["home"]))

    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archive": archive["id"],
        "totals": totals,
        "matches": rows,
        "live_count": sum(1 for r in rows if r["status"] == "live"),
    }
    os.makedirs(HIST, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Tracked {len(rows)} of {len(archive.get('matches', []))} archived fixtures "
          f"({payload['live_count']} in play)")
    for market in ("1x2", "ou25", "ou15", "ou35", "btts", "value"):
        t = totals[market]
        if t["decided"]:
            print(f"  {market:5s} {t['won']}/{t['decided']} decided so far")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
