"""
Grade archived predictions against real results.

    python scripts/grade_predictions.py [--days 10]

Results come from two places:
  * data/raw/*_merged.csv  — every finished domestic match already on disk
  * Flashscore day feeds    — the last `--days` of finished matches, which also
                              covers the Champions League and Europa League

Output: data/history/graded.json, read by the Accuracy tab.
"""
import os
import re
import sys
import json
import glob
import argparse
from datetime import datetime, timedelta
from difflib import SequenceMatcher

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(BASE, "data", "history")
INDEX = os.path.join(HIST, "index.json")
OUT = os.path.join(HIST, "graded.json")
sys.path.insert(0, os.path.join(BASE, "flashscore_api"))

MARKETS = ("1x2", "ou15", "ou25", "ou35", "btts")


from teams import normalize as norm, side_similarity      # noqa: E402


def similar(a, b):
    return side_similarity(a, b)


# --------------------------------------------------------------------------- #
# Result sources
# --------------------------------------------------------------------------- #
def results_from_raw():
    out = []
    for path in glob.glob(os.path.join(BASE, "data", "raw", "*_merged.csv")):
        try:
            import pandas as pd
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(df.columns):
            continue
        for row in df.itertuples(index=False):
            try:
                hg, ag = int(row.FTHG), int(row.FTAG)
            except (TypeError, ValueError):
                continue
            out.append({
                "home": str(row.HomeTeam), "away": str(row.AwayTeam),
                "date": str(row.Date), "hg": hg, "ag": ag, "src": "raw",
            })
    return out


def results_from_flashscore(days=10):
    try:
        from flashscore.fetch import get_feed
        from flashscore import parsers as P
    except Exception as exc:
        print(f"  Flashscore unavailable ({exc})")
        return []

    out = []
    for offset in range(-days, 1):
        text = get_feed(f"f_1_{offset}_5_en_1", optional=True)
        if not text:
            continue
        for r in P.match_list(text, sport_id=1, leaderboard=False):
            if r.get("status") != "finished":
                continue
            hg, ag = r.get("home_score"), r.get("away_score")
            if hg is None or ag is None:
                continue
            out.append({
                "home": (r.get("home") or {}).get("name", ""),
                "away": (r.get("away") or {}).get("name", ""),
                "date": str(r.get("date") or "")[:10],
                "hg": int(hg), "ag": int(ag), "src": "flashscore",
            })
    return out


def parse_date(value):
    """football-data.co.uk writes dd/mm/yyyy, Flashscore writes yyyy-mm-dd."""
    text = str(value or "")[:10]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def find_result(match, pool):
    """Match on both club names AND on the date.

    The date is not optional: without it, a fixture that has not been played
    yet silently picks up an old meeting between the same two clubs and grades
    a prediction against last season's score.
    """
    want = parse_date(match.get("date"))
    if want is None:
        return None

    best, best_score = None, 0.0
    for r in pool:
        played = parse_date(r["date"])
        if played is None:
            continue
        delta = (played - want).days
        if delta < 0 or delta > 3:      # result must land on/after kick-off
            continue
        s1, s2 = similar(match["home"], r["home"]), similar(match["away"], r["away"])
        if min(s1, s2) < 0.75:
            continue
        score = (s1 + s2) / 2
        if score > best_score:
            best, best_score = r, score
    return best


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def outcome_of(hg, ag):
    return "home" if hg > ag else "draw" if hg == ag else "away"


def grade_markets(markets, hg, ag):
    total = hg + ag
    truth = {
        "1x2": outcome_of(hg, ag),
        "ou15": "over" if total > 1.5 else "under",
        "ou25": "over" if total > 2.5 else "under",
        "ou35": "over" if total > 3.5 else "under",
        "btts": "yes" if hg > 0 and ag > 0 else "no",
    }
    picks, hits = {}, {}
    for market in MARKETS:
        probs = markets.get(market) or {}
        if not probs:
            continue
        pick = max(probs, key=lambda k: probs[k] or 0)
        picks[market] = {"pick": pick, "prob": round(probs[pick] or 0, 4)}
        hits[market] = (pick == truth[market])
    return picks, hits, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10, help="how far back to scan Flashscore")
    args = ap.parse_args()

    if not os.path.exists(INDEX):
        print("No archived predictions yet — run Predict & compare first.")
        return

    with open(INDEX, encoding="utf-8") as f:
        index = json.load(f)

    print("Loading results...")
    pool = results_from_raw()
    print(f"  {len(pool)} finished matches from data/raw")
    fs = results_from_flashscore(args.days)
    print(f"  {len(fs)} finished matches from Flashscore (last {args.days} days)")
    pool += fs

    totals = {m: {"correct": 0, "graded": 0} for m in MARKETS}
    totals["value"] = {"correct": 0, "graded": 0}
    runs, recent = [], []
    # Re-running Predict for the same matchday archives the fixtures again. Grade
    # each fixture once, using the newest snapshot that covers it, otherwise one
    # match is counted several times and the accuracy is skewed.
    seen = set()

    for entry in index:
        path = os.path.join(HIST, entry["file"])
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            arch = json.load(f)

        run = {"id": arch["id"], "generated": arch.get("generated"), "markets": {}, "graded": 0}
        for match in arch.get("matches", []):
            key = (norm(match["home"]), norm(match["away"]), str(match.get("date", ""))[:10])
            if key in seen:
                continue
            seen.add(key)
            res = find_result(match, pool)
            if not res:
                continue
            picks, hits, truth = grade_markets(match.get("markets", {}), res["hg"], res["ag"])
            if not picks:
                continue
            run["graded"] += 1
            for market, ok in hits.items():
                run["markets"].setdefault(market, {"correct": 0, "graded": 0})
                run["markets"][market]["graded"] += 1
                run["markets"][market]["correct"] += int(ok)
                totals[market]["graded"] += 1
                totals[market]["correct"] += int(ok)

            # Value bets are graded per selection, not per match
            for vb in arch.get("value_bets", []):
                if vb.get("match") != f"{match['home']} vs {match['away']}":
                    continue
                won = (vb["outcome"] == truth.get(vb["market"]))
                totals["value"]["graded"] += 1
                totals["value"]["correct"] += int(won)

            recent.append({
                "league": match.get("league"),
                "home": match["home"], "away": match["away"],
                "score": f"{res['hg']}-{res['ag']}",
                "picks": picks, "hits": hits, "truth": truth,
                "source": match.get("source", "model"),
                "archive": arch["id"],
            })
        if run["graded"]:
            runs.append(run)

    for market, t in totals.items():
        t["accuracy"] = round(t["correct"] / t["graded"], 4) if t["graded"] else None

    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "runs": runs,
        "totals": totals,
        "recent": sorted(recent, key=lambda r: r["archive"], reverse=True)[:60],
        "pending": sum(1 for e in index for _ in [0]) and None,
    }
    payload.pop("pending")

    os.makedirs(HIST, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nGraded {len(recent)} predictions from {len(runs)} archive(s)")
    for market in ("1x2", "ou25", "ou15", "ou35", "btts", "value"):
        t = totals[market]
        acc = f"{t['accuracy'] * 100:.1f}%" if t["accuracy"] is not None else "—"
        print(f"  {market:5s} {t['correct']:3d}/{t['graded']:3d}  {acc}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
