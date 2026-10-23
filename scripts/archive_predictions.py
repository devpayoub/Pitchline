"""
Snapshot every prediction run so it can be graded against real results later.

    python scripts/archive_predictions.py

Writes data/history/predictions_<id>.json and keeps data/history/index.json
up to date. Grading is done by grade_predictions.py.
"""
import os
import json
import hashlib
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "data", "results.json")
HIST = os.path.join(BASE, "data", "history")
INDEX = os.path.join(HIST, "index.json")


def load_index():
    if not os.path.exists(INDEX):
        return []
    with open(INDEX, encoding="utf-8") as f:
        return json.load(f)


def main():
    if not os.path.exists(RESULTS):
        print("No results.json yet — run Predict & compare first.")
        return

    with open(RESULTS, encoding="utf-8") as f:
        data = json.load(f)

    matches = data.get("matches") or []
    if not matches:
        print("results.json has no matches, nothing to archive.")
        return

    stamp = data.get("generated") or datetime.now().strftime("%Y-%m-%d %H:%M")
    arch_id = stamp.replace("-", "").replace(" ", "T").replace(":", "")
    os.makedirs(HIST, exist_ok=True)

    index = load_index()

    # Only what grading needs: the picks, the probabilities behind them and the
    # market prices that were on screen when the prediction was made.
    slim = []
    for m in matches:
        slim.append({
            "league": m["league"],
            "league_code": m.get("league_code"),
            "home": m["home"],
            "away": m["away"],
            "date": m.get("date", ""),
            "source": m.get("source", "model"),
            "markets": {
                k: m["markets"][k] for k in ("1x2", "ou15", "ou25", "ou35", "btts")
                if k in m.get("markets", {})
            },
            "market_probs": {
                k: v.get("prob") for k, v in (m.get("odds") or {}).get("markets", {}).items()
            } if m.get("odds") else {},
            # the prices on screen when the pick was made: the live view shows them
            # next to the tick so you know what the bet was worth
            "market_prices": {
                k: v.get("price") for k, v in (m.get("odds") or {}).get("markets", {}).items()
            } if m.get("odds") else {},
        })

    payload = {
        "id": arch_id,
        "generated": stamp,
        "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "with_odds": data.get("with_odds", 0),
        "matches": slim,
        "value_bets": data.get("value_bets", []),
    }

    # The same matchday can be re-predicted (fresh odds, retrained model). The id is
    # the model run, so overwrite that snapshot when the content changed instead of
    # piling up near-duplicates that would be graded twice.
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    signature = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    existing = next((e for e in index if e.get("id") == arch_id), None)
    if existing and existing.get("signature") == signature:
        print(f"Archive {arch_id} is already up to date, skipping.")
        return

    path = os.path.join(HIST, f"predictions_{arch_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    entry = {
        "id": arch_id,
        "generated": stamp,
        "file": os.path.basename(path),
        "matches": len(slim),
        "with_odds": data.get("with_odds", 0),
        "value_bets": len(data.get("value_bets", [])),
        "signature": signature,
    }
    if existing:
        index = [entry if e.get("id") == arch_id else e for e in index]
        print(f"Archive {arch_id} refreshed (content changed).")
    else:
        index.insert(0, entry)
    index = index[:200]
    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Archived {len(slim)} predictions -> {path}")


if __name__ == "__main__":
    main()
