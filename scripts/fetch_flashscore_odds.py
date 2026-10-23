"""
Fetch live odds from Flashscore and enrich predictions with value-bet analysis.

This script:
1. Loads the existing predictions from data/final_predictions.txt
2. Uses the Flashscore scraper (curl_cffi) to search for matches and fetch odds
3. Compares model probabilities vs market implied probabilities
4. Outputs a value-bet report highlighting edges

Usage:
    python scripts/fetch_flashscore_odds.py
"""
import os
import sys
import re
import json
from datetime import datetime

# Add flashscore_api to path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flashscore_api"))

from flashscore.fetch import get_search, get_graphql

# Odds GraphQL hash for match odds — this is the persisted query Flashscore uses
ODDS_HASH = "odds_comparison"  # Placeholder; we discover the real hash via feed


def parse_predictions(path: str) -> list:
    """Extract match predictions from the final_predictions.txt file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    matches = []
    # Pattern: - ⚽ **Home** 🆚 **Away**: Over 2.5 Goals! 🔥 (69.38% chance)
    pattern = re.compile(
        r"- ⚽ \*\*(.+?)\*\* 🆚 \*\*(.+?)\*\*: "
        r"(Over 2\.5 Goals! 🔥|Under 2\.5 Goals) "
        r"\((\d+\.?\d*)% chance\)"
    )
    for m in pattern.finditer(text):
        home, away, result, pct = m.groups()
        is_over = "Over" in result
        matches.append({
            "home": home.strip(),
            "away": away.strip(),
            "model_over_prob": float(pct) / 100 if is_over else 1 - float(pct) / 100,
            "model_under_prob": 1 - (float(pct) / 100 if is_over else 1 - float(pct) / 100),
            "prediction": "Over 2.5" if is_over else "Under 2.5",
        })
    return matches


def search_match(home_team: str, away_team: str):
    """Search Flashscore for a match by team names. Returns match id or None."""
    query = f"{home_team} {away_team}"
    try:
        results = get_search(query, type_ids=[1])  # type 1 = matches/events
        for r in results:
            name = r.get("name", "")
            if home_team.lower() in name.lower() and away_team.lower() in name.lower():
                return r.get("id")
    except Exception as e:
        print(f"  Search error for {home_team} vs {away_team}: {e}")
    return None


def fetch_odds_for_match(match_id: str) -> dict:
    """Fetch 1X2 and over/under odds for a match via Flashscore's odds feed."""
    if not match_id:
        return {}

    try:
        # The odds feed uses a specific GraphQL hash. We use the feed API instead
        # which is more stable: dc_<sport>_<match_id> with OD tab
        from flashscore.fetch import get_feed
        from flashscore import parsers as P

        # Get the match detail feed which lists available tabs
        feed_name = f"dc_1_{match_id}"
        text = get_feed(feed_name, optional=True)
        if not text:
            return {}

        # Parse to find if odds tab exists
        records = list(P.parse_records(text))
        tabs = []
        for rec in records:
            if "DX" in rec:
                tabs = rec.get("DX", "").split(",")
                break

        if "OD" not in tabs:
            return {"error": "No odds available"}

        # Fetch odds feed: df_od_1_<match_id>
        odds_feed = f"df_od_1_{match_id}"
        odds_text = get_feed(odds_feed, optional=True)
        if not odds_text:
            return {"error": "Odds feed empty"}

        return parse_odds_feed(odds_text)
    except Exception as e:
        return {"error": str(e)}


def parse_odds_feed(text: str) -> dict:
    """Parse Flashscore odds feed text into structured odds."""
    from flashscore import parsers as P

    odds = {"1X2": {}, "O/U 2.5": {}}
    current_bookmaker = None
    current_market = None

    for rec in P.parse_records(text):
        # Bookmaker name
        if "OB" in rec:
            current_bookmaker = rec.get("OB", "").strip()
        # Market type: 1X2, O/U, etc.
        if "OD" in rec:
            market_id = rec.get("OD", "")
            if market_id == "1":
                current_market = "1X2"
            elif market_id in ("3", "7"):
                current_market = "O/U 2.5"
            else:
                current_market = None
        # Odds values
        if "O" in rec and current_bookmaker and current_market:
            odds_vals = rec.get("O", "").split(";")
            if current_market == "1X2" and len(odds_vals) >= 3:
                try:
                    odds["1X2"][current_bookmaker] = {
                        "1": float(odds_vals[0]),
                        "X": float(odds_vals[1]),
                        "2": float(odds_vals[2]),
                    }
                except ValueError:
                    pass
            elif current_market == "O/U 2.5" and len(odds_vals) >= 2:
                try:
                    odds["O/U 2.5"][current_bookmaker] = {
                        "Over": float(odds_vals[0]),
                        "Under": float(odds_vals[1]),
                    }
                except ValueError:
                    pass

    return odds


def get_best_odds(odds_dict: dict) -> dict:
    """Extract best (highest) odds from all bookmakers for a market."""
    if not odds_dict:
        return {}
    best = {}
    for outcome in list(next(iter(odds_dict.values())).keys()):
        best[outcome] = max(
            (bm.get(outcome, 0), bm_name)
            for bm_name, bm in odds_dict.items()
        )
    return {k: {"odds": v[0], "bookmaker": v[1]} for k, v in best.items()}


def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability (with margin removal)."""
    if odds <= 0:
        return 0
    raw = 1 / odds
    # Simple margin removal: assume proportional method
    # For a 2-outcome market, we normalize by sum of raw probs
    return raw


def analyze_value_bets(predictions: list) -> list:
    """Fetch odds for each predicted match and identify value bets."""
    results = []

    for pred in predictions:
        home = pred["home"]
        away = pred["away"]
        print(f"Searching: {home} vs {away}...")

        match_id = search_match(home, away)
        if not match_id:
            print(f"  ❌ Match not found on Flashscore")
            pred["odds"] = None
            results.append(pred)
            continue

        print(f"  ✅ Found match ID: {match_id}")
        odds_data = fetch_odds_for_match(match_id)

        if "error" in odds_data:
            print(f"  ⚠️  {odds_data['error']}")
            pred["odds"] = None
            results.append(pred)
            continue

        best = get_best_odds(odds_data.get("O/U 2.5", {}))
        pred["odds"] = best

        if "Over" in best and "Under" in best:
            over_odds = best["Over"]["odds"]
            under_odds = best["Under"]["odds"]

            # Remove margin (proportional method)
            raw_over = 1 / over_odds
            raw_under = 1 / under_odds
            margin = raw_over + raw_under - 1
            if margin > 0:
                fair_over = raw_over / (1 + margin)
                fair_under = raw_under / (1 + margin)
            else:
                fair_over = raw_over
                fair_under = raw_under

            pred["market_over_prob"] = fair_over
            pred["market_under_prob"] = fair_under
            pred["over_value"] = pred["model_over_prob"] - fair_over
            pred["under_value"] = pred["model_under_prob"] - fair_under

            print(f"  📊 Over: {over_odds} (fair {fair_over:.1%}), Under: {under_odds} (fair {fair_under:.1%})")
            if pred["over_value"] > 0.05:
                print(f"  🚀 VALUE OVER: model {pred['model_over_prob']:.1%} vs market {fair_over:.1%} (+{pred['over_value']:.1%})")
            elif pred["under_value"] > 0.05:
                print(f"  🚀 VALUE UNDER: model {pred['model_under_prob']:.1%} vs market {fair_under:.1%} (+{pred['under_value']:.1%})")
            else:
                print(f"  ➖ No significant edge")
        else:
            print(f"  ⚠️  Missing O/U 2.5 odds")

        results.append(pred)

    return results


def generate_report(results: list, out_path: str):
    """Generate a markdown report with value bet analysis."""
    lines = [
        "# 🎯 AI Football Predictions + Flashscore Live Odds",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "**Data source:** Flashscore (scraped live)",
        "",
        "---",
        "",
        "## Value Bets (Model Edge > 5%)",
        "",
    ]

    value_bets = [r for r in results if r.get("odds") and (r.get("over_value", 0) > 0.05 or r.get("under_value", 0) > 0.05)]
    value_bets.sort(key=lambda x: max(x.get("over_value", 0), x.get("under_value", 0)), reverse=True)

    if not value_bets:
        lines.append("*No value bets found with >5% edge today.*")
    else:
        for r in value_bets:
            home, away = r["home"], r["away"]
            if r.get("over_value", 0) > 0.05:
                edge = r["over_value"]
                pick = "Over 2.5"
                model_p = r["model_over_prob"]
                market_p = r["market_over_prob"]
                odds = r["odds"]["Over"]["odds"]
                bm = r["odds"]["Over"]["bookmaker"]
            else:
                edge = r["under_value"]
                pick = "Under 2.5"
                model_p = r["model_under_prob"]
                market_p = r["market_under_prob"]
                odds = r["odds"]["Under"]["odds"]
                bm = r["odds"]["Under"]["bookmaker"]

            lines.append(f"### {home} vs {away}")
            lines.append(f"- **Pick:** {pick}")
            lines.append(f"- **Model probability:** {model_p:.1%}")
            lines.append(f"- **Market probability:** {market_p:.1%}")
            lines.append(f"- **Edge:** +{edge:.1%}")
            lines.append(f"- **Best odds:** {odds} @ {bm}")
            lines.append("")

    lines.extend([
        "---",
        "",
        "## All Matches with Odds",
        "",
        "| Match | Prediction | Model | Market Over | Market Under | Edge |",
        "|-------|------------|-------|-------------|--------------|------|",
    ])

    for r in results:
        home, away = r["home"], r["away"]
        pred = r["prediction"]
        model = f"{r['model_over_prob']:.0%}"
        if r.get("odds"):
            m_over = f"{r['market_over_prob']:.0%}" if "market_over_prob" in r else "-"
            m_under = f"{r['market_under_prob']:.0%}" if "market_under_prob" in r else "-"
            edge = max(r.get("over_value", 0), r.get("under_value", 0))
            edge_str = f"+{edge:.1%}" if edge > 0.05 else f"{edge:.1%}"
        else:
            m_over = m_under = edge_str = "N/A"
        lines.append(f"| {home} vs {away} | {pred} | {model} | {m_over} | {m_under} | {edge_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Disclaimer: This is for informational purposes only. Past performance does not guarantee future results.*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Report saved to {out_path}")


def main():
    predictions_file = "data/final_predictions.txt"
    report_file = "data/value_bets_report.md"

    if not os.path.exists(predictions_file):
        print(f"Predictions file not found: {predictions_file}")
        print("Run: python scripts/make_predictions.py ...")
        return

    print("Loading predictions...")
    predictions = parse_predictions(predictions_file)
    print(f"Found {len(predictions)} predicted matches\n")

    print("Fetching live odds from Flashscore...")
    results = analyze_value_bets(predictions)

    print(f"\nGenerating report...")
    generate_report(results, report_file)


if __name__ == "__main__":
    main()
