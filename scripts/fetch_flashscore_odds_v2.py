"""
Fetch live odds from Flashscore and enrich predictions with value-bet analysis.

Scans Flashscore's day feeds (±7 days) to find matches by team name,
then fetches odds via the odds feed. Outputs a value-bet report.

Usage:
    python scripts/fetch_flashscore_odds_v2.py
"""
import os
import sys
import re
from datetime import datetime
from difflib import SequenceMatcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flashscore_api"))

from flashscore.fetch import get_feed
from flashscore import parsers as P

# Name mappings: your data -> Flashscore common names
TEAM_ALIASES = {
    "Nott'm Forest": "Nottingham",
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "M'gladbach": "B. Monchengladbach",
    "FC Koln": "Koln",
    "Schalke 04": "Schalke",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Bayern Munich": "Bayern",
    "Ath Bilbao": "Athletic Bilbao",
    "Vallecano": "Rayo Vallecano",
    "Sociedad": "Real Sociedad",
    "Paris SG": "Paris Saint-Germain",
    "Paris FC": "Paris FC",
}


def normalize(name: str) -> str:
    """Normalize a team name for fuzzy matching."""
    name = TEAM_ALIASES.get(name, name)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def fetch_all_matches() -> list:
    """Fetch all football matches from Flashscore for days -1 to +7."""
    all_matches = []
    for offset in range(-1, 8):
        text = get_feed(f"f_1_{offset}_5_en_1", optional=True)
        if not text:
            continue
        rows = P.match_list(text, sport_id=1, leaderboard=False)
        for r in rows:
            if r.get("status") == "finished":
                continue
            home = r.get("home", {}).get("name", "")
            away = r.get("away", {}).get("name", "")
            comp = r.get("competition", {}).get("name", "")
            all_matches.append({
                "id": r.get("id"),
                "home": home,
                "away": away,
                "competition": comp,
                "status": r.get("status"),
                "day_offset": offset,
            })
    return all_matches


def find_match(home: str, away: str, matches: list) -> dict:
    """Find the best matching match from the Flashscore list."""
    best = None
    best_score = 0.0
    for m in matches:
        s1 = similarity(home, m["home"])
        s2 = similarity(away, m["away"])
        score = (s1 + s2) / 2
        if score > best_score and score > 0.5:
            best_score = score
            best = m
    return best


def fetch_odds(match_id: str) -> dict:
    """Fetch 1X2 and O/U 2.5 odds for a match."""
    if not match_id:
        return {}

    # First get the detail feed to confirm OD tab exists
    detail = get_feed(f"dc_1_{match_id}", optional=True)
    if not detail:
        return {"error": "No detail feed"}

    tabs = []
    for rec in P.parse_records(detail):
        if "DX" in rec:
            tabs = rec.get("DX", "").split(",")
            break

    if "OD" not in tabs:
        return {"error": "No odds tab"}

    # Fetch odds feed
    odds_text = get_feed(f"df_od_1_{match_id}", optional=True)
    if not odds_text:
        return {"error": "Empty odds feed"}

    return parse_odds(odds_text)


def parse_odds(text: str) -> dict:
    """Parse Flashscore odds feed into structured data."""
    odds = {"1X2": {}, "O/U 2.5": {}}
    current_bm = None
    current_market = None

    for rec in P.parse_records(text):
        if "OB" in rec:
            current_bm = rec.get("OB", "").strip()
        if "OD" in rec:
            mid = rec.get("OD", "")
            if mid == "1":
                current_market = "1X2"
            elif mid in ("3", "7"):
                current_market = "O/U 2.5"
            else:
                current_market = None
        if "O" in rec and current_bm and current_market:
            vals = rec.get("O", "").split(";")
            if current_market == "1X2" and len(vals) >= 3:
                try:
                    odds["1X2"][current_bm] = {
                        "1": float(vals[0]),
                        "X": float(vals[1]),
                        "2": float(vals[2]),
                    }
                except ValueError:
                    pass
            elif current_market == "O/U 2.5" and len(vals) >= 2:
                try:
                    odds["O/U 2.5"][current_bm] = {
                        "Over": float(vals[0]),
                        "Under": float(vals[1]),
                    }
                except ValueError:
                    pass

    return odds


def get_best_odds(odds_dict: dict) -> dict:
    if not odds_dict:
        return {}
    best = {}
    for outcome in list(next(iter(odds_dict.values())).keys()):
        best_val = 0
        best_bm = ""
        for bm, vals in odds_dict.items():
            if vals.get(outcome, 0) > best_val:
                best_val = vals[outcome]
                best_bm = bm
        best[outcome] = {"odds": best_val, "bookmaker": best_bm}
    return best


def parse_predictions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    matches = []
    pattern = re.compile(
        r"- ⚽ \*\*(.+?)\*\* 🆚 \*\*(.+?)\*\*: "
        r"(Over 2\.5 Goals! 🔥|Under 2\.5 Goals) "
        r"\((\d+\.?\d*)% chance\)"
    )
    for m in pattern.finditer(text):
        home, away, result, pct = m.groups()
        is_over = "Over" in result
        model_over = float(pct) / 100 if is_over else 1 - float(pct) / 100
        matches.append({
            "home": home.strip(),
            "away": away.strip(),
            "model_over_prob": model_over,
            "model_under_prob": 1 - model_over,
            "prediction": "Over 2.5" if is_over else "Under 2.5",
        })
    return matches


def main():
    predictions_file = "data/final_predictions.txt"
    report_file = "data/value_bets_report.md"

    if not os.path.exists(predictions_file):
        print(f"Predictions not found: {predictions_file}")
        return

    print("Loading predictions...")
    predictions = parse_predictions(predictions_file)
    print(f"Found {len(predictions)} predicted matches\n")

    print("Fetching all Flashscore matches for next 7 days...")
    flashscore_matches = fetch_all_matches()
    print(f"Scanned {len(flashscore_matches)} matches from Flashscore\n")

    results = []
    found_count = 0

    for pred in predictions:
        home = pred["home"]
        away = pred["away"]
        match = find_match(home, away, flashscore_matches)

        if not match:
            print(f"❌ {home} vs {away}: not found on Flashscore")
            pred["flashscore"] = None
            results.append(pred)
            continue

        found_count += 1
        print(f"✅ {home} vs {away} -> {match['home']} vs {match['away']} "
              f"({match['competition']}) [ID: {match['id']}, sim={similarity(home, match['home']):.2f}]")

        odds_data = fetch_odds(match["id"])
        pred["flashscore"] = {
            "home": match["home"],
            "away": match["away"],
            "competition": match["competition"],
        }

        if "error" in odds_data:
            print(f"   ⚠️  Odds: {odds_data['error']}")
            pred["odds"] = None
            results.append(pred)
            continue

        best = get_best_odds(odds_data.get("O/U 2.5", {}))
        pred["odds"] = best

        if "Over" in best and "Under" in best:
            over_odds = best["Over"]["odds"]
            under_odds = best["Under"]["odds"]

            # Remove bookmaker margin (proportional method)
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

            ov = pred["over_value"]
            uv = pred["under_value"]
            if ov > 0.05:
                print(f"   🚀 VALUE OVER: model {pred['model_over_prob']:.1%} vs market {fair_over:.1%} (+{ov:.1%}), best odds {over_odds} @ {best['Over']['bookmaker']}")
            elif uv > 0.05:
                print(f"   🚀 VALUE UNDER: model {pred['model_under_prob']:.1%} vs market {fair_under:.1%} (+{uv:.1%}), best odds {under_odds} @ {best['Under']['bookmaker']}")
            else:
                print(f"   ➖ No edge: over {over_odds} / under {under_odds}")
        else:
            print(f"   ⚠️  Missing O/U 2.5 odds")

        results.append(pred)

    print(f"\n{'='*60}")
    print(f"Matched {found_count}/{len(predictions)} matches on Flashscore")

    # Generate report
    lines = [
        "# 🎯 AI Football Predictions + Flashscore Live Odds",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Matches found:** {found_count}/{len(predictions)} on Flashscore",
        "",
        "---",
        "",
        "## 🚀 Value Bets (Model Edge > 5%)",
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
        "| Match | Prediction | Model Over | Market Over | Market Under | Edge |",
        "|-------|------------|------------|-------------|--------------|------|",
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

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Report saved to {report_file}")


if __name__ == "__main__":
    main()
