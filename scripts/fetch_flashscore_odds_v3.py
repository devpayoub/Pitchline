"""
Fetch live odds from Flashscore and enrich predictions with value-bet analysis.

Scans Flashscore's day feeds (±7 days) to find matches by team name,
then fetches odds via the GraphQL odds API. Outputs a value-bet report.

Usage:
    python scripts/fetch_flashscore_odds_v3.py
"""
import os
import sys
import re
from datetime import datetime
from difflib import SequenceMatcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flashscore_api"))

from flashscore.fetch import get_feed, get_graphql
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
    name = TEAM_ALIASES.get(name, name)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def fetch_all_matches() -> list:
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
            })
    return all_matches


def find_match(home: str, away: str, matches: list) -> dict:
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


def fetch_ou25_odds(match_id: str) -> dict:
    """Fetch Over/Under 2.5 odds for a match via GraphQL."""
    if not match_id:
        return {}

    try:
        payload = get_graphql(
            "oce",
            {"eventId": match_id, "projectId": 2, "geoIpCode": "US", "geoIpSubdivisionCode": ""},
            odds=True,
        )
    except Exception as e:
        return {"error": str(e)}

    holder = payload.get("findOddsByEventId") or {}
    bookmakers = {}
    for entry in holder.get("settings", {}).get("bookmakers", []):
        book = entry.get("bookmaker") or {}
        bookmakers[book.get("id")] = P.clean(book.get("name"))

    ou25 = {}
    for row in holder.get("odds", []):
        if row.get("bettingType") != "OVER_UNDER":
            continue
        if row.get("bettingScope") != "FULL_TIME":
            continue

        bm_id = row.get("bookmakerId")
        bm_name = bookmakers.get(bm_id, f"BM:{bm_id}")

        for sel in row.get("odds", []):
            handicap = sel.get("handicap") or {}
            handicap_val = handicap.get("value") if isinstance(handicap, dict) else None
            if handicap_val != "2.5":
                continue

            selection = sel.get("selection", "").upper()
            value = sel.get("value")
            if selection in ("OVER", "UNDER") and value is not None:
                if bm_name not in ou25:
                    ou25[bm_name] = {}
                ou25[bm_name][selection] = float(value)

    return ou25


def get_best_odds(odds_dict: dict) -> dict:
    if not odds_dict:
        return {}
    best = {}
    for outcome in ("OVER", "UNDER"):
        best_val = 0
        best_bm = ""
        for bm, vals in odds_dict.items():
            if outcome in vals and vals[outcome] > best_val:
                best_val = vals[outcome]
                best_bm = bm
        if best_val > 0:
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
    # The legacy prediction script writes to the repo root, so accept either location.
    predictions_file = "final_predictions.txt"
    if not os.path.exists(predictions_file):
        predictions_file = "data/final_predictions.txt"
    report_file = "data/value_bets_report.md"

    if len(sys.argv) > 1:
        predictions_file = sys.argv[1]
    if len(sys.argv) > 2:
        report_file = sys.argv[2]

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
    odds_found = 0

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
              f"({match['competition']}) [sim={similarity(home, match['home']):.2f}]")

        pred["flashscore"] = {
            "home": match["home"],
            "away": match["away"],
            "competition": match["competition"],
        }

        odds_data = fetch_ou25_odds(match["id"])

        if "error" in odds_data:
            print(f"   ⚠️  Odds error: {odds_data['error']}")
            pred["odds"] = None
            results.append(pred)
            continue

        if not odds_data:
            print(f"   ⚠️  No O/U 2.5 odds available")
            pred["odds"] = None
            results.append(pred)
            continue

        odds_found += 1
        best = get_best_odds(odds_data)
        pred["odds"] = best

        if "OVER" in best and "UNDER" in best:
            over_odds = best["OVER"]["odds"]
            under_odds = best["UNDER"]["odds"]

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
                print(f"   🚀 VALUE OVER: model {pred['model_over_prob']:.1%} vs market {fair_over:.1%} (+{ov:.1%}), best odds {over_odds} @ {best['OVER']['bookmaker']}")
            elif uv > 0.05:
                print(f"   🚀 VALUE UNDER: model {pred['model_under_prob']:.1%} vs market {fair_under:.1%} (+{uv:.1%}), best odds {under_odds} @ {best['UNDER']['bookmaker']}")
            else:
                print(f"   ➖ No edge: over {over_odds} / under {under_odds}")
        else:
            print(f"   ⚠️  Incomplete O/U 2.5 odds")

        results.append(pred)

    print(f"\n{'='*60}")
    print(f"Matched {found_count}/{len(predictions)} matches on Flashscore")
    print(f"Got odds for {odds_found}/{found_count} matches")

    # Generate report
    lines = [
        "# 🎯 AI Football Predictions + Flashscore Live Odds",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Matches found:** {found_count}/{len(predictions)} on Flashscore",
        f"**Matches with odds:** {odds_found}/{found_count}",
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
                odds = r["odds"]["OVER"]["odds"]
                bm = r["odds"]["OVER"]["bookmaker"]
            else:
                edge = r["under_value"]
                pick = "Under 2.5"
                model_p = r["model_under_prob"]
                market_p = r["market_under_prob"]
                odds = r["odds"]["UNDER"]["odds"]
                bm = r["odds"]["UNDER"]["bookmaker"]

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
