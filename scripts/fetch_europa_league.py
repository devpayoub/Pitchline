"""
Fetch UEFA Europa League fixtures and merge them into data/next_matches.json.

Why this script exists
----------------------
The football-data.org API returns 403 for the Europa League (competition 2143) on a
free subscription, and football-data.co.uk only publishes domestic leagues. TheSportsDB
also refuses its free key. Wikipedia keeps the league phase fixtures in a structured
`{{#invoke:Football box|main}}` template per matchday, which is stable enough to parse.

The fixtures are written under the 'EL' key in data/next_matches.json using the same
schema as the API-based leagues, so make_predictions.py picks them up unchanged.

Usage:
    python scripts/fetch_europa_league.py                 # next 7 days, merged in place
    python scripts/fetch_europa_league.py --window_days 14
    python scripts/fetch_europa_league.py --stats         # over-2.5 base rate from played games
"""
import os
import re
import json
import argparse
from datetime import datetime, timedelta

import requests

WIKI_RAW_URL = (
    "https://en.wikipedia.org/w/index.php"
    "?title=2026%E2%80%9327_UEFA_Europa_League_league_phase&action=raw"
)

MATCHDAY_RE = re.compile(r"===\s*Matchday\s+(\d+)\s*===")
BOX_START = "{{#invoke:Football box|main"
DATE_RE = re.compile(r"\{\{Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})")

# Wikipedia display names -> names used in data/processed (football-data.co.uk spelling)
TEAM_ALIASES = {
    "AZ": "AZ Alkmaar",
    "NEC": "Nijmegen",
    "Bayer Leverkusen": "Leverkusen",
    "TSG Hoffenheim": "Hoffenheim",
    "Celta Vigo": "Celta",
    "Real Sociedad": "Sociedad",
    "Olympiacos": "Olympiakos",
    "Beşiktaş": "Besiktas",
    "Union Saint-Gilloise": "St. Gilloise",
    "Red Bull Salzburg": "Salzburg",
    "Sparta Prague": "Sparta Prague",
    "Viktoria Plzeň": "Plzen",
    "Ferencváros": "Ferencvaros",
    "Lech Poznań": "Lech",
    "Jagiellonia Białystok": "Jagiellonia",
    "Dinamo Zagreb": "Dinamo Zagreb",
    "Levski Sofia": "Levski Sofia",
    "Hapoel Be'er Sheva": "Hapoel Be'er Sheva",
    "Lillestrøm": "Lillestrom",
    "Sturm Graz": "Sturm Graz",
    "Ararat-Armenia": "Ararat-Armenia",
}


def load_known_teams(processed_dir: str = "data/processed") -> set:
    """Every club name that appears in the processed training data."""
    import pandas as pd
    import glob

    teams = set()
    for path in glob.glob(os.path.join(processed_dir, "*.csv")):
        df = pd.read_csv(path)
        teams |= set(df["HomeTeam"]) | set(df["AwayTeam"])
    return teams


def resolve_team(name: str, known: set) -> str:
    """Map a Wikipedia name onto the training-data spelling when possible."""
    candidate = TEAM_ALIASES.get(name, name)
    return candidate if candidate in known else name



def clean_team(raw: str) -> str:
    """Turn '{{fbaicon|NED}} [[AZ Alkmaar|AZ]]' into 'AZ'."""
    link = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", raw)
    if link:
        return (link.group(2) or link.group(1)).strip()
    return re.sub(r"\{\{[^}]*\}\}", "", raw).strip()


def parse_matchdays(wikitext: str) -> dict:
    """Return {matchday_number: [ {date, home, away, score}, ... ]}."""
    matches = {}
    headings = list(MATCHDAY_RE.finditer(wikitext))

    for i, heading in enumerate(headings):
        number = int(heading.group(1))
        start = heading.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(wikitext)
        block = wikitext[start:end]

        for chunk in block.split(BOX_START)[1:]:
            body = chunk.split("\n}}")[0]
            date_match = DATE_RE.search(body)
            if not date_match:
                continue

            year, month, day = (int(g) for g in date_match.groups())
            fields = {}
            for line in body.splitlines():
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                fields[key.strip().lstrip("|").strip()] = value.strip()

            score = fields.get("score", "").strip()
            matches.setdefault(number, []).append({
                "date": datetime(year, month, day),
                "home": clean_team(fields.get("team1", "")),
                "away": clean_team(fields.get("team2", "")),
                "score": score,
            })
    return matches


def base_rate(matches: dict) -> None:
    """Print the over-2.5 base rate from matches that already have a score."""
    totals = []
    for games in matches.values():
        for game in games:
            parts = re.findall(r"\d+", game["score"])
            if len(parts) >= 2:
                totals.append(int(parts[0]) + int(parts[1]))

    if not totals:
        print("No played matches with a score found.")
        return

    over = sum(1 for t in totals if t > 2)
    print(f"Played matches parsed: {len(totals)}")
    print(f"Average goals: {sum(totals) / len(totals):.2f}")
    print(f"Over 2.5 rate: {over / len(totals) * 100:.1f}%")


def merge_into_json(path: str, fixtures: list) -> None:
    """Write (or update) the 'EL' key of next_matches.json, keeping every other league."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-16") as fh:
            payload = json.load(fh)
    else:
        payload = {}

    payload["EL"] = {
        "id": 2143,
        "crest": "https://crests.football-data.org/EL.png",
        "name": "Europa League",
        "next_matches": fixtures,
    }

    with open(path, "w", encoding="utf-16") as fh:
        json.dump(payload, fh, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Europa League fixtures from Wikipedia.")
    parser.add_argument("--output_file", type=str, default="data/next_matches.json")
    parser.add_argument("--window_days", type=int, default=7, help="How many days ahead to keep")
    parser.add_argument("--stats", action="store_true", help="Only print the over-2.5 base rate")
    args = parser.parse_args()

    response = requests.get(WIKI_RAW_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    matches = parse_matchdays(response.text)

    print("Parsed matchdays:", {k: len(v) for k, v in sorted(matches.items())})

    if args.stats:
        base_rate(matches)
        return

    now = datetime.now()
    window_end = now + timedelta(days=args.window_days)
    known = load_known_teams()

    fixtures = []
    unpredicted = set()
    for number, games in sorted(matches.items()):
        for game in games:
            if game["score"]:
                continue
            if not (now.date() <= game["date"].date() <= window_end.date()):
                continue
            home = resolve_team(game["home"], known)
            away = resolve_team(game["away"], known)
            for team in (home, away):
                if team not in known:
                    unpredicted.add(team)
            fixtures.append({
                "date": game["date"].strftime("%Y-%m-%d %H:%M:%S"),
                "home_team": home,
                "away_team": away,
                "home_team_crest": "",
                "away_team_crest": "",
            })

    fixtures.sort(key=lambda f: f["date"])
    covered = sum(1 for f in fixtures if f["home_team"] in known and f["away_team"] in known)
    print(f"Fixtures within {args.window_days} days: {len(fixtures)} ({covered} with both clubs in training data)")
    for f in fixtures:
        mark = "OK " if (f["home_team"] in known and f["away_team"] in known) else "--- "
        print(f"  {mark}{f['date']}  {f['home_team']} vs {f['away_team']}")
    if unpredicted:
        print("Clubs with no training data (matches skipped):", ", ".join(sorted(unpredicted)))

    merge_into_json(args.output_file, fixtures)
    print(f"Merged 'EL' into {args.output_file}")


if __name__ == "__main__":
    main()
