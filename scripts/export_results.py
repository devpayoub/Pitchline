"""
Turn the trained models into structured JSON for the UI.

    python scripts/export_results.py model     # -> data/results_model.json
    python scripts/export_results.py merge     # -> data/results.json (model + live odds)

The "model" stage must run before fetch_odds_v4.py (it tells the odds scraper
which fixtures to look for); "merge" runs afterwards.
"""
import os
import sys
import json
import argparse
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))

import make_predictions_full as mpf      # noqa: E402

MODEL_FILES = (
    ("ou25", "voting_classifier"),
    ("1x2", "result_classifier"),
    ("btts", "btts_classifier"),
    ("ou15", "over15_classifier"),
    ("ou35", "over35_classifier"),
)

# Competition priors, used only for fixtures whose clubs have no history in the
# pooled training set (Armenia, Croatia, Cyprus, Czechia, Hungary, Israel,
# Slovenia, Bulgaria, Scotland … are not on football-data.co.uk). Those rows are
# emitted with source="baseline" so the UI can mark them as low confidence —
# better than hiding two thirds of a Europa League matchday.
BASELINE = {
    "EL": {
        "1x2": {"home": 0.40, "draw": 0.26, "away": 0.34},
        "ou15": {"over": 0.80, "under": 0.20},
        "ou25": {"over": 0.639, "under": 0.361},   # measured on 36 played EL matches
        "ou35": {"over": 0.38, "under": 0.62},
        "btts": {"yes": 0.55, "no": 0.45},
    },
    "CL": {
        "1x2": {"home": 0.44, "draw": 0.25, "away": 0.31},
        "ou15": {"over": 0.86, "under": 0.14},
        "ou25": {"over": 0.733, "under": 0.267},   # 2026-27 league phase
        "ou35": {"over": 0.48, "under": 0.52},
        "btts": {"yes": 0.62, "no": 0.38},
    },
}


def with_derived(markets):
    """Add double-chance and draw-no-bet, which are arithmetic on the 1X2 split."""
    p = markets.get("1x2") or {}
    h, d, a = p.get("home") or 0, p.get("draw") or 0, p.get("away") or 0
    markets["dc"] = {"1x": h + d, "12": h + a, "x2": d + a}
    nd = h + a
    markets["dnb"] = {"home": h / nd if nd else 0.5, "away": a / nd if nd else 0.5}
    return markets


def load_models(models_dir, model_league):
    models = {}
    for key, suffix in MODEL_FILES:
        path = os.path.join(models_dir, f"{model_league}_{suffix}.pkl")
        if os.path.exists(path):
            models[key] = mpf.load_model(path)
    return models


def stage_model(args):
    with open(args.next_matches, encoding="utf-16") as f:
        competitions = json.load(f)

    matches = []
    baselined = 0
    for league in mpf.VALID_LEAGUES:
        model_league = mpf.MODEL_ALIAS.get(league, league)
        data_path = os.path.join(args.input_data_predict_dir, f"{model_league}_merged_preprocessed.csv")
        if not os.path.exists(data_path):
            continue
        models = load_models(args.input_leagues_models_dir, model_league)
        if not models:
            continue
        league_data = mpf.load_league_data(data_path)
        info = competitions.get(league) or {}
        for match in info.get("next_matches", []):
            pred = mpf.make_match_predictions(match, league, models, league_data,
                                              args.input_leagues_models_dir, model_league)
            if not pred:
                base = BASELINE.get(league)
                if not base:
                    continue
                derived = with_derived({k: dict(v) for k, v in base.items()})
                pred = {
                    "home": match["home_team"],
                    "away": match["away_team"],
                    "date": match.get("date", ""),
                    "source": "baseline",
                    "dc_1x": derived["dc"]["1x"], "dc_12": derived["dc"]["12"],
                    "dc_x2": derived["dc"]["x2"],
                    "dnb_home": derived["dnb"]["home"], "dnb_away": derived["dnb"]["away"],
                    "p_home": base["1x2"]["home"], "p_draw": base["1x2"]["draw"],
                    "p_away": base["1x2"]["away"],
                    "p_over15": base["ou15"]["over"], "p_under15": base["ou15"]["under"],
                    "p_over25": base["ou25"]["over"], "p_under25": base["ou25"]["under"],
                    "p_over35": base["ou35"]["over"], "p_under35": base["ou35"]["under"],
                    "p_btts_yes": base["btts"]["yes"], "p_btts_no": base["btts"]["no"],
                }
                baselined += 1
            matches.append({
                "league": info.get("name", league),
                "league_code": league,
                "home": pred["home"],
                "away": pred["away"],
                "date": pred.get("date", ""),
                "source": pred.get("source", "model"),
                "markets": {
                    "1x2": {"home": pred.get("p_home"), "draw": pred.get("p_draw"),
                            "away": pred.get("p_away")},
                    "ou15": {"over": pred.get("p_over15"), "under": pred.get("p_under15")},
                    "ou25": {"over": pred.get("p_over25"), "under": pred.get("p_under25")},
                    "ou35": {"over": pred.get("p_over35"), "under": pred.get("p_under35")},
                    "btts": {"yes": pred.get("p_btts_yes"), "no": pred.get("p_btts_no")},
                    "dc": {"1x": pred.get("dc_1x"), "12": pred.get("dc_12"), "x2": pred.get("dc_x2")},
                    "dnb": {"home": pred.get("dnb_home"), "away": pred.get("dnb_away")},
                },
            })
        print(f"  {league}: {sum(1 for m in matches if m['league_code'] == league)} matches")

    payload = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "matches": matches, "baseline": baselined}
    with open(args.model_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n{len(matches)} predictions ({baselined} baseline, no club history) -> {args.model_out}")


def stage_merge(args):
    with open(args.model_out, encoding="utf-8") as f:
        payload = json.load(f)

    odds = {"matches": {}}
    if os.path.exists(args.odds):
        with open(args.odds, encoding="utf-8") as f:
            odds = json.load(f)

    def edge(model_p, market_p):
        if model_p is None or market_p is None:
            return None
        return model_p - market_p

    value_bets = []
    with_odds = 0

    for m in payload["matches"]:
        key = f"{m['home']} vs {m['away']}"
        o = odds["matches"].get(key)
        m["odds"] = None
        m["edges"] = {}
        if not o:
            continue
        with_odds += 1
        m["odds"] = {
            "competition": o["flashscore"]["competition"],
            "similarity": o.get("similarity"),
            "markets": o["markets"],
        }
        for market, outcomes in (("1x2", ("home", "draw", "away")),
                                 ("ou15", ("over", "under")),
                                 ("ou25", ("over", "under")),
                                 ("ou35", ("over", "under")),
                                 ("btts", ("yes", "no"))):
            mk = o["markets"].get(market)
            if not mk:
                continue
            for outcome in outcomes:
                e = edge(m["markets"][market].get(outcome), mk["prob"].get(outcome))
                if e is not None:
                    m["edges"][f"{market}_{outcome}"] = round(e, 4)
                    if e > 0.05:
                        value_bets.append({
                            "match": key, "league": m["league"], "market": market,
                            "outcome": outcome, "edge": round(e, 4),
                            "model": round(m["markets"][market][outcome], 4),
                            "market_prob": round(mk["prob"][outcome], 4),
                            "price": mk["price"].get(outcome),
                        })

    value_bets.sort(key=lambda x: -x["edge"])
    payload["odds_summary"] = odds.get("summary", {})
    payload["with_odds"] = with_odds
    payload["value_bets"] = value_bets

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{len(payload['matches'])} matches, {with_odds} with odds, {len(value_bets)} value bets")
    print(f"Saved {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["model", "merge"])
    ap.add_argument("--input_leagues_models_dir", default=os.path.join(BASE, "models"))
    ap.add_argument("--input_data_predict_dir", default=os.path.join(BASE, "data", "processed"))
    ap.add_argument("--next_matches", default=os.path.join(BASE, "data", "next_matches.json"))
    ap.add_argument("--model_out", default=os.path.join(BASE, "data", "results_model.json"))
    ap.add_argument("--odds", default=os.path.join(BASE, "data", "odds.json"))
    ap.add_argument("--out", default=os.path.join(BASE, "data", "results.json"))
    args = ap.parse_args()

    if args.stage == "model":
        stage_model(args)
    else:
        stage_merge(args)


if __name__ == "__main__":
    main()
