import math
import os
import json
import pandas as pd
from bs4 import BeautifulSoup
import requests
import pickle
from datetime import date

from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.environ["ODDS_API_KEY"]
ODDS_CACHE_FILE = f"data/daily/{date.today()}_odds_cache.pkl"

CACHED_ODDS_RESULTS = None


def line_to_prob(line):
    if line is None or pd.isna(line):
        return -1
    if line < 0:
        return abs(line) / (abs(line) + 100)
    else:
        return 100 / (line + 100)


def prob_to_line(prob):
    if prob <= 0 or prob >= 1:
        return None
    if prob > 0.5:
        line = -100 * (prob / (1 - prob))
    else:
        line = 100 * ((1 - prob) / prob)
    return None if pd.isna(line) else line


def line_to_bet(model_prob):
    if model_prob is None or not (0 < model_prob < 1):
        return None
    target_implied_prob = model_prob
    if target_implied_prob <= 0 or target_implied_prob >= 1:
        return None
    if target_implied_prob > 0.5:
        line = -100 * (target_implied_prob / (1 - target_implied_prob))
    else:
        line = 100 * ((1 - target_implied_prob) / target_implied_prob)
    return math.floor(line)


def calculate_edge(model_prob, market_line):
    if model_prob is None or market_line is None:
        return None
    implied_prob = line_to_prob(market_line)
    edge = (model_prob - implied_prob) * 100
    return f"{round(edge, 2)}%"


def get_stripped_team_val(team):
    parts = team.split()
    if parts[-1] in ["Sox", "Jays"]:
        return " ".join(parts[-2:])
    return parts[-1]


def extract_total_odds(data):
    extracted_data = {}
    json_data = json.loads(data)
    for game in json_data:
        team_h = game["home_team"]
        team_v = game["away_team"]
        date = game["commence_time"]

        over_under_line = None
        over_under_price_o = None
        over_under_price_u = None
        spread_line = None
        spread_price = None
        moneyline_price_h = None
        moneyline_price_v = None

        for bookmaker in game.get("bookmakers", []):
            if bookmaker["key"] == "fanduel":
                for market in bookmaker["markets"]:
                    if market["key"] == "totals":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == "Over":
                                over_under_line = outcome["point"]
                                over_under_price_o = outcome["price"]
                            if outcome["name"] == "Under":
                                over_under_price_u = outcome["price"]
                    if market["key"] == "spreads":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == team_h:
                                spread_line = outcome["point"]
                                spread_price = outcome["price"]
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == team_h:
                                moneyline_price_h = outcome["price"]
                            if outcome["name"] == team_v:
                                moneyline_price_v = outcome["price"]

        home_team = get_stripped_team_val(team_h)
        if extracted_data.get(home_team) is None:
            extracted_data[home_team] = {
                "team_v": team_v,
                "date": date,
                "over_under_line": over_under_line,
                "spread_line": spread_line,
                "moneyline_price": moneyline_price_h,
                "over_under_price_o": over_under_price_o,
                "over_under_price_u": over_under_price_u,
                "spread_price": spread_price,
            }

        visiting_team = get_stripped_team_val(team_v)
        if extracted_data.get(visiting_team) is None:
            extracted_data[visiting_team] = {
                "team_h": team_h,
                "date": date,
                "over_under_line": over_under_line,
                "spread_line": spread_line,
                "moneyline_price": moneyline_price_v,
                "over_under_price_o": over_under_price_o,
                "over_under_price_u": over_under_price_u,
                "spread_price": spread_price,
            }

    return extracted_data


def get_odds_results():
    global CACHED_ODDS_RESULTS

    print("Updating Cache for CACHED_ODDS_RESULTS..")
    try:
        url = ...
        ...
    except Exception as e:
        print(f"Error occurred getting Odds: {e}")


def get_odds_results():
    global CACHED_ODDS_RESULTS
    if CACHED_ODDS_RESULTS is not None:
        return CACHED_ODDS_RESULTS
    if os.path.exists(ODDS_CACHE_FILE):
        with open(ODDS_CACHE_FILE, "rb") as f:
            CACHED_ODDS_RESULTS = pickle.load(f)
        return CACHED_ODDS_RESULTS
    print("Updating Cache for CACHED_ODDS_RESULTS..")
    try:
        url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&oddsFormat=american&markets=spreads,totals,h2h&apiKey={ODDS_API_KEY}"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        html = list(soup.children)[0]
        CACHED_ODDS_RESULTS = extract_total_odds(html)
        with open(ODDS_CACHE_FILE, "wb") as f:
            pickle.dump(CACHED_ODDS_RESULTS, f)
        return CACHED_ODDS_RESULTS
    except Exception as e:
        print(f"Error occurred getting Odds: {e}")


def get_hr_prop_odds():
    """Pull today's HR prop odds from the-odds-api."""
    try:
        # step 1: get event IDs
        events_resp = requests.get(
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events",
            params={"apiKey": ODDS_API_KEY, "dateFormat": "iso"},
            timeout=15,
        )
        events_resp.raise_for_status()
        events = events_resp.json()

        all_props = []
        for event in events:
            event_id = event["id"]
            try:
                resp = requests.get(
                    f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds",
                    params={
                        "apiKey": ODDS_API_KEY,
                        "regions": "us",
                        "markets": "batter_home_runs",
                        "oddsFormat": "american",
                    },
                    timeout=15,
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                for bookmaker in data.get("bookmakers", []):
                    book = bookmaker["key"]
                    for market in bookmaker.get("markets", []):
                        if market["key"] != "batter_home_runs":
                            continue
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name", "").lower() != "over":
                                continue
                            if outcome.get("point", 0.5) != 0.5:
                                continue
                            all_props.append(
                                {
                                    "player_name": outcome["description"],
                                    "book": book,
                                    "american_odds": outcome["price"],
                                }
                            )
            except Exception as e:
                continue

        return pd.DataFrame(all_props)
    except Exception as e:
        print(f"Error fetching HR prop odds: {e}")
        return pd.DataFrame()


def get_best_hr_odds(odds_df):
    """Get best available odds per player across all books."""
    if odds_df.empty:
        return pd.DataFrame()
    odds_df["implied_prob"] = odds_df["american_odds"].apply(line_to_prob)
    return (
        odds_df.sort_values("implied_prob")
        .drop_duplicates(subset="player_name")[
            ["player_name", "book", "american_odds", "implied_prob"]
        ]
        .reset_index(drop=True)
    )


def normalize_name(name):
    return name.lower().strip().replace(".", "").replace("'", "").replace("-", " ")


def match_hr_odds(pred_df, odds_df):
    """Join odds to HR predictions by normalized player name."""
    if odds_df.empty:
        return pred_df

    pred_df = pred_df.copy()
    odds_df = odds_df.copy()

    pred_df["name_normalized"] = pred_df["player_name"].apply(
        lambda x: normalize_name(str(x)) if pd.notna(x) else ""
    )
    odds_df["name_normalized"] = odds_df["player_name"].apply(
        lambda x: normalize_name(str(x)) if pd.notna(x) else ""
    )

    odds_slim = odds_df[["name_normalized", "american_odds", "implied_prob", "book"]]
    return pred_df.merge(odds_slim, on="name_normalized", how="left")


def get_money_line_price(team):
    stripped = get_stripped_team_val(team)
    res = get_odds_results().get(stripped)
    return res if res is None else res["moneyline_price"]


def get_over_odds(team):
    stripped = get_stripped_team_val(team)
    res = get_odds_results().get(stripped)
    return res if res is None else res["over_under_price_o"]


def get_under_odds(team):
    stripped = get_stripped_team_val(team)
    res = get_odds_results().get(stripped)
    return res if res is None else res["over_under_price_u"]


def get_total_line(team):
    stripped = get_stripped_team_val(team)
    res = get_odds_results().get(stripped)
    return res if res is None else res["over_under_line"]


def get_spread_line(team):
    stripped = get_stripped_team_val(team)
    res = get_odds_results().get(stripped)
    return res if res is None else res["spread_line"]


def get_spread_odds(team):
    stripped = get_stripped_team_val(team)
    res = get_odds_results().get(stripped)
    return res if res is None else res["spread_price"]
