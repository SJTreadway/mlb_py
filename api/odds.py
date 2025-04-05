import math
import os
import json
import pandas as pd
from bs4 import BeautifulSoup
import requests

from dotenv import load_dotenv
load_dotenv()

ODDS_API_KEY = os.environ['ODDS_API_KEY']

CACHED_ODDS_RESULTS = None

def line_to_prob(line):
  if line is None:
      return -1
  if line < 0:
    # For negative lines, calculate the implied probability
    imp_prob = abs(line) / (abs(line) + 100)
  else:
    # For positive lines, calculate the implied probability
    imp_prob = 100 / (line + 100)
  return imp_prob

def prob_to_line(prob):
  if prob <= 0 or prob >= 1:
    return None  # invalid probability

  if prob > 0.5:
    # Favorite → negative line
    line = -100 * (prob / (1 - prob))
  else:
    # Underdog → positive line
    line = 100 * ((1 - prob) / prob)

  return None if pd.isna(line) else line

def line_to_bet(model_prob, advantage=0.04):
  if model_prob is None or not (0 < model_prob < 1):
    return None

  # The implied line you'd be willing to bet at:
  target_implied_prob = model_prob - advantage

  if target_implied_prob <= 0 or target_implied_prob >= 1:
    return None  # Avoid divide-by-zero or nonsensical odds

  # Convert target implied probability to American odds
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
  return f'{round(edge,2)}%'

def extract_total_odds(data):
  extracted_data = {}
  json_data = json.loads(data)
  for game in json_data:
    team_h = game['home_team']
    team_v = game['away_team']
    date = game['commence_time']
    
    over_under_line = None
    over_under_price_o = None
    over_under_price_u = None
    spread_line = None
    spread_price = None
    moneyline_price_h = None
    moneyline_price_v = None
    
    # Find the totals market
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
              if outcome["name"] == team_h:  # Get spread for home team
                spread_line = outcome["point"]
                spread_price = outcome["price"]
          if market["key"] == "h2h":
            for outcome in market["outcomes"]:
              if outcome["name"] == team_h:  # Get moneyline for home team
                moneyline_price_h = outcome["price"]
              if outcome["name"] == team_v:  # Get moneyline for visiting team
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
        "spread_price": spread_price
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
        "spread_price": spread_price
      }
  return extracted_data

def get_odds_results():
  global CACHED_ODDS_RESULTS
  if CACHED_ODDS_RESULTS is not None:
    return CACHED_ODDS_RESULTS
  print('Updating Cache for CACHED_ODDS_RESULTS..')
  url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&oddsFormat=american&markets=spreads,totals,h2h&apiKey={ODDS_API_KEY}'
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  html=list(soup.children)[0]
  CACHED_ODDS_RESULTS = extract_total_odds(html)
  return CACHED_ODDS_RESULTS

def get_money_line_price(team):
  res = get_odds_results().get(get_stripped_team_val(team), 'Not Found')
  return res if res == 'Not Found' else res['moneyline_price']

def get_over_odds(team):
  res = get_odds_results().get(team, 'Not Found')
  return res if res == 'Not Found' else res['over_under_price_o']

def get_under_odds(team):
  res = get_odds_results().get(team, 'Not Found')
  return res if res == 'Not Found' else res['over_under_price_u']

def get_total_line(team):
  res = get_odds_results().get(team, 'Not Found')
  return res if res == 'Not Found' else res['over_under_line']

def get_spread_line(team):
  res = get_odds_results().get(team, 'Not Found')
  return res if res == 'Not Found' else res['spread_line']

def get_spread_odds(team):
  res = get_odds_results().get(team, 'Not Found')
  return res if res == 'Not Found' else res['spread_price']

def get_stripped_team_val(team):
  return " ".join(team.split()[-2:]) if team.split()[-1] in ['Sox', 'Jays'] else team.split()[-1]