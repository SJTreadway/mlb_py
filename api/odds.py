import math
import os
import json
import pandas as pd
import numpy as np
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
    imp_prob = line / (line + 100)
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

def line_to_bet(implied_prob, advantage=0.04):
  if implied_prob is None:
    return -1

  # Adjust the probability by the desired advantage
  # Ensure the target probability is within a valid range
  if implied_prob > 0.5:
    target_prob = implied_prob - advantage
  else:
    target_prob = implied_prob + advantage
  
  # Convert adjusted probability back to line
  if implied_prob > 0.5:  # Favorite case (negative line)
    new_line = -100 * (1 - target_prob) / target_prob
  else:  # Underdog case (positive line)
    new_line = 100 * target_prob / (1 - target_prob)

  rounded_line = None if pd.isna(new_line) else math.floor(new_line)
  return None if rounded_line is None else 100 if -99 <= rounded_line <= 99 else rounded_line

def extract_total_odds(data):
  extracted_data = {}
  json_data = json.loads(data)
  for game in json_data:
    team_h = game['home_team']
    team_v = game['away_team']
    date = game['commence_time']
    
    over_under_line = None
    over_under_price = None
    spread_line = None
    spread_price = None
    moneyline_price = None
    
    # Find the totals market
    for bookmaker in game.get("bookmakers", []):
      if bookmaker["key"] == "fanduel":
        for market in bookmaker["markets"]:
          if market["key"] == "totals":
            for outcome in market["outcomes"]:
              if outcome["name"] == "Over":
                over_under_line = outcome["point"]
                over_under_price = outcome["price"]
          if market["key"] == "spreads":
            for outcome in market["outcomes"]:
              if outcome["name"] == team_h:  # Get spread for home team
                spread_line = outcome["point"]
                spread_price = outcome["price"]
          if market["key"] == "h2h":
            for outcome in market["outcomes"]:
              if outcome["name"] == team_h:  # Get moneyline for home team
                moneyline_price = outcome["price"]
    home_team = get_stripped_team_val(team_h)
    if extracted_data.get(home_team) is None:
      extracted_data[home_team] = {
        "team_v": team_v,
        "date": date,
        "over_under_line": over_under_line,
        "spread_line": spread_line,
        "moneyline_price": moneyline_price,
        "over_under_price": line_to_prob(over_under_price),
        "spread_price": line_to_prob(spread_price)
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

def get_total_odds(team):
  res = get_odds_results().get(team, 'Not Found')
  return res if res == 'Not Found' else res['over_under_price']

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