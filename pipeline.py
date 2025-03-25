#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import requests
import time
import io
import os
from datetime import date, datetime, timedelta
from pytz import timezone
import pickle

from pybaseball import playerid_lookup

from bs4 import BeautifulSoup

from api.pitchers import get_full_pitching_data
from api.batters import get_full_batting_data

import tweepy

from dotenv import load_dotenv
load_dotenv()

# X essentials
ACCESS_KEY = os.environ['X_ACCESS_KEY']
ACCESS_SECRET = os.environ['X_ACCESS_SECRET']
CONSUMER_KEY = os.environ['X_CONSUMER_KEY']
CONSUMER_SECRET = os.environ['X_CONSUMER_SECRET']
BEARER_TOKEN = os.environ['X_BEARER_TOKEN']

# Flags for Settings
TOMORROW_GAMES = int(os.environ['TOMORROW_GAMES'])

# Rotowire URL for Daily Lineups
RW_URL = "https://www.rotowire.com/baseball/daily-lineups.php"

def get_lineups():
  url = RW_URL if TOMORROW_GAMES == 0 else RW_URL + '?date=tomorrow'
  soup = BeautifulSoup(requests.get(url).content, "html.parser")

  all_data = []
  team_type = ''
  current_game = {}  # Store home and away teams separately before merging

  for e in soup.select('.lineup__box ul li'):
    if team_type != e.parent.get('class')[-1]:
      order_count = 1
      team_type = e.parent.get('class')[-1]
      if current_game:  
        all_data.append(current_game)  # Save previous game before starting a new one
      current_game = {}

    if e.get('class') and 'lineup__player-highlight' in e.get('class'):
      # Pitcher Data
      if e.a is not None:
        name = e.a.get_text(strip=True).split(' ')
        f_name, l_name = name[0], name[-1]
        pitcherid = playerid_lookup(l_name, f_name).get('key_retro')
        suffix = "_h" if team_type == "is-home" else "_v"

        # Check if pitcherid lookup returned a value
        pitcherid = pitcherid.iloc[0] if isinstance(pitcherid, pd.Series) and not pitcherid.empty else ''
        p_data = get_full_pitching_data(pitcherid, suffix) if pitcherid else pd.DataFrame()
        
        current_game.update({
          'date': e.find_previous('main').get('data-gamedate'),
          'game_time': e.find_previous('div', attrs={'class':'lineup__time'}).get_text(strip=True),
          'team_h': e.find_previous('div', attrs={'class': 'is-home'}).next.strip(),
          'team_v': e.find_previous('div', attrs={'class': 'is-visit'}).next.strip(),
          f'starting_pitcher_name{suffix}': e.a.get_text(strip=True),
          f'starting_pitcher_id{suffix}': pitcherid,
        })
        
        if not p_data.empty:
          p_data_dict = p_data.to_dict(orient="records")[0]  # Convert first row to dict
          current_game.update({f'Strt_{k}{suffix}': v for k, v in p_data_dict.items()})

    elif e.get('class') and 'lineup__player' in e.get('class'):
      if e.a is not None:
        # Batter Data
        name = e.a.get('title').split(' ')
        f_name, l_name = name[0], name[-1]
        batterid = playerid_lookup(l_name, f_name).get('key_retro')
        suffix = "_h" if team_type == "is-home" else "_v"

        # Check if batterid lookup returned a value
        batterid = batterid.iloc[0] if isinstance(batterid, pd.Series) and not batterid.empty else ''
        b_data = get_full_batting_data(batterid, order_count, suffix) if batterid else pd.DataFrame()

        current_game.update({
          f'batter{order_count}_name{suffix}': e.a.get_text(strip=True),
          f'batter{order_count}_id{suffix}': batterid,
          f'batter{order_count}_pos{suffix}': e.div.get_text(strip=True),
        })
        
        if not b_data.empty:
          b_data_dict = b_data.to_dict(orient="records")[0]  # Convert first row to dict
          current_game.update({f'batter{order_count}_{k}{suffix}': v for k, v in b_data_dict.items()})

        order_count += 1

  if current_game:
    all_data.append(current_game)  # Add last processed game

  # Convert to DataFrame
  final_df = pd.DataFrame(all_data)
  
  final_df['game_id'] = final_df['date'] + ' ' + final_df['game_time']
  
  # Group by 'game_id' and aggregate data
  merged_df = final_df.groupby('game_id').agg(agg_non_na).reset_index(drop=True)
  
  # Ensure 'date' column is in datetime format
  merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

  # Merge with the existing `df` based on common columns
  return merged_df

def agg_non_na(series):
  return series.dropna().iloc[0] if not series.dropna().empty else None

def predict_winner(X):
  # TODO: pull from Google Drive?
  with open('models/win_model.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
  return model.predict(X)

def predict_runs_scored(X):
  # TODO: pull from Google Drive?
  with open('models/runs_scored_model.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
  return model.predict(X)

def post_to_X():
  client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    access_token=ACCESS_KEY,
    access_token_secret=ACCESS_SECRET,
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
  )

  tweet = f"""xxx"""

  post_result = client.create_tweet(text=tweet)
  return 'Tweet Posted to @MoneyballVo!'

def print_todays_slate(df):
  print(f'\n{df.iloc[:, :4]}')

def lambda_handler(event, context):
  print('--- TIME TO COOK 👨🏻‍🍳 ⚾️ 🚀 💰 ---')

  RUN_DATE = date.today() if TOMORROW_GAMES == 0 else date.today() + timedelta(days=1)
  print(f'\nGetting Starting Lineups for {RUN_DATE}')
  df = get_lineups()
  
  print_todays_slate(df)
  
  print(f'\nSaving Player DataFrame to CSV')
  df.to_csv(f'{RUN_DATE}_player_data.csv')
  
  #TODO: Add 10/35/75/162 Rolling Windows Data for Pitchers & Batters

  #TODO: Add Odds Data

  #df = pd.read_csv('2025-03-25_player_data.csv')
  
  print(f'\nMaking Predictions')
  #df['predict_runs_scored'] = df.apply(predict_runs_scored, axis=1)
  #df['predict_home_win'] = df.apply(predict_winner, axis=1)
  
  #print(df.head(10))
  
  print(f'\nPosting picks to X')
  #post_to_X()
  return {}

if __name__ == "__main__":
  lambda_handler({}, {})