#!/usr/bin/env python3

import os
import warnings
import pandas as pd
from datetime import date, timedelta
import pickle

from api.teams import get_all_teams_data, get_prev_years_data, generate_team_window_features
from api.lineups import get_lineups, get_run_total_feats

import tweepy

from dotenv import load_dotenv
load_dotenv()

# Silence all performance warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", category=UserWarning)

# X essentials
ACCESS_KEY = os.environ['X_ACCESS_KEY']
ACCESS_SECRET = os.environ['X_ACCESS_SECRET']
CONSUMER_KEY = os.environ['X_CONSUMER_KEY']
CONSUMER_SECRET = os.environ['X_CONSUMER_SECRET']
BEARER_TOKEN = os.environ['X_BEARER_TOKEN']

# GitHub Token
GH_TOKEN = os.environ['GH_TOKEN']

# Flags for Settings
TOMORROW_GAMES = int(os.environ['TOMORROW_GAMES'])
REFRESH_DATA = int(os.environ['REFRESH_DATA'])

# Game Windows for Prev Data Lookup
WINDOWS = [162, 90, 30]
YEARS = list(range(2024, 2026))

# Set of features we will predict on
RUNS_SCORED_FEAT_SET = [
  'OBP_162',
  'SLG_162',
  'Strt_WHIP_35',
  'Strt_TB_BB_perc_35',
  'Strt_H_BB_perc_35',
  'Strt_SO_perc_10',
  'Bpen_WHIP_75',
  'Bpen_TB_BB_perc_75',
  'Bpen_SO_perc_75',
  'Bpen_TB_BB_perc_35',
  'lineup8_OBP_162',
  'lineup8_SLG_162',
  'lineup9_OBP_162',
  'lineup9_SLG_162',
  'home_hitting',
  'Bpen_H_BB_perc_75',
  'Bpen_WHIP_35',
  'Bpen_H_BB_perc_35',
  'Bpen_SO_perc_35',
  'Bpen_WHIP_10',
  'Bpen_TB_BB_perc_10',
  'Bpen_H_BB_perc_10',
  'Bpen_SO_perc_10'
]

HOME_VICTORY_FEAT_SET = [
  'OBP_162_h','OBP_162_v',
  'SLG_162_h','SLG_162_v',
  'Strt_WHIP_35_h','Strt_WHIP_35_v',
  'Strt_TB_BB_perc_35_h', 'Strt_TB_BB_perc_35_v',
  'Strt_H_BB_perc_35_h', 'Strt_H_BB_perc_35_v',
  'Strt_SO_perc_10_h', 'Strt_SO_perc_10_v',
  'Bpen_WHIP_75_h','Bpen_WHIP_75_v',
  'Bpen_TB_BB_perc_75_h', 'Bpen_TB_BB_perc_75_v',
  'Bpen_H_BB_perc_75_h', 'Bpen_H_BB_perc_75_v',
  'Bpen_SO_perc_75_h', 'Bpen_SO_perc_75_v',
  'Bpen_WHIP_35_h','Bpen_WHIP_35_v',
  'Bpen_TB_BB_perc_35_h', 'Bpen_TB_BB_perc_35_v',
  'Bpen_H_BB_perc_35_h', 'Bpen_H_BB_perc_35_v',
  'Bpen_SO_perc_35_h', 'Bpen_SO_perc_35_v',
  'Bpen_WHIP_10_h','Bpen_WHIP_10_v',
  'Bpen_TB_BB_perc_10_h', 'Bpen_TB_BB_perc_10_v',
  'Bpen_H_BB_perc_10_h', 'Bpen_H_BB_perc_10_v',
  'Bpen_SO_perc_10_h', 'Bpen_SO_perc_10_v',
  'lineup9_OBP_350_h','lineup9_OBP_350_v',
  'lineup9_SLG_350_h','lineup9_SLG_350_v',
  'lineup9_OBP_162_h','lineup9_OBP_162_v',
  'lineup9_SLG_162_h','lineup9_SLG_162_v',
  'lineup9_OBP_75_h','lineup9_OBP_75_v',
  'lineup9_SLG_75_h','lineup9_SLG_75_v'
]

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
  print(f"\n{df.loc[:, ['date', 'game_time', 'team_h', 'team_v']]}")

def lambda_handler(event, context):
  print('--- TIME TO COOK 👨🏻‍🍳 ⚾️ 🚀 💰 ---')

  RUN_DATE = date.today() if TOMORROW_GAMES == 0 else date.today() + timedelta(days=1)

  if REFRESH_DATA == 1:
    print('\nGetting All Teams Data')
    get_all_teams_data()
    get_prev_years_data()
  
  print(f'\nGetting Starting Lineups for {RUN_DATE}')
  df = get_lineups()
  #df = pd.read_csv('data/daily/2025-03-27_player_data.csv')
  
  print_todays_slate(df)
  
  print(f'\nGenerating Team Window Features')
  df = generate_team_window_features(df)
  
  print(f'\nGetting Features for Run Total Predictions')
  df_runs = get_run_total_feats(df)
  
  print(f'\nGetting Odds Data')
  df['odds'] = None
  df_runs['odds'] = None
  df_runs['over_under_line'] = None
  #df['odds'] = get_odds(df.loc[:, ['team_h', 'date'])
  #df_runs['odds'] = get_odds(df.loc[:, ['team_h', 'date'])
  
  print(f'\nMaking Predictions')
  df_runs['run_total'] = None
  df['home_victory'] = None
  
  df_runs['run_total'] = predict_runs_scored(df_runs.loc[:, RUNS_SCORED_FEAT_SET])
  df['home_victory'] = predict_winner(df.loc[:, HOME_VICTORY_FEAT_SET])
  
  print(f'\nSaving Predictions DataFrames to CSV')
  df.loc[:, ['date', 'game_time', 'team_h', 'team_v', 'odds', 'home_victory']].to_csv(f'data/daily/{RUN_DATE}_home_victory_preds.csv')
  df_runs.loc[:, ['date', 'game_time', 'team_h', 'team_v', 'odds', 'over_under_line', 'run_total']].to_csv(f'data/daily/{RUN_DATE}_run_total_preds.csv')
  
  print(f'\nPosting picks to X')
  #post_to_X()
  return {}

if __name__ == "__main__":
  lambda_handler({}, {})