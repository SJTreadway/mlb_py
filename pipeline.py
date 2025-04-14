#!/usr/bin/env python3

import os
import math
import warnings
# Silence all performance warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# Set options to display all columns and adjust the width
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', 0)  # Automatically adjust to terminal width

from datetime import date, timedelta
import pickle

from api.teams import generate_team_window_features
from api.lineups import get_lineups, get_run_total_feats
from api.odds import get_over_odds, get_under_odds, get_total_line, get_money_line_price, line_to_bet, calculate_edge
from api.pitchers import process_pitching_data
from api.batters import process_batting_data

import tweepy

from dotenv import load_dotenv
load_dotenv()


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

# Causes data pull even if file exists
REFRESH_DATA = int(os.environ['REFRESH_DATA'])

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
  with open('models/win_model_v1.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
  pred = model.predict(X)
  prob = model.predict_proba(X)[:, 1]
  return pred, prob

def predict_runs_scored(X):
  with open('models/runs_scored_model_v1.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
  probs = model.predict_proba(X)
  return probs

def get_runs_scored_prob(probs, line):
  line = pd.to_numeric(line, errors='coerce')
  return round(probs[math.ceil(line):].sum()) if not pd.isna(line) else None

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

def filter_games_by_edge(df):
  filtered_df = df.copy()
  filtered_df['edge_h'] = filtered_df['edge_h'].str.replace('%', '').astype(float)
  filtered_df['edge_v'] = filtered_df['edge_v'].str.replace('%', '').astype(float)
  filtered_df['prob'] = filtered_df['prob'].astype(float)
  filtered_df = filtered_df[
      ((filtered_df['edge_h'] > 4.0) & (filtered_df['prob'] > 0.50)) |
      ((filtered_df['edge_v'] > 4.0) & ((1 - filtered_df['prob']) > 0.50))
  ]
  return filtered_df

def print_todays_home_victory_preds(df):
  filtered_df = filter_games_by_edge(df)
  filtered_df = filtered_df.rename(columns={
    'date_dblhead': 'Date',
    'game_time': 'Time',
    'team_h_full': 'Home',
    'team_v_full': 'Visitor',
    'starting_pitcher_name_h': 'Probable Starter (H)',
    'starting_pitcher_name_v': 'Probable Starter (V)',
    'moneyline_h': 'ML (H)',
    'prob': 'Prob Win (H)',
    'edge_h': 'Edge (H)',
    'moneyline_value_line_h': 'Line to Bet (H)',
    'moneyline_v': 'ML (V)',
    'moneyline_value_line_v': 'Line to Bet (V)',
    'edge_v': 'Edge (V)'
  })
  filtered_df.sort_values('Date', ascending=True, inplace=True)
  cols = [
    'Date', 'Time', 'Visitor', 'Probable Starter (V)', 
    'Home', 'Probable Starter (H)', 'ML (H)', 'Line to Bet (H)',
    'Edge (H)', 'Prob Win (H)', 'ML (V)', 'Line to Bet (V)', 'Edge (V)'
  ]
  print(f"\n{filtered_df.loc[:,cols]}")
  
def print_todays_totals_preds(df):
  df = df.rename(columns={
    'date_dblhead': 'Date',
    'game_time': 'Time',
    'team_h_full': 'Home',
    'team_v_full': 'Visitor',
    'over_under_line': 'O/U Line',
    'over_under_price_o': 'Over Price',
    'over_under_price_u': 'Under Price',
    'total_runs_predicted': 'Total Runs Predicted'
  })
  df.sort_values('Date', ascending=True, inplace=True)
  cols = [
    'Date', 'Time', 'Visitor', 'Home', 'O/U Line',
    'Over Price', 'Under Price', 'Total Runs Predicted'
  ]
  print(f"\n{df.loc[:,cols]}")

def lambda_handler(event, context):
  print('--- TIME TO COOK 👨🏻‍🍳 ⚾️ 🚀 💰 ---')

  RUN_DATE = date.today() if TOMORROW_GAMES == 0 else date.today() + timedelta(days=1)

  print(f'\nGetting Starting Lineups for {RUN_DATE}')
  fname = f'data/daily/{RUN_DATE}_lineup_data.csv'
  if os.path.exists(fname) and REFRESH_DATA != 1:
    print(f'\nLoading Data From File: {fname}')
    lineup_w_pitching_batting_df = pd.read_csv(fname, index_col=False)
  else:
    print('\nLoading Lineup Data')
    df = get_lineups()

    # Get/Store Starting Pitching Data to Files
    print('\nLoading Pitching Data')
    lineup_w_pitching_df = process_pitching_data(df)
    
    # Add Batting Data
    print('\nLoading Batting Data')
    lineup_w_pitching_batting_df = process_batting_data(lineup_w_pitching_df)
    
    print(f'\nSaving Lineup Data to CSV')
    lineup_w_pitching_batting_df.to_csv(fname, index=False)
  
  print(f'\nLoading Team Data')
  lineup_w_pitching_batting_team_df = generate_team_window_features(lineup_w_pitching_batting_df)
  
  print(f'\nGetting Features for Run Total Predictions')
  df_runs = get_run_total_feats(lineup_w_pitching_batting_team_df)
  df_runs.drop_duplicates(subset=['date_dblhead', 'team_h', 'team_v'], inplace=True)
  df_runs.reset_index(drop=True, inplace=True)
  
  print(f'\nGetting Odds Data')
  lineup_w_pitching_batting_team_df['moneyline_h'] = lineup_w_pitching_batting_team_df.apply(lambda row: get_money_line_price(row['team_h_full']), axis=1)
  lineup_w_pitching_batting_team_df['moneyline_v'] = lineup_w_pitching_batting_team_df.apply(lambda row: get_money_line_price(row['team_v_full']), axis=1)

  df_runs['over_under_price_o'] = df_runs.apply(lambda row: get_over_odds(row['team_h_full']), axis=1)
  df_runs['over_under_price_u'] = df_runs.apply(lambda row: get_under_odds(row['team_h_full']), axis=1)
  df_runs['over_under_line'] = df_runs.apply(lambda row: get_total_line(row['team_h_full']), axis=1)

  print(f'\nMaking Predictions')
  lineup_w_pitching_batting_team_df['home_victory'], lineup_w_pitching_batting_team_df['prob'] = predict_winner(lineup_w_pitching_batting_df.loc[:, HOME_VICTORY_FEAT_SET])
  lineup_w_pitching_batting_team_df['moneyline_value_line_h'] = lineup_w_pitching_batting_team_df.apply(lambda row: line_to_bet(row['prob']), axis=1)
  lineup_w_pitching_batting_team_df['moneyline_value_line_v'] = lineup_w_pitching_batting_team_df.apply(lambda row: line_to_bet(1 - row['prob']), axis=1)

  # calculate our edge
  lineup_w_pitching_batting_team_df['edge_h'] = lineup_w_pitching_batting_team_df.apply(lambda row: calculate_edge(row['prob'], row['moneyline_h']), axis=1)
  lineup_w_pitching_batting_team_df['edge_v'] = lineup_w_pitching_batting_team_df.apply(lambda row: calculate_edge(1-row['prob'], row['moneyline_v']), axis=1)

  run_total_probs = predict_runs_scored(df_runs.loc[:, RUNS_SCORED_FEAT_SET])
  df_runs['total_runs_predicted'] = df_runs.apply(lambda row: get_runs_scored_prob(run_total_probs, row['over_under_line']), axis=1)

  lineup_w_pitching_batting_team_df.reset_index(drop=True, inplace=True)
  print_todays_home_victory_preds(lineup_w_pitching_batting_team_df)
  
  print_todays_totals_preds(df_runs)
  
  #print(f'\nHOME VICTORY FEATS:\n{lineup_w_pitching_batting_team_df.loc[:, HOME_VICTORY_FEAT_SET]}')
  #print(f'\nRUNS SCORED FEATS:\n{df_runs.loc[:, RUNS_SCORED_FEAT_SET]}')
  
  print(f'\nSaving Predictions DataFrames to CSV')
  lineup_w_pitching_batting_team_df.loc[:, ['date_dblhead', 'game_time', 'team_h_full', 'team_v_full', 'prob', 'moneyline_h', 'moneyline_value_line_h', 'edge_h', 'moneyline_v', 'moneyline_value_line_v', 'edge_v']].to_csv(f'data/results/{RUN_DATE}_home_victory_preds.csv', index=False)
  df_runs.loc[:, ['date_dblhead', 'game_time', 'team_h_full', 'team_v_full', 'over_under_line', 'total_runs_predicted']].to_csv(f'data/results/{RUN_DATE}_run_total_preds.csv', index=False)
  
  print(f'\nPosting picks to X')
  #post_to_X()
  return {}

if __name__ == "__main__":
  lambda_handler({}, {})