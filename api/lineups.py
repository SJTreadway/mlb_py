#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
import os
from tqdm import tqdm

from pybaseball import playerid_lookup
import statsapi

from bs4 import BeautifulSoup

from api.pitchers import get_full_pitching_data
from api.batters import get_full_batting_data

from helpers import agg_non_na

from dotenv import load_dotenv
load_dotenv()

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

  for e in tqdm(soup.select('.lineup__box ul li')):
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
        pitcher_df = playerid_lookup(l_name, f_name)
        pitcher_df= pitcher_df.sort_values(by="mlb_played_last", ascending=False)
        pitcherid = pitcher_df.get('key_retro')
        suffix = "_h" if team_type == "is-home" else "_v"
        
        date = e.find_previous('main').get('data-gamedate')
        game_time = e.find_previous('div', attrs={'class':'lineup__time'}).get_text(strip=True)
        fmt_date = date.replace('-', '')
        fmt_game_time = game_time.split(' ')[0].replace(':', '')
        date_dblhead = int(fmt_date + fmt_game_time)
        
        team_h = e.find_previous('div', attrs={'class': 'lineup__team is-home'}).find_next('div', attrs={'class': 'lineup__abbr'}).get_text(strip=True)
        team_v = e.find_previous('div', attrs={'class': 'lineup__team is-visit'}).find_next('div', attrs={'class': 'lineup__abbr'}).get_text(strip=True)

        # _full is needed for statsapi team lookup
        team_h_full = e.find_previous('div', attrs={'class': 'lineup__mteam is-home'}).next.get_text(strip=True)
        team_v_full = e.find_previous('div', attrs={'class': 'lineup__mteam is-visit'}).next.get_text(strip=True)
        
        team_h_id = statsapi.lookup_team(team_h_full)[0]['id']
        team_v_id = statsapi.lookup_team(team_v_full)[0]['id']
        team_h_stats = statsapi.get('team_stats', {'teamId': team_h_id, 'season': 2024, 'group': 'hitting', 'stats': 'season', 'sitCodes': 'l10'})['stats'][0]['splits'][0]['stat']
        team_v_stats = statsapi.get('team_stats', {'teamId': team_v_id, 'season': 2024, 'group': 'hitting', 'stats': 'season', 'sitCodes': 'l10'})['stats'][0]['splits'][0]['stat']
        
        gp_h = team_h_stats['gamesPlayed']
        gp_v = team_v_stats['gamesPlayed']
        
        # Check if pitcherid lookup returned a value
        pitcherid = pitcherid.iloc[0] if isinstance(pitcherid, pd.Series) and not pitcherid.empty else ''
        #p_data = get_full_pitching_data(pitcherid) if pitcherid else pd.DataFrame()
        
        current_game.update({
          #'date': date,
          'date_dblhead': date_dblhead,
          'game_time': game_time,
          'team_h': team_h,
          'team_h_full': team_h_full,
          'team_v': team_v,
          'team_v_full': team_v_full,
          # home stats
          'AB_h': team_h_stats['atBats'] / gp_h,
          'BB_h': team_h_stats['baseOnBalls'] / gp_h,
          'H_h': team_h_stats['hits'] / gp_h,
          'R_h': team_h_stats['runs'] / gp_h,
          'x2B_h': team_h_stats['doubles'] / gp_h,
          'x3B_h': team_h_stats['triples'] / gp_h,
          'HR_h': team_h_stats['homeRuns'] / gp_h,
          'HBP_h': team_h_stats['hitByPitch'] / gp_h,
          'SO_h': team_h_stats['strikeOuts'] / gp_h,
          'SB_h': team_h_stats['stolenBases'] / gp_h,
          'CS_h': team_h_stats['caughtStealing'] / gp_h,
          # visitor stats
          'AB_v': team_v_stats['atBats'] / gp_v,
          'BB_v': team_v_stats['baseOnBalls'] / gp_v,
          'H_v': team_v_stats['hits'] / gp_v,
          'R_v': team_v_stats['runs'] / gp_v,
          'x2B_v': team_v_stats['doubles'] / gp_v,
          'x3B_v': team_v_stats['triples'] / gp_v,
          'HR_v': team_v_stats['homeRuns'] / gp_v,
          'HBP_v': team_v_stats['hitByPitch'] / gp_v,
          'SO_v': team_v_stats['strikeOuts'] / gp_v,
          'SB_v': team_v_stats['stolenBases'] / gp_v,
          'CS_v': team_v_stats['caughtStealing'] / gp_v,
          f'starting_pitcher_name{suffix}': e.a.get_text(strip=True),
          f'starting_pitcher_id{suffix}': pitcherid,
        })
        
        '''
        if not p_data.empty:
          p_data_dict = p_data.to_dict(orient="records")[0]  # Convert first row to dict
          current_game.update(p_data_dict)
        '''

    elif e.get('class') and 'lineup__player' in e.get('class'):
      if e.a is not None:
        # Batter Data
        name = e.a.get('title').split(' ')
        f_name, l_name = name[0], name[-1]
        batterid = playerid_lookup(l_name, f_name).get('key_retro')
        suffix = "_h" if team_type == "is-home" else "_v"

        # Check if batterid lookup returned a value
        batterid = batterid.iloc[0] if isinstance(batterid, pd.Series) and not batterid.empty else ''
        #b_data = get_full_batting_data(batterid) if batterid else pd.DataFrame()

        current_game.update({
          f'batter{order_count}_name{suffix}': e.a.get_text(strip=True),
          f'batter{order_count}_id{suffix}': batterid,
          f'batter{order_count}_pos{suffix}': e.div.get_text(strip=True)
        })
        
        '''
        if b_data is not None and not b_data.empty:
          b_data_dict = b_data.to_dict(orient="records")[0]  # Convert first row to dict
          current_game.update(b_data_dict)
        '''

        order_count += 1

  if current_game:
    all_data.append(current_game)  # Add last processed game

  # Convert to DataFrame
  final_df = pd.DataFrame(all_data)
  
  final_df['game_id'] = final_df['date_dblhead'].astype(str) + final_df['team_h'] + final_df['team_v']

  # Group by 'game_id' and aggregate data
  merged_df = final_df.groupby('game_id').agg(agg_non_na)
  
  return merged_df

def get_run_total_feats(df):
  cols_ref = ['date_dblhead','game_time','team_h','team_h_full','team_v','team_v_full']

  team_hit_stems = ['BATAVG','OBP','SLG','OBS','SB','CS']
  lineup_hit_stems = ['BATAVG','OBP','SLG','OBS','SLGmod','SObat_perc']
  strt_pitch_stems = ['ERA','WHIP','SO_perc','H_BB_perc','TB_BB_perc','FIP','FIP_perc']
  bpen_pitch_stems = ['WHIP','SO_perc','H_BB_perc','TB_BB_perc']

  team_hit_winsizes = [30,90,162]
  lineup_hit_winsizes = [30,75,162,350]
  strt_pitch_winsizes = [10,35,75]
  bpen_pitch_winsizes = [10,35,75]
  team_hit_features_a = [x+'_'+str(winsize)+'_h' for winsize in team_hit_winsizes for x in team_hit_stems ]
  lineup_hit_features_a = ['lineup'+n89+'_'+x+'_'+str(winsize)+wornot+'_h' for winsize in lineup_hit_winsizes
                          for x in lineup_hit_stems for wornot in ['','_w'] for n89 in ['8','9']]
  start_pitch_features_a = ['Strt_'+x+'_'+str(winsize)+'_v' for winsize in strt_pitch_winsizes for x in strt_pitch_stems]
  bpen_pitch_features_a = ['Bpen_'+x+'_'+str(winsize)+'_v' for winsize in bpen_pitch_winsizes for x in bpen_pitch_stems]

  team_hit_features_b = [x+'_'+str(winsize)+'_v' for winsize in team_hit_winsizes for x in team_hit_stems ]
  lineup_hit_features_b = ['lineup'+n89+'_'+x+'_'+str(winsize)+wornot+'_v' for winsize in lineup_hit_winsizes
                          for x in lineup_hit_stems for wornot in ['','_w'] for n89 in ['8','9']]
  start_pitch_features_b = ['Strt_'+x+'_'+str(winsize)+'_h' for winsize in strt_pitch_winsizes for x in strt_pitch_stems]
  bpen_pitch_features_b = ['Bpen_'+x+'_'+str(winsize)+'_h' for winsize in bpen_pitch_winsizes for x in bpen_pitch_stems]

  cols_a = cols_ref + team_hit_features_a + lineup_hit_features_a + start_pitch_features_a + bpen_pitch_features_a
  df_a = df.loc[:,cols_a]
  df_a['home_hitting'] = 1

  cols_b = cols_ref + team_hit_features_b + lineup_hit_features_b + start_pitch_features_b + bpen_pitch_features_b
  df_b = df.loc[:,cols_b]
  df_b['home_hitting'] = 0

  stripped_feats  = [x[:-2] for x in team_hit_features_a + lineup_hit_features_a + start_pitch_features_a + bpen_pitch_features_a]

  final_col_list = cols_ref + stripped_feats + ['home_hitting']
  
  df_a.columns = final_col_list
  df_b.columns = final_col_list
  df_runs = pd.concat((df_a,df_b))
  
  df_runs.set_index('date_dblhead', inplace=True)
  df_runs.sort_values('date_dblhead', ascending=False, inplace=True)

  return df_runs
  
