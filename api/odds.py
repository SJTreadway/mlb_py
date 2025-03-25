import pandas as pd
import numpy as np

SEASON = 2025
ODDSSHARK_TEAM_ID_MAP = {
  26995 + i: team for i, team in enumerate([
    'PHI', 'SDN', 'SFN', 'ANA', 'DET', 'CIN', 'NYA', 'TEX', 'TBA', 'COL',
    'MIN', 'KCA', 'ARI', 'BAL', 'ATL', 'TOR', 'SEA', 'MIL', 'PIT', 'NYN',
    'LAN', 'OAK', 'WAS', 'CHA', 'SLN', 'CHN', 'BOS', 'MIA', 'HOU', 'CLE'
  ])
}

def line_to_prob(line):
  prob_underdog = 100/(np.abs(line)+100) # this is the probability for the underdog
  add_term = ((1-np.sign(line))/2) # 0 if negative, 1 if positive
  mult_factor = np.sign(line) # -1 if negative, 1 if positive
  # if line is positive, team is underdog, give 0 + 1*prob_underdog
  # if line is negative, team is favorites, give 1 + (-1)*prob_underdog
  imp_prob = add_term + mult_factor * prob_underdog 
  return imp_prob

def get_key_for_team(team):
  return next((key for key, value in ODDSSHARK_TEAM_ID_MAP.items() if value == team), None)

def get_odds(team_h, team_v, date):
  team_h_key = get_key_for_team(team_h)
  team_v_key = get_key_for_team(team_v)
  df = pd.DataFrame()
  for i in [team_h_key, team_v_key]:
    team_name = ODDSSHARK_TEAM_ID_MAP[i]
    print(team_name)
    url = 'https://www.oddsshark.com/stats/gamelog/baseball/mlb/'+str(i)+'?season='+str(SEASON)
    df_temp = pd.read_html(url)[0]
    df_temp = df_temp[(df_temp.Game == 'REG') & (df_temp.Date == date)]
    print(df_temp.shape)
    df_temp['team_source'] = team_name
    df_temp['season'] = SEASON
    df_temp['date_numeric'] = pd.to_datetime(df_temp.Date).astype(str).str.replace('-','')
    df_temp['game_no'] = np.arange(1,df_temp.shape[0]+1)
    df_temp['prob_implied'] = line_to_prob(df_temp['Line'])      
    next_game_date = np.concatenate((df_temp['date_numeric'].iloc[1:],[0]))
    previous_game_date = np.concatenate(([0], df_temp['date_numeric'].iloc[:-1]))
    game_1_dblheader = (df_temp.date_numeric.to_numpy()==next_game_date).astype(int)
    game_2_dblheader = (df_temp.date_numeric.to_numpy()==previous_game_date).astype(int)*2
    df_temp['dblheader_num'] = game_1_dblheader+game_2_dblheader        
    df = df.merge(df_temp)
  return df