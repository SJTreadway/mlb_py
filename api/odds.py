import pandas as pd
import numpy as np

SEASON = 2025
ODDSSHARK_TEAM_ID_MAP = {
  26995 + i: team for i, team in enumerate([
    'ARI', 'CHC', 'SEA', 'ATH', 'NYY', 'MIL', 'TOR', 'BAL', 'TEX',
    'BOS', 'WSH', 'PHI', 'CIN', 'SF', 'CWS', 'LAA', 'HOU', 'NYM',
    'KC', 'CLE', 'MIA', 'PIT', 'SD', 'ATL', 'STL', 'MIN', 'LAD',
    'DET', 
    'ANA', 'TBA', 'COL' ## TBD
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

def get_odds(team_h, date):
  team_h_key = get_key_for_team(team_h)
  print(team_h)
  url = 'https://www.oddsshark.com/stats/gamelog/baseball/mlb/'+str(team_h_key)+'?season='+str(SEASON)
  df_temp = pd.read_html(url)[0]
  df_temp = df_temp[(df_temp.Game == 'REG') & (df_temp.Date == date)]
  print(df_temp['Line'])
  return line_to_prob(df_temp['Line'])      