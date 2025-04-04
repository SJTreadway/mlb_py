from pybaseball import team_game_logs, season_game_logs
import pandas as pd
import numpy as np

from helpers import strip_suffix, agg_non_na

from tqdm import tqdm

pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)
              
TEAMS = [
  "LAA", "MIL", "HOU", "BAL",
  "BOS", "CHW", "CLE", "DET", 
  "KCR", "MIN", "NYY", "OAK", 
  "SEA", "TBR", "TEX", "TOR", 
  "ARI", "ATL", "CIN", "COL",
  "SDP", "MIA", "NYM", "PHI",
  "PIT", "SFG", "STL", "WSN",
  "LAD", "CHC"
]

# Game Windows for Prev Data Lookup
WINDOWS = [162, 90, 30]

def get_team_cols(df):
  visiting_cols = [col for col in df.columns if not col.endswith('_h')]
  visiting_cols_stripped = [strip_suffix(col, '_v') for col in visiting_cols]
  home_cols = [col for col in df.columns if not col.endswith('_v')]
  home_cols_stripped = [strip_suffix(col, '_h') for col in home_cols]
  return home_cols, home_cols_stripped, visiting_cols, visiting_cols_stripped

# create team dataframe to easily aggregate rolling window game data
def create_team_df(df, team):
  cols = ['AB', 'H', 'x2B', 'x3B', 'HR', 'BB', 'SB', 'CS']
  for c in cols:
    df[c] = np.nan

  cols_w_idx = cols + ['date_dblhead']

  df_team_v = df[(df.team_v == team)]
  opp = df_team_v['team_h']
  df_team_v = df_team_v[cols_w_idx]
  df_team_v['home_game'] = 0
  df_team_v['opponent'] = opp
  
  df_team_h = df[(df.team_h == team)]
  opp = df_team_h['team_v']
  df_team_h = df_team_h[cols_w_idx]
  df_team_h['home_game'] = 1
  df_team_h['opponent'] = opp
  
  df_team = df_team_h if not df_team_h.empty else df_team_v
  
  df_team = df_team.set_index('date_dblhead')

  # Create Empty Cols before inserting data
  df_team = df_team.assign(**{col: None for col in cols})
  
  for winsize in WINDOWS:
    suff = str(winsize)
    for raw_col in cols:
      new_col = 'rollsum_' + raw_col + '_' + suff
      df_team[new_col] = df_team[raw_col].rolling(winsize, closed='left').sum()
    
    df_team['rollsum_BATAVG_' + suff] = df_team['rollsum_H_' + suff] / df_team['rollsum_AB_' + suff]
    df_team['rollsum_OBP_' + suff] = (df_team['rollsum_H_' + suff] + df_team['rollsum_BB_' + suff]) / (
        df_team['rollsum_BB_' + suff] + df_team['rollsum_AB_' + suff])
    df_team['rollsum_SLG_' + suff] = (df_team['rollsum_H_' + suff] + df_team['rollsum_x2B_' + suff] + 
                                      2 * df_team['rollsum_x3B_' + suff] + 3 * df_team['rollsum_HR_' + suff]) / (
                                      df_team['rollsum_AB_' + suff])
    df_team['rollsum_OBS_' + suff] = df_team['rollsum_OBP_' + suff] + df_team['rollsum_SLG_' + suff]
      
  return df_team

def generate_team_window_features(df):
  team_data_dict = {}
  teams = df[['team_h', 'team_v']].stack().unique().tolist()
  for team in teams:
    team_data_dict[team] = create_team_df(df, team)
  ## Create a variety of summarized statistics for each game
  ## For each game, we look up the home and visiting team in the team
  ## data dictionary, and then look up the game, and pull the relevant stats
  
  stats = ['BATAVG', 'OBP', 'SLG', 'OBS', 'SB', 'CS']
  teams = ['h', 'v']

  # Initialize arrays
  arrays = {f"{stat}_{window}_{team}": np.zeros(df.shape[0]) for stat in stats for window in WINDOWS for team in teams}

  # Populate the arrays
  for i, (index, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
    home_team = row['team_h']
    visit_team = row['team_v']
    game_index = row['date_dblhead']
    
    for window in WINDOWS:
      for stat in stats:
        arrays[f'{stat}_{window}_h'][i] = team_data_dict[home_team].loc[game_index, f'rollsum_{stat}_{window}']
        arrays[f'{stat}_{window}_v'][i] = team_data_dict[visit_team].loc[game_index, f'rollsum_{stat}_{window}']

  # Add arrays to DataFrame
  for key, value in arrays.items():
    df[key] = value

  return df
