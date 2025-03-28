from pybaseball import team_game_logs, season_game_logs
import pandas as pd
import numpy as np

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
YEARS = list(range(2024, 2026))

def get_game_data_by_team(team, stat_type):
  game_df = pd.DataFrame()
  for year in YEARS:
    temp_df = pd.DataFrame()
    try:
      temp_df = team_game_logs(year, team, stat_type)
      temp_df['Team'] = team
      temp_df['Season'] = year
      game_df = pd.concat((game_df, temp_df))
      print(game_df)
      print(f'Game {stat_type} data loaded for {team} in {year}')
    except RuntimeError as error:
      print(f'Unable to load game {stat_type} data for {team} in {year}')
  return game_df

def get_all_teams_data():
  for team in TEAMS:
    for stat_type in ['pitching', 'batting']:
      game_df = get_game_data_by_team(team, stat_type)
      game_df.to_csv(f'data/raw/{team}_{stat_type}.csv')
      print(f'**Game {stat_type} data written to csv for {team}**')
      
def get_prev_years_data():
  event_df = pd.DataFrame()
  for year in YEARS:
    try:
      temp_df = season_game_logs(year)
      temp_df['season'] = year
      event_df = pd.concat((event_df, temp_df))
      print(f'Event data loaded for {year}')
    except ValueError as error:
      print(error)

  event_df.to_csv('data/raw/retrosheet_events.csv')
  
# strip away suffix, e.g., '_h', '_v', for given column
def strip_suffix(col, suffix):
  return col[:-len(suffix)] if col.endswith(suffix) else col

def get_team_cols(df):
  visiting_cols = [col for col in df.columns if not col.endswith('_h')]
  visiting_cols_stripped = [strip_suffix(col, '_v') for col in visiting_cols]
  home_cols = [col for col in df.columns if not col.endswith('_v')]
  home_cols_stripped = [strip_suffix(col, '_h') for col in home_cols]
  return home_cols, home_cols_stripped, visiting_cols, visiting_cols_stripped

# create team dataframe to easily aggregate rolling window game data
def create_team_df(df, team):
  home_cols, home_cols_stripped, visiting_cols, visiting_cols_stripped = get_team_cols(df)
  df_team_v = df[(df.team_v == team)]
  opp = df_team_v['team_h']
  df_team_v = df_team_v[visiting_cols]
  df_team_v.columns = visiting_cols_stripped
  df_team_v['home_game'] = 0
  df_team_v['opponent'] = opp
  
  df_team_h = df[(df.team_h == team)]
  opp = df_team_h['team_v']
  df_team_h = df_team_h[home_cols]
  df_team_h.columns = home_cols_stripped
  df_team_h['home_game'] = 1
  df_team_h['opponent'] = opp
  
  cols = ['AB', 'H', 'x2B', 'x3B', 'HR', 'BB', 'runs', 'SB', 'CS', 'ERR']
  
  df_team = pd.concat((df_team_h, df_team_v))
  df_team.sort_values(['date_dblhead'], inplace=True)

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
      
  df_team.set_index('date_dblhead', inplace=True)
  return df_team

def generate_team_window_features(df):
  team_data_dict = {}
  teams = pd.concat([df['team_h'], df['team_v']]).unique()
  for team in teams:
    team_data_dict[team] = create_team_df(df, team)
  ## Create a variety of summarized statistics for each game
  ## For each game, we look up the home and visiting team in the team
  ## data dictionary, and then look up the game, and pull the relevant stats

  stats = ['BATAVG', 'OBP', 'SLG', 'OBS', 'SB', 'CS', 'ERR']
  teams = ['h', 'v']

  # Initialize arrays
  arrays = {f"{stat}_{window}_{team}": np.zeros(df.shape[0]) for stat in stats for window in WINDOWS for team in teams}

  # Populate the arrays
  for i, (index, row) in enumerate(df.iterrows()):
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
