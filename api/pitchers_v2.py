import re
import time
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from pybaseball import playerid_reverse_lookup, statcast_pitcher

from helpers import roll_column, strip_suffix, get_team_league_map, safe_float, safe_int

YEAR = int(os.environ['YEAR'])
WINDOWS = [10,35,75]

def process_pitching_data(df, debug=False):
  """
  Process pitching data using pybaseball API instead of web scraping.
  
  Args:
    df: DataFrame with game data
    debug: If True, enable verbose debugging output for bullpen calculations
  
  Returns:
    DataFrame with pitching and bullpen features
  """
  start_pitchers_h = [p for p in df.starting_pitcher_id_h.unique() if p is not None]
  start_pitchers_v = [p for p in df.starting_pitcher_id_v.unique() if p is not None]
  start_pitchers_all = np.union1d(start_pitchers_h, start_pitchers_v)

  # step 1: get pitching data for all starting pitchers and store to csv
  load_pitching_data(start_pitchers_all)

  # step 2: load data from files and store into dataframe
  strt_pitch_df = get_rolling_pitching_feats(df, start_pitchers_all)
  
  # step 3: add bullpen features with optional debug mode
  return get_bullpen_data(strt_pitch_df, debug=debug)

def retro_to_mlbam(retro_id):
  """Convert retro ID to MLBAM ID for API queries."""
  if not retro_id or pd.isna(retro_id):
    return None
  try:
    rev_lkp = playerid_reverse_lookup([retro_id], key_type='retro')
    if rev_lkp is not None and not rev_lkp.empty:
      return int(rev_lkp.iloc[0]['key_mlbam'])
  except Exception as e:
    print(f'Error looking up retro ID {retro_id}: {e}')
  return None

def load_pitching_data(start_pitchers_all):
  """
  Load pitching data using pybaseball API (statcast_pitcher).
  Much faster than scraping Retrosheet and Baseball-Reference.
  """
  for p_id in tqdm(start_pitchers_all, desc='Loading pitcher data via API'):
    if p_id and not pd.isna(p_id):
      fname_out = 'data/pitch/pitching_data_'+p_id+'.csv'
      
      # Convert retro ID to MLBAM ID for API
      mlbam_id = retro_to_mlbam(p_id)
      if not mlbam_id:
        print(f'Skipping pitcher {p_id} - could not convert to MLBAM ID')
        continue
      
      # Fetch current season data via API
      start_date = f'{YEAR}-03-01'
      end_date = f'{YEAR}-11-30'
      
      try:
        df_season = statcast_pitcher(start_date, end_date, mlbam_id)
        if df_season.empty:
          print(f'No data found for pitcher {p_id} (MLBAM: {mlbam_id})')
          continue
          
        # Transform statcast data to match expected format
        df_season = transform_statcast_pitcher(df_season)
        
        if not os.path.exists(fname_out):
          # Get historical data using API (start from 2008 when Statcast began)
          df_historical = get_historical_pitching_data(mlbam_id)
          df_temp = pd.concat((df_historical, df_season))
        else:
          # Load existing data and concatenate
          df_existing = pd.read_csv(fname_out)
          df_temp = pd.concat((df_existing, df_season))
          # Remove duplicates
          df_temp = df_temp.drop_duplicates(subset=['date', 'dblhead_num'], keep='first')
        
        # Save the updated data
        df_temp.to_csv(fname_out, index=False)
        
        # Small delay to be nice to the API
        time.sleep(0.1)
        
      except Exception as e:
        print(f'Error fetching data for pitcher {p_id}: {e}')
        continue

def transform_statcast_pitcher(df):
  """
  Transform Statcast pitcher data to match the expected format from Retrosheet.
  """
  if df.empty:
    return pd.DataFrame()
  
  # Group by game to get game-level stats
  df['game_date'] = pd.to_datetime(df['game_date'])
  
  # Initialize list for aggregated data
  games = []
  
  # Group by game_date and game_pk to get game-level stats
  for (game_date, game_pk), group in df.groupby(['game_date', 'game_pk']):
    # Determine if home or away
    at_vs = 'VS' if group['home_team'].iloc[0] == group['pitcher'].iloc[0] else 'AT'
    opponent = group['away_team'].iloc[0] if at_vs == 'VS' else group['home_team'].iloc[0]
    
    # Determine if starter (GS) by checking if pitched in first inning
    gs = 1 if any(group['inning'] == 1) else 0
    
    # Calculate innings pitched (convert outs to innings)
    outs = len(group[group['description'].str.contains('out|field_out|strikeout|double_play|triple_play', case=False, na=False)])
    ip = outs / 3.0
    
    # Calculate hits, walks, etc.
    h = len(group[group['events'].isin(['single', 'double', 'triple', 'home_run'])])
    x2b = len(group[group['events'] == 'double'])
    x3b = len(group[group['events'] == 'triple'])
    hr = len(group[group['events'] == 'home_run'])
    bb = len(group[group['events'].isin(['walk', 'intent_walk'])])
    ibb = len(group[group['events'] == 'intent_walk'])
    so = len(group[group['events'] == 'strikeout'])
    hbp = len(group[group['events'] == 'hit_by_pitch'])
    
    # Batters faced
    bfp = len(group)
    
    # Runs and earned runs (approximate)
    r = len(group[group['score'] == True]) if 'score' in group.columns else 0
    er = r  # Simplification - earned runs approximated as runs
    
    # Sacrifice hits/flies
    sh = len(group[group['events'] == 'sac_bunt'])
    sf = len(group[group['events'] == 'sac_fly'])
    
    # Double plays
    gdp = len(group[group['events'] == 'grounded_into_double_play'])
    
    # Win/loss determination (simplified)
    w = 0
    l = 0
    
    games.append({
      'date': game_date.strftime('%-m-%-d-%Y'),
      'dblhead_num': '',
      'at_vs': at_vs,
      'Opponent': opponent,
      'League': get_team_league_map().get(opponent, ''),
      'GS': gs,
      'CG': 0,  # Complete games not easily available in statcast
      'SHO': 0,  # Shutouts not easily available
      'GF': 0,  # Games finished
      'SV': 0,  # Saves not available in statcast
      'IP': ip,
      'H': h,
      'BFP': bfp,
      'HR': hr,
      'R': r,
      'ER': er,
      'BB': bb,
      'IB': ibb,
      'SO': so,
      'SH': sh,
      'SF': sf,
      'WP': 0,  # Wild pitches not directly in statcast
      'HBP': hbp,
      'BK': 0,  # Balks not in statcast
      'x2B': x2b,
      'x3B': x3b,
      'GDP': gdp,
      'ROE': 0,
      'W': w,
      'L': l,
      'ERA': 0.0  # Will be calculated cumulatively
    })
  
  result_df = pd.DataFrame(games)
  
  # Calculate cumulative ERA
  if not result_df.empty:
    result_df = calculate_cumulative_pitching_stats(result_df)
  
  return result_df

def calculate_cumulative_pitching_stats(df):
  """Calculate cumulative pitching statistics (ERA)."""
  if df.empty:
    return df
  df = df.sort_values('date')
  df['cum_ER'] = df['ER'].cumsum()
  df['cum_IP'] = df['IP'].cumsum()
  
  # ERA = (ER / IP) * 9
  df['ERA'] = (df['cum_ER'] / df['cum_IP'].replace(0, np.nan)) * 9
  
  # Drop cumulative columns
  df = df.drop(['cum_ER', 'cum_IP'], axis=1)
  
  return df

def get_historical_pitching_data(mlbam_id):
  """
  Get historical pitching data from 2008 onwards (when Statcast began).
  """
  all_data = []
  
  # Fetch data from 2008 to current year - 1
  for year in range(max(2008, YEAR - 5), YEAR):
    try:
      start_date = f'{year}-03-01'
      end_date = f'{year}-11-30'
      df_year = statcast_pitcher(start_date, end_date, mlbam_id)
      if not df_year.empty:
        all_data.append(transform_statcast_pitcher(df_year))
      time.sleep(0.1)  # Be nice to the API
    except Exception as e:
      print(f'Error fetching data for year {year}: {e}')
      continue
  
  if all_data:
    return pd.concat(all_data, ignore_index=True)
  return pd.DataFrame()

def load_and_process_pitch_df(p_id, filepath=''):
  if not p_id:
    return pd.DataFrame()

  fname = filepath+'pitching_data_'+p_id+'.csv'
  pitch_df = pd.DataFrame()
  
  try:
    pitch_df = pd.read_csv(fname)
  except Exception as e:
    print(f'Error loading pitcher data for {p_id} from {fname}, i.e., DO NOT BET ON THIS MF: {e}')
    return pd.DataFrame()
  
  # Convert date, fix dblhead_num to be 0,1,2
  pitch_df.dropna(subset=['date'], inplace=True)
  pitch_df['date'] = (
      pd.to_datetime(pitch_df['date'], format='mixed', errors='coerce')
        .dt.strftime('%Y%m%d')
        .astype(int)
  )
  pitch_df.dblhead_num.fillna(0, inplace=True)
  pitch_df['dblhead_num'] = pitch_df['dblhead_num'].astype(int)
  
  # Convert IP to proper mathematical format
  pitch_df['IP_real'] = (pitch_df.IP - (pitch_df.IP % 1)) + (pitch_df.IP % 1) * (10/3)
  
  # Collect all new columns in a dictionary to avoid DataFrame fragmentation
  new_columns = {}
  
  cols_to_agg = ['IP_real', 'H','BFP', 'HR', 'R', 'ER', 'BB', 'IB', 'SO', 'SH', 'SF', 'WP', 'HBP', 'BK',
      'x2B', 'x3B']
  for winsize in WINDOWS:
    for raw_col in cols_to_agg:
      new_colname = 'rollsum_'+raw_col+'_'+str(winsize)        
      new_columns[new_colname] = roll_column(pitch_df, raw_col, winsize)
    
  er_per_ip_def = (5/9)
  h_bb_per_ip_def = 1.5
  h_bb_per_bf_def = .37
  so_per_bf_def = .2
  ip_per_game_def = 3
  bf_per_game_def = 12
  tb_bb_perc_def = .45
  fip_numer_per_ip_def = .124*13 + 1.5*3 - 2*.8
  fip_numer_per_bf_def = .03*13 + .37*3 - 2*.2

  for winsize in WINDOWS:
    hit_col = 'rollsum_H_'+str(winsize)
    bb_col = 'rollsum_BB_'+str(winsize)
    h_bb_col = 'H_BB_roll_'+str(winsize)
    double_col = 'rollsum_x2B_'+str(winsize)
    triple_col = 'rollsum_x3B_'+str(winsize)
    hr_col = 'rollsum_HR_'+str(winsize)
    xb_col = 'XB_roll_'+str(winsize)
    tb_col = 'TB_roll_'+str(winsize)
    so_col = 'rollsum_SO_'+str(winsize)
    so_mod_col = 'SO_mod_'+str(winsize)
    ip_col = 'rollsum_IP_real_'+str(winsize)
    ip_mod_col = 'IP_mod_'+str(winsize)
    er_col = 'rollsum_ER_'+str(winsize)
    er_mod_col = 'ER_mod_'+str(winsize)
    bf_col = 'rollsum_BFP_'+str(winsize)
    bf_mod_col = 'BF_mod_'+str(winsize)
    era_col = 'ERA_'+str(winsize)
    fip_col = 'FIP_'+str(winsize)
    fip_perc_col = 'FIP_perc_'+str(winsize)
    fip_numer_col = 'FIP_numer_'+str(winsize)
    fip_numer_mod_col = 'FIP_numer_mod_'+str(winsize)
    fip_numer_mod2_col = 'FIP_numer_mod2_'+str(winsize)
    whip_col = 'WHIP_'+str(winsize)
    so_perc_col = 'SO_perc_'+str(winsize)
    h_bb_perc_col = 'H_BB_perc_'+str(winsize)
    h_bb_mod_col = 'H_BB_mod_'+str(winsize)
    h_bb_mod2_col = 'H_BB_mod2_'+str(winsize)
    tb_bb_mod_col = 'TB_BB_mod_'+str(winsize)
    tb_bb_perc_col = 'TB_BB_perc_'+str(winsize)
    
    # Calculate values using the new_columns dictionary
    h_bb = new_columns[hit_col] + new_columns[bb_col]
    new_columns[h_bb_col] = h_bb
    xb = new_columns[double_col] + 2*new_columns[triple_col] + 3*new_columns[hr_col]
    new_columns[xb_col] = xb
    tb = new_columns[hit_col] + xb
    new_columns[tb_col] = tb
    ip_mod = np.maximum(new_columns[ip_col], winsize*ip_per_game_def)
    new_columns[ip_mod_col] = ip_mod
    bf_mod = np.maximum(new_columns[bf_col], winsize*bf_per_game_def)
    new_columns[bf_mod_col] = bf_mod
    er_mod = new_columns[er_col] + er_per_ip_def*(ip_mod-new_columns[ip_col])
    new_columns[er_mod_col] = er_mod
    fip_numer = 13*new_columns[hr_col] + 3*h_bb - 2*new_columns[so_col]
    new_columns[fip_numer_col] = fip_numer
    new_columns[fip_numer_mod_col] = fip_numer + fip_numer_per_ip_def*(ip_mod-new_columns[ip_col])
    new_columns[fip_numer_mod2_col] = fip_numer + fip_numer_per_bf_def*(bf_mod-new_columns[bf_col])
    h_bb_mod = h_bb + h_bb_per_ip_def*(ip_mod-new_columns[ip_col])
    new_columns[h_bb_mod_col] = h_bb_mod
    h_bb_mod2 = h_bb + h_bb_per_bf_def*(bf_mod-new_columns[bf_col])
    new_columns[h_bb_mod2_col] = h_bb_mod2
    so_mod = new_columns[so_col] + so_per_bf_def*(bf_mod-new_columns[bf_col])
    new_columns[so_mod_col] = so_mod
    tb_bb_mod = (tb + new_columns[bb_col]) + tb_bb_perc_def*(bf_mod-new_columns[bf_col])
    new_columns[tb_bb_mod_col] = tb_bb_mod
    new_columns[era_col] = (er_mod/ip_mod)*9
    new_columns[fip_col] = (new_columns[fip_numer_mod_col]/ip_mod)
    new_columns[fip_perc_col] = (new_columns[fip_numer_mod_col]/bf_mod)
    new_columns[whip_col] = h_bb_mod/ip_mod
    new_columns[so_perc_col] = so_mod/bf_mod
    new_columns[tb_bb_perc_col] = tb_bb_mod/bf_mod
    new_columns[h_bb_perc_col] = h_bb_mod2/bf_mod
  
  # Concatenate all new columns at once to avoid fragmentation
  if new_columns:
    new_df = pd.DataFrame(new_columns, index=pitch_df.index)
    pitch_df = pd.concat([pitch_df, new_df], axis=1)
  
  pitch_df['date_dblhead'] = (pitch_df['date'].astype(str) + pitch_df['dblhead_num'].astype(str)).astype(int)
  pitch_df.set_index('date_dblhead', inplace=True)
  return pitch_df 
    
def get_rolling_pitching_feats(df, start_pitchers_all):
  pitcher_data_dict = {}
  for p_id in start_pitchers_all:
    pitcher_data_dict[p_id] = load_and_process_pitch_df(p_id,'data/pitch/')

  raw_cols_to_add = ['GS',  'IP',
      'H', 'BFP', 'HR', 'R', 'ER', 'BB', 'IB', 'SO', 'SH', 'SF', 'WP',
      'HBP', 'BK', 'x2B', 'x3B', 'IP_real', 'rollsum_IP_real_10', 'rollsum_H_10',
      'rollsum_BFP_10', 'rollsum_HR_10', 'rollsum_R_10', 'rollsum_ER_10',
      'rollsum_BB_10', 'rollsum_IB_10', 'rollsum_SO_10', 'rollsum_SH_10',
      'rollsum_SF_10', 'rollsum_WP_10', 'rollsum_HBP_10',
      'rollsum_BK_10', 'rollsum_x2B_10', 'rollsum_x3B_10',
      'rollsum_IP_real_35', 'rollsum_H_35', 'rollsum_BFP_35',
      'rollsum_HR_35', 'rollsum_R_35', 'rollsum_ER_35', 'rollsum_BB_35',
      'rollsum_IB_35', 'rollsum_SO_35', 'rollsum_SH_35', 'rollsum_SF_35',
      'rollsum_WP_35', 'rollsum_HBP_35', 'rollsum_BK_35',
      'rollsum_x2B_35', 'rollsum_x3B_35', 'rollsum_IP_real_75',
      'rollsum_H_75', 'rollsum_BFP_75', 'rollsum_HR_75', 'rollsum_R_75',
      'rollsum_ER_75', 'rollsum_BB_75', 'rollsum_IB_75', 'rollsum_SO_75',
      'rollsum_SH_75', 'rollsum_SF_75', 'rollsum_WP_75',
      'rollsum_HBP_75', 'rollsum_BK_75', 'rollsum_x2B_75',
      'rollsum_x3B_75', 'H_BB_roll_10', 'XB_roll_10', 'TB_roll_10',
      'IP_mod_10', 'BF_mod_10', 'ER_mod_10', 'FIP_numer_10',
      'FIP_numer_mod_10', 'FIP_numer_mod2_10', 'H_BB_mod_10',
      'H_BB_mod2_10', 'SO_mod_10', 'TB_BB_mod_10', 'ERA_10', 'FIP_10',
      'FIP_perc_10', 'WHIP_10', 'SO_perc_10', 'TB_BB_perc_10',
      'H_BB_perc_10', 'H_BB_roll_35', 'XB_roll_35', 'TB_roll_35',
      'IP_mod_35', 'BF_mod_35', 'ER_mod_35', 'FIP_numer_35',
      'FIP_numer_mod_35', 'FIP_numer_mod2_35', 'H_BB_mod_35',
      'H_BB_mod2_35', 'SO_mod_35', 'TB_BB_mod_35', 'ERA_35', 'FIP_35',
      'FIP_perc_35', 'WHIP_35', 'SO_perc_35', 'TB_BB_perc_35',
      'H_BB_perc_35', 'H_BB_roll_75', 'XB_roll_75', 'TB_roll_75',
      'IP_mod_75', 'BF_mod_75', 'ER_mod_75', 'FIP_numer_75',
      'FIP_numer_mod_75', 'FIP_numer_mod2_75', 'H_BB_mod_75',
      'H_BB_mod2_75', 'SO_mod_75', 'TB_BB_mod_75', 'ERA_75', 'FIP_75',
      'FIP_perc_75', 'WHIP_75', 'SO_perc_75', 'TB_BB_perc_75',
      'H_BB_perc_75']

  cols_to_add = ['Strt_'+col+suff for col in raw_cols_to_add for suff in ['_h','_v']]
  col_add_dict = {col:np.zeros(df.shape[0]) for col in cols_to_add}
  
  for i in range(df.shape[0]):
    row = df.iloc[i,:]
    sp_id_v = row['starting_pitcher_id_v']
    sp_id_h = row['starting_pitcher_id_h']
    if sp_id_v in pitcher_data_dict.keys():
      curr_df = pitcher_data_dict[sp_id_v]
      if not curr_df.empty:
        for col in raw_cols_to_add:
          value = curr_df[col].iloc[-1]
          col_add_dict['Strt_'+col+'_v'][i] = value
    if sp_id_h in pitcher_data_dict.keys():
      curr_df = pitcher_data_dict[sp_id_h]
      if not curr_df.empty:
        for col in raw_cols_to_add:
          value = curr_df[col].iloc[-1]
          col_add_dict['Strt_'+col+'_h'][i] = value
  
  # Concatenate all new columns at once to avoid fragmentation
  if col_add_dict:
    new_df = pd.DataFrame(col_add_dict, index=df.index)
    df = pd.concat([df, new_df], axis=1)
  
  return df
  
def get_bullpen_team_df(team, df, debug=False):
  """
  Calculate rolling bullpen statistics for a team.
  
  Args:
    team: Team abbreviation (e.g., 'NYY')
    df: DataFrame with game data
    debug: If True, print debug information
  
  Returns:
    DataFrame with rolling bullpen stats indexed by date_dblhead
  """
  visit_cols = [col for col in df.columns if not col.endswith('_h')]
  visit_cols_stripped = [strip_suffix(col,'_v') for col in visit_cols]
  home_cols = [col for col in df.columns if not col.endswith('_v')]
  home_cols_stripped = [strip_suffix(col,'_h') for col in home_cols]    

  df_team_v = df[(df.team_v==team)]
  opponent = df_team_v['team_h']
  df_team_v = df_team_v[visit_cols]
  df_team_v.columns = visit_cols_stripped
  df_team_v['home_game'] = 0
  df_team_v['opponent'] = opponent

  df_team_h = df[(df.team_h==team)]
  opponent = df_team_h['team_v']
  df_team_h = df_team_h[home_cols]
  df_team_h.columns = home_cols_stripped
  df_team_h['home_game'] = 1
  df_team_h['opponent'] = opponent
  
  # FIX: Use both home and away games, not just one or the other
  if df_team_h.empty and df_team_v.empty:
    if debug:
      print(f"[BULLPEN DEBUG] No games found for team {team}")
    return pd.DataFrame()
  
  df_team = pd.concat([df_team_h, df_team_v], ignore_index=True)
  df_team.sort_values(['date_dblhead'], inplace=True)
  df_team.reset_index(drop=True, inplace=True)
  
  if debug:
    print(f"[BULLPEN DEBUG] Team {team}: {len(df_team)} total games ({len(df_team_h)} home, {len(df_team_v)} away)")
  
  # Validate required Bpen columns exist
  required_bpen_cols = ['Bpen_IP', 'Bpen_H', 'Bpen_BFP', 'Bpen_HR', 'Bpen_R', 'Bpen_BB', 'Bpen_SO', 'Bpen_HBP', 'Bpen_x2B', 'Bpen_x3B']
  missing_cols = [col for col in required_bpen_cols if col not in df_team.columns]
  if missing_cols:
    if debug:
      print(f"[BULLPEN DEBUG] Team {team}: Missing required columns: {missing_cols}")
    return pd.DataFrame()
  
  # Check for data quality issues
  null_counts = df_team[required_bpen_cols].isnull().sum()
  if null_counts.any():
    if debug:
      print(f"[BULLPEN DEBUG] Team {team}: NULL values detected: {null_counts[null_counts > 0].to_dict()}")
  
  # defaults for pitching
  er_per_ip_def = (5/9)
  h_bb_per_ip_def = 1.5
  h_bb_per_bf_def = .37
  so_per_bf_def = .2
  ip_per_game_def = 3
  bf_per_game_def = 12
  tb_bb_perc_def = .45

  cols_to_agg = ['IP', 'H','BFP', 'HR', 'R',  'BB', 'SO',  'HBP', 'x2B', 'x3B']
  winsizes = [10,35,75]
  
  # Collect all new columns to avoid fragmentation
  new_columns = {}
  
  for winsize in winsizes:
    for raw_col in cols_to_agg:
      col_agg = 'Bpen_'+raw_col
      new_colname = 'Bpen_rollsum_'+raw_col+'_'+str(winsize)
      if col_agg in df_team.columns:
        new_columns[new_colname] = roll_column(df_team, col_agg, winsize)
      else:
        if debug:
          print(f"[BULLPEN DEBUG] Team {team}: Missing column {col_agg} for winsize {winsize}")
        new_columns[new_colname] = np.zeros(len(df_team))

    hit_col = 'Bpen_rollsum_H_'+str(winsize)
    bb_col = 'Bpen_rollsum_BB_'+str(winsize)
    h_bb_col = 'Bpen_H_BB_roll_'+str(winsize)
    double_col = 'Bpen_rollsum_x2B_'+str(winsize)
    triple_col = 'Bpen_rollsum_x3B_'+str(winsize)
    hr_col = 'Bpen_rollsum_HR_'+str(winsize)
    xb_col = 'Bpen_XB_roll_'+str(winsize)
    tb_col = 'Bpen_TB_roll_'+str(winsize)
    so_col = 'Bpen_rollsum_SO_'+str(winsize)
    so_mod_col = 'Bpen_SO_mod_'+str(winsize)
    ip_col = 'Bpen_rollsum_IP_'+str(winsize)
    ip_mod_col = 'Bpen_IP_mod_'+str(winsize)
    bf_col = 'Bpen_rollsum_BFP_'+str(winsize)
    bf_mod_col = 'Bpen_BF_mod_'+str(winsize)
    whip_col = 'Bpen_WHIP_'+str(winsize)
    so_perc_col = 'Bpen_SO_perc_'+str(winsize)
    h_bb_perc_col = 'Bpen_H_BB_perc_'+str(winsize)
    h_bb_mod_col = 'Bpen_H_BB_mod_'+str(winsize)
    h_bb_mod2_col = 'Bpen_H_BB_mod2_'+str(winsize)
    tb_bb_mod_col = 'Bpen_TB_BB_mod_'+str(winsize)
    tb_bb_perc_col = 'Bpen_TB_BB_perc_'+str(winsize)
    
    # Calculate using new_columns dict to avoid fragmentation
    h_bb = new_columns[hit_col] + new_columns[bb_col]
    new_columns[h_bb_col] = h_bb
    xb = new_columns[double_col] + 2*new_columns[triple_col] + 3*new_columns[hr_col]
    new_columns[xb_col] = xb
    tb = new_columns[hit_col] + xb
    new_columns[tb_col] = tb
    
    ip_mod = np.maximum(new_columns[ip_col], winsize*ip_per_game_def)
    new_columns[ip_mod_col] = ip_mod
    bf_mod = np.maximum(new_columns[bf_col], winsize*bf_per_game_def)
    new_columns[bf_mod_col] = bf_mod
    
    new_columns[h_bb_mod_col] = h_bb + h_bb_per_ip_def*(ip_mod - new_columns[ip_col])
    new_columns[h_bb_mod2_col] = h_bb + h_bb_per_bf_def*(bf_mod - new_columns[bf_col])
    new_columns[so_mod_col] = new_columns[so_col] + so_per_bf_def*(bf_mod - new_columns[bf_col])
    new_columns[tb_bb_mod_col] = (tb + new_columns[bb_col]) + tb_bb_perc_def*(bf_mod - new_columns[bf_col])
    
    # Calculate final metrics
    new_columns[whip_col] = new_columns[h_bb_mod_col] / ip_mod
    new_columns[so_perc_col] = new_columns[so_mod_col] / bf_mod
    new_columns[tb_bb_perc_col] = new_columns[tb_bb_mod_col] / bf_mod
    new_columns[h_bb_perc_col] = new_columns[h_bb_mod2_col] / bf_mod
  
  # Concatenate all new columns at once
  if new_columns:
    new_df = pd.DataFrame(new_columns, index=df_team.index)
    df_team = pd.concat([df_team, new_df], axis=1)
  
  df_team.set_index('date_dblhead', inplace=True)
  
  if debug:
    # Check if we're getting actual values or just defaults
    sample_cols = ['Bpen_WHIP_10', 'Bpen_SO_perc_10']
    for col in sample_cols:
      if col in df_team.columns:
        unique_vals = df_team[col].nunique()
        mean_val = df_team[col].mean()
        print(f"[BULLPEN DEBUG] Team {team}: {col} - {unique_vals} unique values, mean={mean_val:.3f}")
  
  return df_team
  
def get_bullpen_data(df, debug=False):
  """
  Calculate bullpen features for all games.
  
  Args:
    df: DataFrame with game data
    debug: If True, enable verbose debugging output
  
  Returns:
    DataFrame with bullpen features added
  """
  # Check if required starting pitcher columns exist
  required_strt_cols = ['Strt_IP_real_h', 'Strt_IP_real_v', 'Strt_BFP_h', 'Strt_BFP_v',
                       'Strt_R_h', 'Strt_R_v', 'Strt_H_h', 'Strt_H_v', 'Strt_HR_h', 'Strt_HR_v',
                       'Strt_x2B_h', 'Strt_x2B_v', 'Strt_BB_h', 'Strt_BB_v',
                       'Strt_HBP_h', 'Strt_HBP_v', 'Strt_SO_h', 'Strt_SO_v']
  
  missing_strt_cols = [col for col in required_strt_cols if col not in df.columns]
  if missing_strt_cols:
    print(f"[BULLPEN WARNING] Missing starting pitcher columns: {missing_strt_cols}")
    print("[BULLPEN WARNING] Bullpen calculation may use default values")
  
  ## Calculate some game level stats, specifically about
  ## relative stats for starting pitcher vs bullpen
  df['Bpen_IP_h'] = 9.0-df['Strt_IP_real_h']
  df['Bpen_IP_v'] = 9.0-df['Strt_IP_real_v']
  df['Bpen_BFP_h'] = df['AB_v']+df['BB_v']+df['HBP_v']-df['Strt_BFP_h']
  df['Bpen_BFP_v'] = df['AB_h']+df['BB_h']+df['HBP_h']-df['Strt_BFP_v']
  df['Bpen_R_h'] = df['R_v']-df['Strt_R_h']
  df['Bpen_R_v'] = df['R_h']-df['Strt_R_v']
  df['Bpen_H_h'] = df['H_v']-df['Strt_H_h']
  df['Bpen_H_v'] = df['H_h']-df['Strt_H_v']
  df['Bpen_HR_h'] = df['HR_v']-df['Strt_HR_h']
  df['Bpen_HR_v'] = df['HR_h']-df['Strt_HR_v']
  df['Bpen_x2B_h'] = df['x2B_v']-df['Strt_x2B_h']
  df['Bpen_x2B_v'] = df['x2B_h']-df['Strt_x2B_v']
  df['Bpen_x3B_h'] = df['x3B_v']-df['Strt_x3B_h']
  df['Bpen_x3B_v'] = df['x3B_h']-df['Strt_x3B_v']
  df['Bpen_BB_h'] = df['BB_v']-df['Strt_BB_h']
  df['Bpen_BB_v'] = df['BB_h']-df['Strt_BB_v']
  df['Bpen_HBP_h'] = df['HBP_v']-df['Strt_HBP_h']
  df['Bpen_HBP_v'] = df['HBP_h']-df['Strt_HBP_v']
  df['Bpen_SO_h'] = df['SO_v']-df['Strt_SO_h']
  df['Bpen_SO_v'] = df['SO_h']-df['Strt_SO_v']
  
  if debug:
    print(f"[BULLPEN DEBUG] Processing {len(df)} games")
    print(f"[BULLPEN DEBUG] Sample Bpen_IP_h values: {df['Bpen_IP_h'].head().tolist()}")
  
  teams = df[['team_h', 'team_v']].stack().unique().tolist()
  bullpen_team_data_dict = {}
  for team in teams:
    bullpen_team_data_dict[team] = get_bullpen_team_df(team, df, debug=debug)
    
  raw_cols_to_add = ['Bpen_IP', 'Bpen_BFP', 'Bpen_R', 'Bpen_H', 'Bpen_HR', 'Bpen_x2B',
    'Bpen_x3B', 'Bpen_BB', 'Bpen_HBP', 'Bpen_SO',  'Bpen_rollsum_IP_10', 'Bpen_rollsum_H_10',
    'Bpen_rollsum_BFP_10', 'Bpen_rollsum_HR_10', 'Bpen_rollsum_R_10',
    'Bpen_rollsum_BB_10', 'Bpen_rollsum_SO_10', 'Bpen_rollsum_HBP_10',
    'Bpen_rollsum_x2B_10', 'Bpen_rollsum_x3B_10', 'Bpen_H_BB_roll_10',
    'Bpen_XB_roll_10', 'Bpen_TB_roll_10', 'Bpen_IP_mod_10',
    'Bpen_BF_mod_10', 'Bpen_H_BB_mod_10', 'Bpen_H_BB_mod2_10',
    'Bpen_SO_mod_10', 'Bpen_TB_BB_mod_10', 'Bpen_WHIP_10',
    'Bpen_SO_perc_10', 'Bpen_TB_BB_perc_10', 'Bpen_H_BB_perc_10',
    'Bpen_rollsum_IP_35', 'Bpen_rollsum_H_35', 'Bpen_rollsum_BFP_35',
    'Bpen_rollsum_HR_35', 'Bpen_rollsum_R_35', 'Bpen_rollsum_BB_35',
    'Bpen_rollsum_SO_35', 'Bpen_rollsum_HBP_35', 'Bpen_rollsum_x2B_35',
    'Bpen_rollsum_x3B_35', 'Bpen_H_BB_roll_35', 'Bpen_XB_roll_35',
    'Bpen_TB_roll_35', 'Bpen_IP_mod_35', 'Bpen_BF_mod_35',
    'Bpen_H_BB_mod_35', 'Bpen_H_BB_mod2_35', 'Bpen_SO_mod_35',
    'Bpen_TB_BB_mod_35', 'Bpen_WHIP_35', 'Bpen_SO_perc_35',
    'Bpen_TB_BB_perc_35', 'Bpen_H_BB_perc_35', 'Bpen_rollsum_IP_75',
    'Bpen_rollsum_H_75', 'Bpen_rollsum_BFP_75', 'Bpen_rollsum_HR_75',
    'Bpen_rollsum_R_75', 'Bpen_rollsum_BB_75', 'Bpen_rollsum_SO_75',
    'Bpen_rollsum_HBP_75', 'Bpen_rollsum_x2B_75', 'Bpen_rollsum_x3B_75',
    'Bpen_H_BB_roll_75', 'Bpen_XB_roll_75', 'Bpen_TB_roll_75',
    'Bpen_IP_mod_75', 'Bpen_BF_mod_75', 'Bpen_H_BB_mod_75',
    'Bpen_H_BB_mod2_75', 'Bpen_SO_mod_75', 'Bpen_TB_BB_mod_75',
    'Bpen_WHIP_75', 'Bpen_SO_perc_75', 'Bpen_TB_BB_perc_75',
    'Bpen_H_BB_perc_75']

  cols_to_add = [col+suff for col in raw_cols_to_add for suff in ['_h','_v']]
  col_add_dict = {col:np.zeros(df.shape[0]) for col in cols_to_add}
  
  # Track how many lookups succeed vs fail
  lookups_success = 0
  lookups_failed = 0
  
  for i in range(df.shape[0]):
    row = df.iloc[i,:]
    home_team = row['team_h']
    visit_team = row['team_v']
    date_dblhead = row['date_dblhead']
    
    # Get home team bullpen data
    curr_df = bullpen_team_data_dict.get(home_team)
    if curr_df is not None and not curr_df.empty and date_dblhead in curr_df.index:
      for col in raw_cols_to_add:
        if col in curr_df.columns:
          value = curr_df.loc[date_dblhead,col]
          col_add_dict[col+'_h'][i] = value
      lookups_success += 1
    else:
      lookups_failed += 1
      if debug and i < 5:  # Only show first 5 failures to avoid spam
        reason = "empty dataframe" if (curr_df is None or curr_df.empty) else "date not in index"
        print(f"[BULLPEN DEBUG] Failed lookup for home team {home_team} game {i}: {reason}")
    
    # Get visiting team bullpen data
    curr_df = bullpen_team_data_dict.get(visit_team)
    if curr_df is not None and not curr_df.empty and date_dblhead in curr_df.index:
      for col in raw_cols_to_add:
        if col in curr_df.columns:
          value = curr_df.loc[date_dblhead,col]
          col_add_dict[col+'_v'][i] = value
      lookups_success += 1
    else:
      lookups_failed += 1
      if debug and i < 5:
        reason = "empty dataframe" if (curr_df is None or curr_df.empty) else "date not in index"
        print(f"[BULLPEN DEBUG] Failed lookup for away team {visit_team} game {i}: {reason}")
  
  # Concatenate all new columns at once to avoid fragmentation
  if col_add_dict:
    new_df = pd.DataFrame(col_add_dict, index=df.index)
    df = pd.concat([df, new_df], axis=1)
  
  if debug:
    success_rate = lookups_success / (lookups_success + lookups_failed) * 100 if (lookups_success + lookups_failed) > 0 else 0
    print(f"[BULLPEN DEBUG] Bullpen lookups: {lookups_success} succeeded, {lookups_failed} failed ({success_rate:.1f}% success rate)")
    
    # Check final values
    sample_cols = ['Bpen_WHIP_10_h', 'Bpen_SO_perc_10_h']
    for col in sample_cols:
      if col in df.columns:
        unique_vals = df[col].nunique()
        zero_count = (df[col] == 0).sum()
        print(f"[BULLPEN DEBUG] Final {col}: {unique_vals} unique values, {zero_count} zeros ({zero_count/len(df)*100:.1f}%)")
  
  return df

