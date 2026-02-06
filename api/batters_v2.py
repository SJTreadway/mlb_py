import re
import os
import time
import pandas as pd
import numpy as np

from tqdm import tqdm

from helpers import roll_column, get_team_league_map, safe_float, safe_int

from pybaseball import playerid_reverse_lookup, statcast_batter, playerid_lookup

from bs4 import BeautifulSoup

WINDOWS = [30,75,162,350]
YEAR = int(os.environ['YEAR'])

def process_batting_data(df):
  """
  Process batting data using pybaseball API instead of web scraping.
  """
  # step 1: get unique batter ids from our dataframe
  batter_ids = np.array([])
  for num in range(1,10):
    for suffix in ['_h','_v']:
      colname = 'batter'+str(num)+'_id'+suffix
      batter_ids = np.concatenate((batter_ids, pd.unique(df[colname])))
  batter_ids = pd.unique(batter_ids)
  
  # step 2: store batter data for each batter id to csv using API
  load_batting_data(batter_ids)
  
  # step 3: add in all batting feature
  bat_df = get_batting_feats(df, batter_ids)

  return get_lineup_averages(bat_df)

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

def load_batting_data(batter_ids):
  """
  Load batting data using pybaseball API (statcast_batter).
  Much faster than scraping Retrosheet and Baseball-Reference.
  """
  for b_id in tqdm(batter_ids, desc='Loading batter data via API'):
    if b_id and not pd.isna(b_id):
      fname_out = 'data/bat/batting_data_'+b_id+'.csv'
      
      # Convert retro ID to MLBAM ID for API
      mlbam_id = retro_to_mlbam(b_id)
      if not mlbam_id:
        print(f'Skipping batter {b_id} - could not convert to MLBAM ID')
        continue
      
      # Fetch current season data via API
      start_date = f'{YEAR}-03-01'
      end_date = f'{YEAR}-11-30'
      
      try:
        df_season = statcast_batter(start_date, end_date, mlbam_id)
        if df_season.empty:
          print(f'No data found for batter {b_id} (MLBAM: {mlbam_id})')
          continue
          
        # Transform statcast data to match expected format
        df_season = transform_statcast_batter(df_season)
        
        if not os.path.exists(fname_out):
          # Get historical data using API (start from 2008 when Statcast began)
          df_historical = get_historical_batting_data(mlbam_id)
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
        print(f'Error fetching data for batter {b_id}: {e}')
        continue

def transform_statcast_batter(df):
  """
  Transform Statcast batter data to match the expected format from Retrosheet.
  """
  if df.empty:
    return pd.DataFrame()
  
  # Group by game to get game-level stats
  df['game_date'] = pd.to_datetime(df['game_date'])
  df['date'] = df['game_date'].dt.strftime('%-m-%-d-%Y')
  
  # Initialize lists for aggregated data
  games = []
  
  # Group by game_date and game_pk to get game-level stats
  for (game_date, game_pk), group in df.groupby(['game_date', 'game_pk']):
    # Determine if home or away
    at_vs = 'VS' if group['home_team'].iloc[0] == group['stand'].iloc[0] else 'AT'
    opponent = group['away_team'].iloc[0] if at_vs == 'VS' else group['home_team'].iloc[0]
    
    # Calculate stats
    ab = len(group[group['description'].str.contains('hit_into_play|swinging_strike|called_strike|foul', case=False, na=False)])
    h = len(group[group['events'].isin(['single', 'double', 'triple', 'home_run'])])
    x2b = len(group[group['events'] == 'double'])
    x3b = len(group[group['events'] == 'triple'])
    hr = len(group[group['events'] == 'home_run'])
    bb = len(group[group['events'].isin(['walk', 'intent_walk'])])
    so = len(group[group['events'] == 'strikeout'])
    hbp = len(group[group['events'] == 'hit_by_pitch'])
    sb = len(group[group['events'] == 'stolen_base'])
    cs = len(group[group['events'] == 'caught_stealing'])
    # RBI not directly available in statcast - estimate from runs scored
    rbi = 0  # Cannot reliably calculate RBI from pitch-level data
    
    games.append({
      'date': game_date.strftime('%-m-%-d-%Y'),
      'dblhead_num': '',
      'at_vs': at_vs,
      'Opponent': opponent,
      'League': get_team_league_map().get(opponent, ''),
      'GS': 1 if len(group) > 0 else 0,
      'AB': ab,
      'R': 0,  # Runs scored by batter not directly available in pitch-level statcast data
      'H': h,
      'x2B': x2b,
      'x3B': x3b,
      'HR': hr,
      'RBI': rbi,
      'BB': bb,
      'IBB': len(group[group['events'] == 'intent_walk']),
      'SO': so,
      'HBP': hbp,
      'SH': len(group[group['events'] == 'sac_bunt']),
      'SF': len(group[group['events'] == 'sac_fly']),
      'XI': 0,
      'ROE': 0,
      'GDP': len(group[group['events'] == 'grounded_into_double_play']),
      'SB': sb,
      'CS': cs,
      'AVG': 0.0,  # Will be calculated cumulatively
      'OBP': 0.0,
      'SLG': 0.0,
      'BP': group['batting_order'].iloc[0] if 'batting_order' in group.columns else 0,
      'Pos': ','.join(group['position'].unique()) if 'position' in group.columns else ''
    })
  
  result_df = pd.DataFrame(games)
  
  # Calculate cumulative stats
  if not result_df.empty:
    result_df = calculate_cumulative_stats(result_df)
  
  return result_df

def calculate_cumulative_stats(df):
  """Calculate cumulative batting statistics (AVG, OBP, SLG)."""
  if df.empty:
    return df
  df = df.sort_values('date')
  df['cum_AB'] = df['AB'].cumsum()
  df['cum_H'] = df['H'].cumsum()
  df['cum_BB'] = df['BB'].cumsum()
  df['cum_HBP'] = df['HBP'].cumsum()
  df['cum_SF'] = df['SF'].cumsum()
  df['cum_xB'] = (df['x2B'] + 2*df['x3B'] + 3*df['HR']).cumsum()
  
  df['AVG'] = df['cum_H'] / df['cum_AB'].replace(0, np.nan)
  df['OBP'] = (df['cum_H'] + df['cum_BB'] + df['cum_HBP']) / (df['cum_AB'] + df['cum_BB'] + df['cum_HBP'] + df['cum_SF']).replace(0, np.nan)
  df['SLG'] = (df['cum_H'] + df['cum_xB']) / df['cum_AB'].replace(0, np.nan)
  
  # Drop cumulative columns
  df = df.drop(['cum_AB', 'cum_H', 'cum_BB', 'cum_HBP', 'cum_SF', 'cum_xB'], axis=1)
  
  return df

def get_historical_batting_data(mlbam_id):
  """
  Get historical batting data from 2008 onwards (when Statcast began).
  """
  all_data = []
  
  # Fetch data from 2008 to current year - 1
  for year in range(max(2008, YEAR - 5), YEAR):
    try:
      start_date = f'{year}-03-01'
      end_date = f'{year}-11-30'
      df_year = statcast_batter(start_date, end_date, mlbam_id)
      if not df_year.empty:
        all_data.append(transform_statcast_batter(df_year))
      time.sleep(0.1)  # Be nice to the API
    except Exception as e:
      print(f'Error fetching data for year {year}: {e}')
      continue
  
  if all_data:
    return pd.concat(all_data, ignore_index=True)
  return pd.DataFrame()

def process_batter_df(b_id):
  dict_def = get_position_defaults()
  fname = f'data/bat/batting_data_{b_id}.csv'
  pos = None  # Initialize to avoid NameError in except block
  try:
    batter_df = pd.read_csv(fname)
    pos = batter_df.Pos.mode()[0]

    # corner cases where the most common position was a pair
    if ',' in pos:
      pos = pos.split(',')[0]

    batter_df['date'] = pd.to_datetime(batter_df['date'], format='%m-%d-%Y').dt.strftime('%Y%m%d').astype(int)
    t_col = batter_df['dblhead_num'].copy()
    t_col = t_col.fillna(0)
    batter_df['dblheader_int'] = t_col.astype(int)
    
    # Collect all new columns in a dictionary to avoid DataFrame fragmentation
    new_columns = {}
    
    for winsize in WINDOWS:
      suff = str(winsize)
      for raw_col in ['AB','BB','H','x2B','x3B','HR','HBP','SO','SB','CS']:
        new_col = 'rollsum_'+raw_col+'_'+suff
        new_columns[new_col] = roll_column(batter_df, raw_col, winsize)

      ab_per_game_def = 2
      pa_per_game_def = 2
      batavg_def = dict_def[pos]['batavg']
      obp_def = dict_def[pos]['obp']
      slg_def = dict_def[pos]['slg']
      slgmod_def = dict_def[pos]['slgmod']
      so_bat_perc_def = dict_def[pos]['sobat']

      # Columns created by aggregation above
      ab_col = 'rollsum_AB_'+str(winsize)
      h_col = 'rollsum_H_'+str(winsize)
      bb_col = 'rollsum_BB_'+str(winsize)
      hbp_col = 'rollsum_HBP_'+str(winsize)
      doub_col = 'rollsum_x2B_'+str(winsize)
      trip_col = 'rollsum_x3B_'+str(winsize)
      hr_col = 'rollsum_HR_'+str(winsize)
      so_col = 'rollsum_SO_'+str(winsize)

      # Calculate intermediate values
      abmod_col = 'ABmod_'+str(winsize)
      fakeab_col = 'fakeAB_'+str(winsize)
      pa_col = 'PA_'+str(winsize)
      pamod_col = 'PAmod_'+str(winsize)
      fakepa_col = 'fakePA_'+str(winsize)
      xb_col = 'XB_'+str(winsize) # represents extra bases beyond hits
      slg_col = 'SLG_'+str(winsize)
      slgmod_col = 'SLGmod_'+str(winsize)
      batavg_col = 'BATAVG_'+str(winsize)
      so_bat_perc_col = 'SObat_perc_'+str(winsize)
      obp_col = 'OBP_'+str(winsize)
      obs_col = 'OBS_'+str(winsize)

      # calculate BATAVG, with smoothing for low AB numbers
      abmod = np.maximum(new_columns[ab_col], winsize*ab_per_game_def)
      new_columns[abmod_col] = abmod
      fakeab = np.minimum(abmod - new_columns[ab_col], 0)
      new_columns[fakeab_col] = fakeab
      new_columns[batavg_col] = (new_columns[h_col] + (fakeab * batavg_def)) / abmod

      # calculate SLG, with smoothing for low AB numbers
      xb = new_columns[doub_col] + 2*new_columns[trip_col] + 3*new_columns[hr_col]
      new_columns[xb_col] = xb
      new_columns[slg_col] = (new_columns[h_col] + xb + (fakeab * slg_def)) / abmod

      # calculate OBP, with smoothing for low PA numbers
      pa = new_columns[ab_col] + new_columns[bb_col] + new_columns[hbp_col]
      new_columns[pa_col] = pa
      pamod = np.maximum(pa, winsize*pa_per_game_def)
      new_columns[pamod_col] = pamod
      fakepa = np.minimum(pamod - pa, 0)
      new_columns[fakepa_col] = fakepa
      new_columns[obp_col] = (new_columns[h_col] + new_columns[bb_col] + new_columns[hbp_col] + (fakepa * obp_def)) / pamod

      # calculate SLGmod, with smoothing for low PA numbers
      new_columns[slgmod_col] = (new_columns[so_col] + new_columns[bb_col] + new_columns[hbp_col] + xb + (fakepa * slgmod_def)) / pamod

      # calculate SObat_perc, with smoothing for low PA numbers
      new_columns[so_bat_perc_col] = (new_columns[so_col] + (fakepa * so_bat_perc_def)) / pamod

      # calculate OBS
      new_columns[obs_col] = new_columns[obp_col] + new_columns[slg_col]
    
    # Concatenate all new columns at once to avoid fragmentation
    if new_columns:
      new_df = pd.DataFrame(new_columns, index=batter_df.index)
      batter_df = pd.concat([batter_df, new_df], axis=1)
    
    # Set index after all columns are added
    batter_df['date_dblhead'] = (batter_df['date'].astype(str) + batter_df['dblheader_int'].astype(str)).astype(int)
    batter_df.set_index('date_dblhead', inplace=True)
  except Exception as e:
    try:
      print(f'issue for {fname} at position {pos}, returning None: {e}')
    except:
      print(f'issue for {fname}, returning None')
    batter_df = None
  return batter_df

def get_batter_ids_from_row(row):
  b_cols = ['batter1_id_h', 'batter1_id_v', 'batter2_id_h', 'batter2_id_v',
      'batter3_id_h', 'batter3_id_v', 'batter4_id_h', 'batter4_id_v',
      'batter5_id_h', 'batter5_id_v', 'batter6_id_h', 'batter6_id_v',
      'batter7_id_h', 'batter7_id_v', 'batter8_id_h', 'batter8_id_v',
      'batter9_id_h', 'batter9_id_v']
  return row.loc[b_cols].to_dict()

def get_batting_feats(df, batter_ids):
  batter_data_dict = {}
  for b_id in batter_ids:
    batter_data_dict[b_id] = process_batter_df(b_id) 
  new_col_dict = {}
  colstems = ['BATAVG', 'OBP', 'SLG', 'OBS', 'SLGmod','SObat_perc']
  new_col_list = [stem+'_'+str(winsize)+'_b'+str(i)+hv for stem in colstems for winsize in WINDOWS
                  for i in range(1,10) for hv in ['_h','_v']]
  for col in new_col_list:
    new_col_dict[col] = np.empty(df.shape[0])
    new_col_dict[col].fill(np.nan)

  for i in range(df.shape[0]):
    row = df.iloc[i,:]
    bid_dict = get_batter_ids_from_row(row)
    date_dblhead = row['date_dblhead']
    for hv in ['_h','_v']:
      for j in range(1,10):
        curr_col = 'batter'+str(j)+'_id'+hv
        curr_b_id = bid_dict[curr_col]
        if curr_b_id in batter_data_dict.keys():
          curr_batter_df = batter_data_dict[curr_b_id]
          if (curr_batter_df is not None) and (curr_batter_df.shape[0]>0):
            try:
              curr_batter_row = curr_batter_df.loc[date_dblhead,:]
            except:
              #print(f'date not found for batter {curr_b_id} game {date_dblhead}')
              prev_game_indices = np.where(curr_batter_df.index<date_dblhead)[0]
              if len(prev_game_indices)==0:
                index_to_use = 0
              else:
                index_to_use = np.max(prev_game_indices)
              curr_batter_row = curr_batter_df.iloc[index_to_use,:]
              #print(f'using date {curr_batter_df.index[index_to_use]}')
            if (curr_batter_row.ndim>1):
              curr_batter_row = curr_batter_row.iloc[0,:]
            for stem in colstems:
              for winsize in WINDOWS:
                newcolname = stem+'_'+str(winsize)+'_b'+str(j)+hv
                new_col_dict[newcolname][i] = curr_batter_row[stem+'_'+str(winsize)]
          else:
            print(f'No data found for {curr_b_id}')
        else:
          print(f'batter not found for {curr_b_id}')
  for key, val in new_col_dict.items():
    df[key] = val
  return df

def get_lineup_averages(df):
  default_dict = get_position_defaults()
  colstems = ['BATAVG', 'OBP', 'SLG', 'OBS', 'SLGmod','SObat_perc']
  newcols89 = [stem+'_'+str(winsize)+'_b'+str(i)+hv for stem in colstems for winsize in WINDOWS
                for hv in ['_h','_v'] for i in range(1,10)]
  for col in newcols89:
    stem = col.split('_')[0].lower()
    df[col].fillna(default_dict['p'][stem])

  w9 = np.array([0.12541131, 0.12159052, 0.11787189, 0.11434144, 0.11096691,
      0.10772781, 0.10430724, 0.10078822, 0.09699465])
  w8 = w9[:-1]/np.sum(w9[:-1])
  for col in colstems:
    for winsize in WINDOWS:
      for hv in ['_h','_v']:
        b_cols9 = [col+'_'+str(winsize)+'_b'+str(i)+hv for i in range(1,10)]
        b_cols8 = [col+'_'+str(winsize)+'_b'+str(i)+hv for i in range(1,9)]
        fcolname9 = 'lineup9_'+col+'_'+str(winsize)+hv
        fcolname8 = 'lineup8_'+col+'_'+str(winsize)+hv
        fcolname9w = 'lineup9_'+col+'_'+str(winsize)+'_w'+hv
        fcolname8w = 'lineup8_'+col+'_'+str(winsize)+'_w'+hv
        df[fcolname9] = np.mean(df.loc[:,b_cols9].to_numpy(),axis=1)
        df[fcolname8] = np.mean(df.loc[:,b_cols8].to_numpy(),axis=1)
        df[fcolname9w] = df.loc[:,b_cols9].to_numpy().dot(w9)
        df[fcolname8w] = df.loc[:,b_cols8].to_numpy().dot(w8)
  return df

def get_position_defaults():
  ## Set up position level defaults
  dd = {}
  dd_p = {'batavg': .100, 'obp': .150, 'slg': .180, 'slgmod': .220, 'obs': .330, 'sobat': .3}
  dd_ss_c = {'batavg': .205, 'obp': .260, 'slg': .300, 'slgmod': .320, 'obs': .540, 'sobat': .25}
  dd_2b_3b = {'batavg': .240, 'obp': .280, 'slg': .350, 'slgmod': .355, 'obs': .630, 'sobat': .2}
  dd_rest = {'batavg': .255, 'obp': .310, 'slg': .380, 'slgmod': .430, 'obs': .690, 'sobat': .2}
  dd['p'] = dd_p
  dd['ss'] = dd_ss_c
  dd['c'] = dd_ss_c
  dd['2b'] = dd_2b_3b
  dd['3b'] = dd_2b_3b
  dd['1b'] = dd_rest
  dd['lf'] = dd_rest
  dd['rf'] = dd_rest
  dd['cf'] = dd_rest
  dd['ph'] = dd_rest
  dd['pr'] = dd_ss_c
  dd['dh'] = dd_rest
  return dd
  
