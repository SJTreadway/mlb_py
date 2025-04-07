import os
import pandas as pd
import numpy as np
import requests

from tqdm import tqdm

from helpers import roll_column

from bs4 import BeautifulSoup

URL_PREFIX = 'https://www.retrosheet.org/boxesetc/'
BREF_URL_PREFIX = 'https://www.baseball-reference.com/players/gl.fcgi'
WINDOWS = [30,75,162,350]
YEAR = 2025

def process_batting_data(df):
  # step 1: get unique batter ids from our dataframe
  batter_ids = np.array([])
  for num in range(1,10):
    for suffix in ['_h','_v']:
      colname = 'batter'+str(num)+'_id'+suffix
      batter_ids = np.concatenate((batter_ids, pd.unique(df[colname])))
  batter_ids = pd.unique(batter_ids)
  
  # step 2: store batter data for each batter id to csv
  load_batting_data(batter_ids)

  # get current season data from BREF
  #get_bref_current_season_data(batter_ids[-1])
  
  # step 3: add in all batting feature
  bat_df = get_batting_feats(df, batter_ids)

  return get_lineup_averages(bat_df)

def load_batting_data(batter_ids):
  for i,b_id in tqdm(enumerate(batter_ids), total=len(batter_ids)):
    if b_id:
      df_temp = get_full_batting_data(b_id)
      fname_out = 'data/bat/batting_data_'+b_id+'.csv'
      if not os.path.exists(fname_out):
        df_temp.to_csv(fname_out, index=False)

# Get all the data for a particular batter
def get_full_batting_data(batter_id):
  if not batter_id:
    return pd.DataFrame()
  link_list = get_daily_season_links(batter_id)
  df_batting = pd.DataFrame()
  for url in link_list:
    df_batting = pd.concat((df_batting, get_season_batting_data(url)))
  return df_batting

        
def get_bref_current_season_data(pid):
  url = BREF_URL_PREFIX+'?id='+pid+'&t=b&year='+str(YEAR)
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  html=list(soup.children)
  return None
 

def get_daily_season_links(batter_id):
  letter = batter_id.upper()[0]
  url = URL_PREFIX+letter+'/P'+batter_id+'.htm'
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  html=list(soup.children)
  body = list(html[2].children)[5]
  pre_texts = [x for x in body.find_all('pre')]
  secnum = np.where([x.get_text().strip().startswith('Batting Record') for x in pre_texts])[0][0]
  a_pre_texts = pre_texts[secnum].find_all('a')
  daily_season_links = [URL_PREFIX+x.attrs['href'][3:] for x in a_pre_texts if x.get_text()=='Daily']
  return daily_season_links
  
def get_season_batting_data(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  html=list(soup.children)[-1]
  body = list(html.children)[-1]
  sec_next = list(body.children)
  secnum = np.where(["Opponent" in str(x) for x in sec_next])[0][0]
  key_section = sec_next[secnum]
  working_part = list(key_section.children)
  p_header = working_part[0].strip().split()
  mod_header= ['at_vs','Opponent','League', 'GS', 'AB', 'R', 'H', 'x2B', 'x3B', 'HR',
      'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SH', 'SF', 'XI', 'ROE', 'GDP',
      'SB', 'CS', 'AVG', 'OBP', 'SLG', 'BP', 'Pos']

  date_list = []
  day_href_list = []
  for k in range(1,len(working_part),4):
    date_list.append(working_part[k].get_text().strip())
    day_href_list.append(working_part[k].attrs['href'])

  dblhead_num_list = []
  for k in range(2,len(working_part),4):
    dblhead_num_list.append(working_part[k].strip())

  game_href_list = []
  for k in range(3,len(working_part),4):
    game_href_list.append(working_part[k].attrs['href'])

  main_data_matrix = []
  for k in range(4,len(working_part),4):
    main_data_row = (working_part[k].strip().split())[:27]
    main_data_matrix.append(main_data_row)
  row_sizes = [len(x) for x in main_data_matrix]
  max_row_size = max(row_sizes)
  min_row_size = min(row_sizes)
  if (min_row_size == max_row_size) and (max_row_size==27):
    # Everything has all 27 columns
    batter_df = pd.DataFrame(main_data_matrix, columns = mod_header)
  elif (min_row_size == max_row_size) and (max_row_size==26):
    # Everything has 26 columns, will guess position is missing
    batter_df = pd.DataFrame(main_data_matrix, columns = mod_header[:26])
    batter_df['Pos'] = ''
  elif (min_row_size == 26) and (max_row_size==27):
    # Guessing position is missing for some rows but not others
    main_data_matrix = [x if len(x)==27 else x+[''] for x in main_data_matrix]
    batter_df = pd.DataFrame(main_data_matrix, columns = mod_header)
  else:
    print('finding rows with less than 26 or more than 27 entries - Returning None')
    return(None)
  batter_df['date'] = date_list
  batter_df['dblhead_num'] = dblhead_num_list
  return batter_df

def process_batter_df(b_id):
  dict_def = get_position_defaults()
  fname = f'data/bat/batting_data_{b_id}.csv'
  try:
    batter_df = pd.read_csv(fname)
    pos = batter_df.Pos.mode()[0]

    # corner cases where the most common position was a pair
    if ',' in pos:
      pos = pos.split(',')[0]

    batter_df['date'] = pd.to_datetime(batter_df['date'], format='%m-%d-%Y').dt.strftime('%Y%m%d').astype(int)
    t_col = batter_df['dblhead_num'].copy()
    t_col[np.isnan(t_col)] = 0
    batter_df['dblheader_int'] = t_col.astype(int)
    for winsize in WINDOWS:
      suff = str(winsize)
      for raw_col in ['AB','BB','H','x2B','x3B','HR','HBP','SO','SB','CS']:
        new_col = 'rollsum_'+raw_col+'_'+suff
        batter_df[new_col] = roll_column(batter_df, raw_col, winsize)

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

      # Columns I will define below
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
      batter_df[abmod_col] = np.maximum(batter_df[ab_col],winsize*ab_per_game_def)
      batter_df[fakeab_col] = np.minimum(batter_df[abmod_col]-batter_df[ab_col],0)
      batter_df[batavg_col] = (batter_df[h_col] + (batter_df[fakeab_col]*batavg_def))/(batter_df[abmod_col])

      # calculate SLG, with smoothing for low AB numbers
      batter_df[xb_col] = batter_df[doub_col] + 2*batter_df[trip_col] + 3*batter_df[hr_col]
      batter_df[slg_col] = (batter_df[h_col] + batter_df[xb_col] +
                                (batter_df[fakeab_col]*slg_def))/(batter_df[abmod_col])

      # calculate OBP, with smoothing for low PA numbers
      batter_df[pa_col] = batter_df[ab_col]+batter_df[bb_col]+batter_df[hbp_col]
      batter_df[pamod_col] = np.maximum(batter_df[pa_col],winsize*pa_per_game_def)
      batter_df[fakepa_col] = np.minimum(batter_df[pamod_col]-batter_df[pa_col],0)
      batter_df[obp_col] = (batter_df[h_col] + batter_df[bb_col] + batter_df[hbp_col]
                            + (batter_df[fakepa_col]*obp_def))/(
                              batter_df[pamod_col])

      # calculate SLGmod, with smoothing for low PA numbers
      batter_df[slgmod_col] = (batter_df[so_col] + batter_df[bb_col] + batter_df[hbp_col]
                                +batter_df[xb_col] + (batter_df[fakepa_col]*slgmod_def))/(
                              batter_df[pamod_col])

      # calculate SObat_perc, with smoothing for low PA numbers
      batter_df[so_bat_perc_col] = (batter_df[so_col] + (batter_df[fakepa_col]*so_bat_perc_def))/(
                              batter_df[pamod_col])

      # calculate OBS
      batter_df[obs_col] = batter_df[obp_col]+batter_df[slg_col]

      batter_df['date_dblhead'] = (batter_df['date'].astype(str) + batter_df['dblheader_int'].astype(str)).astype(int)
      batter_df.set_index('date_dblhead', inplace=True)
  except Exception as e:
    try:
      print(f'issue for {fname} at position {pos}, returning None')
    except:
      print(f'issue for {fname}, returning None')
    batter_df = None
  return batter_df

def get_batter_ids_from_row(row):
  b_cols = ['batter1_id_h', 'batter1_id_v', 'batter2_id_h',
      'batter2_id_v', 'batter3_id_h', 'batter3_id_v',
      'batter4_id_h', 'batter4_id_v', 'batter5_id_h',
      'batter5_id_v', 'batter6_id_h', 'batter6_id_v',
      'batter7_id_h', 'batter7_id_v', 'batter8_id_h',
      'batter8_id_v', 'batter9_id_h', 'batter9_id_v']
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
              print(f'date not found for batter {curr_b_id} game {date_dblhead}')
              prev_game_indices = np.where(curr_batter_df.index<date_dblhead)[0]
              if len(prev_game_indices)==0:
                index_to_use = 0
              else:
                index_to_use = np.max(prev_game_indices)
              curr_batter_row = curr_batter_df.iloc[index_to_use,:]
              print(f'using date {curr_batter_df.index[index_to_use]}')
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
  