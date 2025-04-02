import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import requests

from helpers import roll_column, strip_suffix

URL_PREFIX = 'https://www.retrosheet.org/boxesetc/'
WINDOWS = [10,35,75]

# Get all the data for a particular pitcher
def get_full_pitching_data(pitcher_id):
  if not pitcher_id:
    return pd.DataFrame()
  link_list = get_daily_season_links(pitcher_id)
  df_pitching = pd.DataFrame()
  for url in link_list:
    df_pitching = pd.concat((df_pitching, get_season_pitching_data(url)))
  return df_pitching

### Get the links to the pitcher-season tables given the pitcher id
def get_daily_season_links(pitcher_id):
  letter = pitcher_id.upper()[0]
  url = URL_PREFIX+letter+'/P'+pitcher_id+'.htm'
  #time.sleep(.1)
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  html=list(soup.children)
  body = list(html[2].children)[5]
  pre_texts = [x for x in body.find_all('pre')]
  secnum = np.where([x.get_text().strip().startswith('Pitching Record') for x in pre_texts])[0][0]
  a_pre_texts = pre_texts[secnum].find_all('a')
  daily_season_links = [URL_PREFIX+x.attrs['href'][3:] for x in a_pre_texts if x.get_text()=='Daily']
  return daily_season_links

## Given the url that refers to a specific pitcher and season
## we scrape the data and process it a bit
def get_season_pitching_data(url):    
  #time.sleep(.1)
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  html=list(soup.children)[-1]
  body = list(html.children)[-1]
  sec_next = list(body.children)
  secnum = np.where(["Opponent" in str(x) for x in sec_next])[0][0]
  key_section = sec_next[secnum]
  working_part = list(key_section.children)
  p_header = working_part[0].strip().split()
  mod_header= ['at_vs','Opponent','League', 'GS', 'CG', 'SHO', 'GF', 'SV', 'IP', 'H',
          'BFP', 'HR', 'R', 'ER', 'BB', 'IB', 'SO', 'SH', 'SF', 'WP', 'HBP',
          'BK', 'x2B', 'x3B', 'GDP', 'ROE', 'W', 'L', 'ERA']

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
    main_data_row = (working_part[k].strip().split())[:29]
    main_data_matrix.append(main_data_row)

  pitch_df = pd.DataFrame(main_data_matrix, columns = mod_header)
  pitch_df['date'] = date_list
  pitch_df['dblhead_num'] = dblhead_num_list
  
  pitch_df['date'] = pd.to_datetime(pitch_df['date'], format='%m-%d-%Y').dt.strftime('%Y%m%d').astype(int)
  pitch_df['dblhead_num'] = pitch_df['dblhead_num'].apply(lambda x: int(x) if str(x).strip().isdigit() else 0)

  # Convert IP to proper mathematical format
  ip = float(pitch_df['IP'][0])
  pitch_df['IP_real'] = (ip - (ip % 1)) + (ip % 1) * (10/3)
  cols_to_agg = ['IP_real', 'H','BFP', 'HR', 'R', 'ER', 'BB', 'IB', 'SO', 'SH', 'SF', 'WP', 'HBP', 'BK', 'x2B', 'x3B']
  
  # defaults for pitching
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
    for raw_col in cols_to_agg:
      new_colname = 'rollsum_'+raw_col+'_'+str(winsize)        
      pitch_df[new_colname] = roll_column(pitch_df, raw_col, winsize)

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
    pitch_df[h_bb_col] = pitch_df[hit_col]+pitch_df[bb_col]
    pitch_df[xb_col] = pitch_df[double_col]+2*pitch_df[triple_col]+3*pitch_df[hr_col]
    pitch_df[tb_col] = pitch_df[hit_col]+pitch_df[xb_col]
    pitch_df[ip_mod_col] = np.maximum(pitch_df[ip_col], winsize*ip_per_game_def)
    pitch_df[bf_mod_col] = np.maximum(pitch_df[bf_col], winsize*bf_per_game_def)
    pitch_df[er_mod_col] = pitch_df[er_col] + er_per_ip_def*(pitch_df[ip_mod_col]-pitch_df[ip_col])
    pitch_df[fip_numer_col] = 13*pitch_df[hr_col] + 3*pitch_df[h_bb_col] -2*pitch_df[so_col]
    pitch_df[fip_numer_mod_col] = pitch_df[fip_numer_col] + fip_numer_per_ip_def*(pitch_df[ip_mod_col]-pitch_df[ip_col])
    pitch_df[fip_numer_mod2_col] = pitch_df[fip_numer_col] + fip_numer_per_bf_def*(pitch_df[bf_mod_col]-pitch_df[bf_col])
    pitch_df[h_bb_mod_col] = pitch_df[h_bb_col] + h_bb_per_ip_def*(pitch_df[ip_mod_col]-pitch_df[ip_col])
    pitch_df[h_bb_mod2_col] = pitch_df[h_bb_col] + h_bb_per_bf_def*(pitch_df[bf_mod_col]-pitch_df[bf_col])
    pitch_df[so_mod_col] = pitch_df[so_col] + so_per_bf_def*(pitch_df[bf_mod_col]-pitch_df[bf_col])
    pitch_df[tb_bb_mod_col] = (pitch_df[tb_col] + pitch_df[bb_col])+ tb_bb_perc_def*(pitch_df[bf_mod_col]-pitch_df[bf_col])
    pitch_df[era_col] = (pitch_df[er_mod_col]/pitch_df[ip_mod_col])*9
    pitch_df[fip_col] = (pitch_df[fip_numer_mod_col]/pitch_df[ip_mod_col])
    pitch_df[fip_perc_col] = (pitch_df[fip_numer_mod_col]/pitch_df[bf_mod_col])
    pitch_df[whip_col] = pitch_df[h_bb_mod_col]/pitch_df[ip_mod_col]
    pitch_df[so_perc_col] = pitch_df[so_mod_col]/pitch_df[bf_mod_col]
    pitch_df[tb_bb_perc_col] = pitch_df[tb_bb_mod_col]/pitch_df[bf_mod_col]
    pitch_df[h_bb_perc_col] = pitch_df[h_bb_mod2_col]/pitch_df[bf_mod_col]

  pitch_df['date_dblhead'] = (pitch_df['date'].astype(str) + pitch_df['dblhead_num'].astype(str)).astype(int)
    
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

  col_add_dict = {col:np.zeros(pitch_df.shape[0]) for col in cols_to_add}
  
  for col in cols_to_add:
    pitch_df[col] = col_add_dict[col]
  
  pitch_df.set_index('date_dblhead', inplace=True)
  pitch_df.drop(['at_vs', 'Opponent', 'League'], axis=1, inplace=True)

  return pitch_df
  
def get_bullpen_data_per_team(df):
  # Team Bullpen Average Data
  ## Calculate some game level stats, specifically about
  ## relative stats for starting pitcher vs bullpen
  df['Bpen_IP'] = 9.0-df['IP_real']
  df['Bpen_BFP'] = df['AB_v']+df['BB_v']+df['HBP_v']-df['Strt_BFP_h']
  df['Bpen_R'] = df['R_v']-df['Strt_R_h']
  df['Bpen_H'] = df['H_v']-df['Strt_H_h']
  df['Bpen_HR'] = df['HR_v']-df['Strt_HR_h']
  df['Bpen_x2B'] = df['x2B_v']-df['Strt_x2B_h']
  df['Bpen_x3B'] = df['x3B_v']-df['Strt_x3B_h']
  df['Bpen_BB'] = df['BB_v']-df['Strt_BB_h']
  df['Bpen_HBP'] = df['HBP_v']-df['Strt_HBP_h']
  df['Bpen_SO'] = df['SO_v']-df['Strt_SO_h']
  
  er_per_ip_def = (5/9)
  h_bb_per_ip_def = 1.5
  h_bb_per_bf_def = .37
  so_per_bf_def = .2
  ip_per_game_def = 2
  bf_per_game_def = 6
  tb_bb_perc_def = .45

  cols_to_agg = ['IP', 'H','BFP', 'HR', 'R',  'BB', 'SO',  'HBP', 'x2B', 'x3B']
  for col in cols_to_agg:
    df[col] = 0.0

  winsizes = [10,35,75]
  for winsize in winsizes:
    for raw_col in cols_to_agg:
      col_agg = 'Bpen_'+raw_col
      new_colname = 'Bpen_rollsum_'+raw_col+'_'+str(winsize)        
      df[new_colname] = roll_column(df, col_agg, winsize)

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
    h_bb_mod2_col = 'Bpen_Bpen_H_BB_mod2_'+str(winsize)
    tb_bb_mod_col = 'Bpen_TB_BB_mod_'+str(winsize)
    tb_bb_perc_col = 'Bpen_TB_BB_perc_'+str(winsize)
    df[h_bb_col] = df[hit_col]+df[bb_col]
    df[xb_col] = df[double_col]+2*df[triple_col]+3*df[hr_col]
    df[tb_col] = df[hit_col]+df[xb_col]
    df[ip_mod_col] = np.maximum(df[ip_col], winsize*ip_per_game_def)
    df[bf_mod_col] = np.maximum(df[bf_col], winsize*bf_per_game_def)
    df[h_bb_mod_col] = df[h_bb_col] + h_bb_per_ip_def*(df[ip_mod_col]-df[ip_col])
    df[h_bb_mod2_col] = df[h_bb_col] + h_bb_per_bf_def*(df[bf_mod_col]-df[bf_col])
    df[so_mod_col] = df[so_col] + so_per_bf_def*(df[bf_mod_col]-df[bf_col])
    df[tb_bb_mod_col] = (df[tb_col] + df[bb_col])+ tb_bb_perc_def*(df[bf_mod_col]-df[bf_col])
    df[whip_col] = df[h_bb_mod_col]/df[ip_mod_col]
    df[so_perc_col] = df[so_mod_col]/df[bf_mod_col]
    df[tb_bb_perc_col] = df[tb_bb_mod_col]/df[bf_mod_col]
    df[h_bb_perc_col] = df[h_bb_mod2_col]/df[bf_mod_col]
       
  return df
  
def get_bullpen_data(df):
  bp_df = get_bullpen_data_per_team(df)

  raw_cols_to_add = ['Bpen_IP', 'Bpen_BFP', 'Bpen_R', 'Bpen_H', 'Bpen_HR', 'Bpen_x2B',
    'Bpen_x3B', 'Bpen_BB', 'Bpen_HBP', 'Bpen_SO',  'Bpen_rollsum_IP_10', 'Bpen_rollsum_H_10',
    'Bpen_rollsum_BFP_10', 'Bpen_rollsum_HR_10', 'Bpen_rollsum_R_10',
    'Bpen_rollsum_BB_10', 'Bpen_rollsum_SO_10', 'Bpen_rollsum_HBP_10',
    'Bpen_rollsum_x2B_10', 'Bpen_rollsum_x3B_10', 'Bpen_H_BB_roll_10',
    'Bpen_XB_roll_10', 'Bpen_TB_roll_10', 'Bpen_IP_mod_10',
    'Bpen_BF_mod_10', 'Bpen_H_BB_mod_10', 'Bpen_Bpen_H_BB_mod2_10',
    'Bpen_SO_mod_10', 'Bpen_TB_BB_mod_10', 'Bpen_WHIP_10',
    'Bpen_SO_perc_10', 'Bpen_TB_BB_perc_10', 'Bpen_H_BB_perc_10',
    'Bpen_rollsum_IP_35', 'Bpen_rollsum_H_35', 'Bpen_rollsum_BFP_35',
    'Bpen_rollsum_HR_35', 'Bpen_rollsum_R_35', 'Bpen_rollsum_BB_35',
    'Bpen_rollsum_SO_35', 'Bpen_rollsum_HBP_35', 'Bpen_rollsum_x2B_35',
    'Bpen_rollsum_x3B_35', 'Bpen_H_BB_roll_35', 'Bpen_XB_roll_35',
    'Bpen_TB_roll_35', 'Bpen_IP_mod_35', 'Bpen_BF_mod_35',
    'Bpen_H_BB_mod_35', 'Bpen_Bpen_H_BB_mod2_35', 'Bpen_SO_mod_35',
    'Bpen_TB_BB_mod_35', 'Bpen_WHIP_35', 'Bpen_SO_perc_35',
    'Bpen_TB_BB_perc_35', 'Bpen_H_BB_perc_35', 'Bpen_rollsum_IP_75',
    'Bpen_rollsum_H_75', 'Bpen_rollsum_BFP_75', 'Bpen_rollsum_HR_75',
    'Bpen_rollsum_R_75', 'Bpen_rollsum_BB_75', 'Bpen_rollsum_SO_75',
    'Bpen_rollsum_HBP_75', 'Bpen_rollsum_x2B_75', 'Bpen_rollsum_x3B_75',
    'Bpen_H_BB_roll_75', 'Bpen_XB_roll_75', 'Bpen_TB_roll_75',
    'Bpen_IP_mod_75', 'Bpen_BF_mod_75', 'Bpen_H_BB_mod_75',
    'Bpen_Bpen_H_BB_mod2_75', 'Bpen_SO_mod_75', 'Bpen_TB_BB_mod_75',
    'Bpen_WHIP_75', 'Bpen_SO_perc_75', 'Bpen_TB_BB_perc_75',
    'Bpen_H_BB_perc_75']

  cols_to_add = [col+suff for col in raw_cols_to_add for suff in ['_h','_v']]
  col_add_dict = {col:np.zeros(df.shape[0]) for col in cols_to_add}
  
  for col in cols_to_add:
    bp_df[col] = col_add_dict[col]

  #bp_df.reset_index(drop=True, inplace=True)
  #bp_df['date'] = pd.to_datetime(bp_df['date'], format='%m-%d-%Y').dt.strftime('%Y%m%d').astype(int)
  #bp_df['dblhead_num'] = bp_df['dblhead_num'].apply(lambda x: int(x) if str(x).strip().isdigit() else 0)
  #bp_df['date_dblhead'] = bp_df['dblhead_num'].astype(int)
  #bp_df.set_index('date_dblhead', inplace=True)
  #print(bp_df.sample(5))
  return bp_df
