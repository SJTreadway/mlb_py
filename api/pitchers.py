import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import requests

from helpers import roll_column

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
  pitch_df['dblheader_int'] = pitch_df['dblhead_num'].apply(lambda x: int(x) if str(x).strip().isdigit() else 0)

  # Convert IP to proper mathematical format
  ip = float(pitch_df['IP'][0])
  pitch_df['IP_real'] = (ip - (ip % 1)) + (ip % 1) * (10/3)
  
  er_per_ip_def = (5/9)
  h_bb_per_ip_def = 1.5
  h_bb_per_bf_def = .37
  so_per_bf_def = .2
  ip_per_game_def = 3
  bf_per_game_def = 12
  tb_bb_perc_def = .45
  fip_numer_per_ip_def = .124*13 + 1.5*3 - 2*.8
  fip_numer_per_bf_def = .03*13 + .37*3 - 2*.2
  cols_to_agg = ['IP_real', 'H','BFP', 'HR', 'R', 'ER', 'BB', 'IB', 'SO', 'SH', 'SF', 'WP', 'HBP', 'BK', 'x2B', 'x3B']

  # Team Bullpen Average Data
  bp_ip_per_game_def = 2
  bp_bf_per_game_def = 6
  bp_cols_to_agg = ['IP', 'H','BFP', 'HR', 'R',  'BB', 'SO',  'HBP', 'x2B', 'x3B']
  
  bp_cols_to_add = ['Bpen_IP', 'Bpen_BFP', 'Bpen_R', 'Bpen_H', 'Bpen_HR', 'Bpen_x2B',
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
    'Bpen_H_BB_perc_75'
  ]
  for col in bp_cols_to_add:
    pitch_df[col] = np.nan
  
  for winsize in WINDOWS:
    # Add Starter Data
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
      
    # Add Bullpen Data
    for raw_col in bp_cols_to_agg:
      col_agg = 'Bpen_'+raw_col
      new_colname = 'Bpen_rollsum_'+raw_col+'_'+str(winsize)        
      pitch_df[new_colname] = roll_column(pitch_df, col_agg, winsize)
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
    pitch_df[h_bb_col] = pitch_df[hit_col]+pitch_df[bb_col]
    pitch_df[xb_col] = pitch_df[double_col]+2*pitch_df[triple_col]+3*pitch_df[hr_col]
    pitch_df[tb_col] = pitch_df[hit_col]+pitch_df[xb_col]
    pitch_df[ip_mod_col] = np.maximum(pitch_df[ip_col], winsize*bp_ip_per_game_def)
    pitch_df[bf_mod_col] = np.maximum(pitch_df[bf_col], winsize*bp_bf_per_game_def)
    pitch_df[h_bb_mod_col] = pitch_df[h_bb_col] + h_bb_per_ip_def*(pitch_df[ip_mod_col]-pitch_df[ip_col])
    pitch_df[h_bb_mod2_col] = pitch_df[h_bb_col] + h_bb_per_bf_def*(pitch_df[bf_mod_col]-pitch_df[bf_col])
    pitch_df[so_mod_col] = pitch_df[so_col] + so_per_bf_def*(pitch_df[bf_mod_col]-pitch_df[bf_col])
    pitch_df[tb_bb_mod_col] = (pitch_df[tb_col] + pitch_df[bb_col])+ tb_bb_perc_def*(pitch_df[bf_mod_col]-pitch_df[bf_col])
    pitch_df[whip_col] = pitch_df[h_bb_mod_col]/pitch_df[ip_mod_col]
    pitch_df[so_perc_col] = pitch_df[so_mod_col]/pitch_df[bf_mod_col]
    pitch_df[tb_bb_perc_col] = pitch_df[tb_bb_mod_col]/pitch_df[bf_mod_col]
    pitch_df[h_bb_perc_col] = pitch_df[h_bb_mod2_col]/pitch_df[bf_mod_col]
  pitch_df.drop(['at_vs', 'Opponent', 'League'], axis=1, inplace=True)
  return pitch_df

