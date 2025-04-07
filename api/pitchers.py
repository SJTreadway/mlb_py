import re
import time
import os
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import requests

from tqdm import tqdm

from pybaseball import playerid_reverse_lookup

from helpers import roll_column, strip_suffix, get_team_league_map

URL_PREFIX = 'https://www.retrosheet.org/boxesetc/'
BREF_URL_PREFIX = 'https://www.baseball-reference.com/players/gl.fcgi'
YEAR = 2025
WINDOWS = [10,35,75]

def process_pitching_data(df):
  start_pitchers_h = [p for p in df.starting_pitcher_id_h.unique() if p is not None]
  start_pitchers_v = [p for p in df.starting_pitcher_id_v.unique() if p is not None]
  start_pitchers_all = np.union1d(start_pitchers_h, start_pitchers_v)

  # step 1: get pitching data for all starting pitchers and store to csv
  load_pitching_data(start_pitchers_all)

  # step 2: load data from files and store into dataframe
  strt_pitch_df = get_rolling_pitching_feats(df, start_pitchers_all)
  
  return get_bullpen_data(strt_pitch_df)

# Get data for each starting pitcher and store to csv
def load_pitching_data(start_pitchers_all):
  for p_id in tqdm(start_pitchers_all):
    if p_id:
      df_temp = get_full_pitching_data(p_id)
      df_season = get_bref_current_season_data(p_id)
      df_temp = pd.concat((df_temp, df_season))
      fname_out = 'data/pitch/pitching_data_'+p_id+'.csv'
      if not os.path.exists(fname_out):
        df_temp.to_csv(fname_out, index=False)

# Get all the data for a particular pitcher
def get_full_pitching_data(pitcher_id):
  if not pitcher_id:
    return pd.DataFrame()
  link_list = get_daily_season_links(pitcher_id)
  df_pitching = pd.DataFrame()
  for url in link_list:
    df_pitching = pd.concat((df_pitching, get_season_pitching_data(url)))
  return df_pitching

def get_bref_current_season_data(pid):
  time.sleep(1)
  bref_pid = None
  rev_lkp = playerid_reverse_lookup([pid], key_type='retro')
  if rev_lkp is not None:
    bref_pid = rev_lkp.loc[0]['key_bbref']
  url = BREF_URL_PREFIX+'?id='+bref_pid+'&t=p&year='+str(YEAR)
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  target_table = soup.find("table", id="pitching_gamelogs")
  if target_table is None:
    #print(f'Skipping pitcher {pid} ({bref_pid}) No table found. ')
    return pd.DataFrame()
  target_element = target_table.find('tbody')
  working_part = list(target_element.find_all('tr'))
  mod_header = ['at_vs','Opponent','League', 'GS', 'CG', 'SHO', 'GF', 'SV', 'IP', 'H',
      'BFP', 'HR', 'R', 'ER', 'BB', 'IB', 'SO', 'SH', 'SF', 'WP', 'HBP',
      'BK', 'x2B', 'x3B', 'GDP', 'ROE', 'W', 'L', 'ERA']
  bref_headers = ['team_homeORaway', 'opp_ID', 'player_game_span', 'CG', 'GF', 'SV', 'IP', 'H',
      'BF', 'HR', 'R', 'ER', 'BB', 'IBB', 'SO', 'SF', 'HBP',
      '2B', '3B', 'GDP', 'ROE', 'player_game_result', 'earned_run_avg']
  date_list = []
  dblhead_num_list = []
  for k in range(0, len(working_part)):
    td_cells = working_part[k].find_all("td", attrs={"data-stat": True})
    for d in range(0, len(td_cells)):
      if td_cells[d]["data-stat"] == "date_game":
        dat = td_cells[d]['csk'].split('.')[0]
        date_list.append(dat)
        dbl_head_num = ''.join(
            str(c) for c in td_cells[d].contents if not getattr(c, 'name', None) == 'a'
        ).strip()
        digit = re.sub(r'[()]', '', dbl_head_num)
        dblhead_num_list.append(str(digit) if digit else '')

  main_data_matrix = []
  matrix_to_convert = []
  for k in range(0, len(working_part)):
    td_cells = working_part[k].find_all("td", attrs={"data-stat": True})
    if (len(td_cells) > 0):
      data = {
        td["data-stat"]: td.get_text(strip=True)
        for td in td_cells
        if td["data-stat"] in bref_headers
      }
      matrix_to_convert.append(data)
    
  main_data_matrix = convert_header_values(matrix_to_convert)
  pitch_df = pd.DataFrame(main_data_matrix, columns = mod_header)
  pitch_df['date'] = date_list
  pitch_df['dblhead_num'] = dblhead_num_list
  return pitch_df

def convert_header_values(main_data_matrix):
  converted_matrix = []
  team_league_map = get_team_league_map()
  for row in main_data_matrix:
    opp = row.get('opp_ID', '')
    pgs = row.get('player_game_span', '').split('-')
    converted_matrix.append({
      'at_vs': 'AT' if row.get('team_homeORaway', '') == '@' else 'VS',
      'Opponent': opp,
      'League': team_league_map[opp],
      'GS': 1 if pgs[0] == 'GS' else 0,
      'CG': 1 if pgs[0] == 'CG' else 0,
      'SHO': 1 if pgs[0] == 'CG' and int(row.get('R', 0)) == 0 else 0,
      'GF': 1 if len(pgs) > 1 and pgs[1] == 'GF' else 0,
      'SV': 0,  # Placeholder; can be derived if needed
      'IP': row.get('IP', 0),
      'H': row.get('H', 0),
      'BFP': row.get('BF', 0) if 'BF' in row else int(row.get('batters_faced', 0)),
      'HR': row.get('HR', 0),
      'R': row.get('R', 0),
      'ER': row.get('ER', 0),
      'BB': row.get('BB', 0),
      'IB': row.get('IBB', 0),
      'SO': row.get('SO', 0),
      'SH': row.get('SF', 0),  # Reuse SF for now
      'SF': row.get('SF', 0),
      'WP': 0,
      'HBP': row.get('HBP', 0),
      'BK': 0,
      'x2B': row.get('2B', 0),
      'x3B': row.get('3B', 0),
      'GDP': row.get('GIDP', 0),
      'ROE': row.get('ROE', 0),
      'W': 1 if row.get('player_game_result', '').split('(')[0] == 'W' else 0,
      'L': 1 if row.get('player_game_result', '').split('(')[0] == 'L' else 0,
      'ERA': row.get('earned_run_avg', 0.0)
    })
  return converted_matrix
 

### Get the links to the pitcher-season tables given the pitcher id
def get_daily_season_links(pitcher_id):
  letter = pitcher_id.upper()[0]
  url = URL_PREFIX+letter+'/P'+pitcher_id+'.htm'
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
  return pitch_df
  
def load_and_process_pitch_df(p_id, filepath=''):
  if not p_id:
    return pd.DataFrame()

  fname = filepath+'pitching_data_'+p_id+'.csv'
  pitch_df = pd.read_csv(fname)
  
  # Convert date, fix dblhead_num to be 0,1,2
  pitch_df['date'] = (
      pd.to_datetime(pitch_df['date'], format='mixed', errors='coerce')
        .dt.strftime('%Y%m%d')
        .astype(int)
  )
  pitch_df.dblhead_num.fillna(0, inplace=True)
  pitch_df['dblhead_num'] = pitch_df['dblhead_num'].astype(int)
  
  # Convert IP to proper mathematical format
  pitch_df['IP_real'] = (pitch_df.IP - (pitch_df.IP % 1)) + (pitch_df.IP % 1) * (10/3)
  
  cols_to_agg = ['IP_real', 'H','BFP', 'HR', 'R', 'ER', 'BB', 'IB', 'SO', 'SH', 'SF', 'WP', 'HBP', 'BK',
      'x2B', 'x3B']
  for winsize in WINDOWS:
    for raw_col in cols_to_agg:
      new_colname = 'rollsum_'+raw_col+'_'+str(winsize)        
      pitch_df[new_colname] = roll_column(pitch_df, raw_col, winsize)
    
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
  
  for col in cols_to_add:
    df[col] = col_add_dict[col]
  
  return df
  
def get_bullpen_team_df(team, df):
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
  
  df_team = df_team_h if not df_team_h.empty else df_team_v
  df_team.sort_values(['date_dblhead'],inplace=True)
  
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
  for winsize in winsizes:
    for raw_col in cols_to_agg:
      col_agg = 'Bpen_'+raw_col
      new_colname = 'Bpen_rollsum_'+raw_col+'_'+str(winsize)        
      df_team[new_colname] = roll_column(df_team, col_agg, winsize)

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
    df_team[h_bb_col] = df_team[hit_col]+df_team[bb_col]
    df_team[xb_col] = df_team[double_col]+2*df_team[triple_col]+3*df_team[hr_col]
    df_team[tb_col] = df_team[hit_col]+df_team[xb_col]
    df_team[ip_mod_col] = np.maximum(df_team[ip_col], winsize*ip_per_game_def)
    df_team[bf_mod_col] = np.maximum(df_team[bf_col], winsize*bf_per_game_def)
    df_team[h_bb_mod_col] = df_team[h_bb_col] + h_bb_per_ip_def*(df_team[ip_mod_col]-df_team[ip_col])
    df_team[h_bb_mod2_col] = df_team[h_bb_col] + h_bb_per_bf_def*(df_team[bf_mod_col]-df_team[bf_col])
    df_team[so_mod_col] = df_team[so_col] + so_per_bf_def*(df_team[bf_mod_col]-df_team[bf_col])
    df_team[tb_bb_mod_col] = (df_team[tb_col] + df_team[bb_col])+ tb_bb_perc_def*(df_team[bf_mod_col]-df_team[bf_col])
    df_team[whip_col] = df_team[h_bb_mod_col]/df_team[ip_mod_col]
    df_team[so_perc_col] = df_team[so_mod_col]/df_team[bf_mod_col]
    df_team[tb_bb_perc_col] = df_team[tb_bb_mod_col]/df_team[bf_mod_col]
    df_team[h_bb_perc_col] = df_team[h_bb_mod2_col]/df_team[bf_mod_col]

  df_team.set_index('date_dblhead', inplace=True)
  return df_team
  
def get_bullpen_data(df):
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
  
  teams = df[['team_h', 'team_v']].stack().unique().tolist()
  bullpen_team_data_dict = {}
  for team in teams:
    bullpen_team_data_dict[team] = get_bullpen_team_df(team, df)
    
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
  
  for i in range(df.shape[0]):
    row = df.iloc[i,:]
    home_team = row['team_h']
    visit_team = row['team_v']
    date_dblhead = row['date_dblhead']
    curr_df = bullpen_team_data_dict[home_team]
    if date_dblhead in curr_df.index:
      for col in raw_cols_to_add:
        value = curr_df.loc[date_dblhead,col]
        col_add_dict[col+'_h'][i] = value
    curr_df = bullpen_team_data_dict[visit_team]
    if date_dblhead in curr_df.index:
      for col in raw_cols_to_add:
        value = curr_df.loc[date_dblhead,col]
        col_add_dict[col+'_v'][i] = value
  
  for col in cols_to_add:
    df[col] = col_add_dict[col]

  return df
