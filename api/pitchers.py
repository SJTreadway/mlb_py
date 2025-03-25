import numpy as np
import pandas as pd

import time

from bs4 import BeautifulSoup
import requests

URL_PREFIX = 'https://www.retrosheet.org/boxesetc/'

# Get all the data for a particular pitcher
def get_full_pitching_data(pitcher_id, suffix):
  if not pitcher_id:
    return pd.DataFrame()
  link_list = get_daily_season_links(pitcher_id)
  df_pitching = pd.DataFrame()
  for url in link_list:
    df_pitching = pd.concat((df_pitching, get_season_pitching_data(url)))
  #df_pitching[f'starting_pitcher_id{suffix}'] = pitcher_id
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

  out_df = pd.DataFrame(main_data_matrix, columns = mod_header)
  #out_df['date'] = date_list
  #out_df['dblhead_num'] = dblhead_num_list
  out_df.drop(['at_vs', 'Opponent', 'League'], axis=1, inplace=True)
  return out_df

