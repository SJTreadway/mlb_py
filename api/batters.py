import pandas as pd
import numpy as np
import requests
import time

from bs4 import BeautifulSoup

URL_PREFIX = 'https://www.retrosheet.org/boxesetc/'

# Get all the data for a particular batter
def get_full_batting_data(batter_id, order_id, suffix):
  if not batter_id:
    return pd.DataFrame()
  link_list = get_daily_season_links(batter_id)
  df_batting = pd.DataFrame()
  for url in link_list:
    df_batting = pd.concat((df_batting, get_season_batting_data(url)))
  #df_batting[f'batter{order_id}_id{suffix}'] = batter_id
  return df_batting

def get_daily_season_links(batter_id):
  letter = batter_id.upper()[0]
  url = URL_PREFIX+letter+'/P'+batter_id+'.htm'
  #time.sleep(.1)
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
    out_df = pd.DataFrame(main_data_matrix, columns = mod_header)
  elif (min_row_size == max_row_size) and (max_row_size==26):
    # Everything has 26 columns, will guess position is missing
    out_df = pd.DataFrame(main_data_matrix, columns = mod_header[:26])
    out_df['Pos'] = ''
  elif (min_row_size == 26) and (max_row_size==27):
    # Guessing position is missing for some rows but not others
    main_data_matrix = [x if len(x)==27 else x+[''] for x in main_data_matrix]
    out_df = pd.DataFrame(main_data_matrix, columns = mod_header)
  else:
    print('finding rows with less than 26 or more than 27 entries - Returning None')
    return(None)
  out_df.drop(['at_vs', 'Opponent', 'League', 'Pos'], axis=1, inplace=True)
  #out_df['date'] = date_list
  #out_df['dblhead_num'] = dblhead_num_list
  return out_df
  