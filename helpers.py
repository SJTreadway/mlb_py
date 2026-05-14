import numpy as np

def roll_column(df, col, winsize):
  # do the standard Pandas rolling calc
  t_col = df[col].rolling(winsize, closed='left').sum().to_numpy()
  # for the early columns, just do a rolling sum from the beginning
  t_col[:winsize] = np.concatenate(([0],df[col].iloc[:(winsize)].cumsum().to_numpy()[:-1]))
  return t_col

def agg_non_na(series):
  return series.dropna().iloc[0] if not series.dropna().empty else None

# strip away suffix, e.g., '_h', '_v', for given column
def strip_suffix(col, suffix):
  return col[:-len(suffix)] if col.endswith(suffix) else col

def safe_int(s):
  try:
    return int(s)
  except ValueError:
    return 0
  
def safe_float(s):
  try:
    return float(s)
  except ValueError:
    return 0

def get_team_league_map():
   return {
   # AL East
   'BAL': 'A', 'BOS': 'A', 'NYY': 'A', 'TB': 'A', 'TBR': 'A', 'TOR': 'A',
   # AL Central
   'CWS': 'A', 'CHW': 'A', 'CLE': 'A', 'DET': 'A', 'KC': 'A', 'KCR': 'A', 'MIN': 'A',
   # AL West
   'HOU': 'A', 'LAA': 'A', 'SEA': 'A', 'TEX': 'A', 'OAK': 'A', 'ATH': 'A',
   
   # NL East
   'ATL': 'N', 'MIA': 'N', 'NYM': 'N', 'PHI': 'N', 'WSN': 'N', 'WAS': 'N',
   # NL Central
   'CHC': 'N', 'CIN': 'N', 'MIL': 'N', 'PIT': 'N', 'STL': 'N',
   # NL West
   'ARI': 'N', 'AZ': 'N', 'COL': 'N', 'LAD': 'N', 'SDP': 'N', 'SD': 'N', 'SFG': 'N', 'SF': 'N'
}