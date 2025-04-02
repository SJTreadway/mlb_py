import numpy as np

def roll_column(df, col, winsize):
  try:
    # do the standard Pandas rolling calc
    t_col = df[col].rolling(winsize, closed='left').sum().to_numpy()
    # for the early columns, just do a rolling sum from the beginning
    t_col[:winsize] = np.concatenate(([0],df[col].iloc[:(winsize)].cumsum().to_numpy()[:-1]))
    return t_col
  except:
    print(f'error: {col, df[col]}')

def agg_non_na(series):
  return series.dropna().iloc[0] if not series.dropna().empty else None

# strip away suffix, e.g., '_h', '_v', for given column
def strip_suffix(col, suffix):
  return col[:-len(suffix)] if col.endswith(suffix) else col