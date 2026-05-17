import webbrowser
import tempfile
import os
import pandas as pd

# League average thresholds
LEAGUE_AVG = {
    "Barrel%": 0.083,
    "EV": 89.1,
    "HARDHIT%": 0.397,
    "SWSPOT%": 0.334,
    "HR/PA": 0.038,
    "FB%": 0.261,
}


def _stat_color(col, val_str):
    """Color based on league average comparison."""
    if val_str == "N/A" or col not in LEAGUE_AVG:
        return "neutral"
    try:
        # EV uses absolute thresholds
        if col == "EV":
            val = float(str(val_str).replace(" mph", ""))
            if val >= 91.0:
                return "strong-pos"
            elif val >= 89.0:
                return "pos"
            elif val >= 87.0:
                return "yellow"
            else:
                return "neg"

        avg = LEAGUE_AVG[col]
        val = (
            float(str(val_str).replace("%", "").replace(" mph", "")) / 100
            if "%" in str(val_str)
            else float(str(val_str).replace(" mph", ""))
        )

        diff = (val - avg) / avg

        if diff >= 0.20:
            return "strong-pos"
        elif diff >= 0.05:
            return "pos"
        elif diff >= -0.10:
            return "yellow"
        else:
            return "neg"
    except Exception:
        return "neutral"


def _edge_color(edge_str):
    """Return color class based on edge value string like '-4.29%' or '+2.1%'."""
    if not edge_str or edge_str == "None" or pd.isna(edge_str):
        return "neutral"
    try:
        val = float(str(edge_str).replace("%", ""))
        if val >= 3:
            return "strong-pos"
        elif val >= 1:
            return "pos"
        elif val >= -2:
            return "neutral"
        else:
            return "neg"
    except Exception:
        return "neutral"


def _wind_color(wind_out_str):
    """Return color class based on wind out value like '-12.8' or '+5.2'."""
    if not wind_out_str or wind_out_str == "N/A":
        return "neutral"
    try:
        val = float(str(wind_out_str).replace("+", ""))
        if val >= 5:
            return "strong-pos"
        elif val >= 1:
            return "pos"
        elif val >= -3:
            return "neutral"
        else:
            return "neg"
    except Exception:
        return "neutral"


def _prob_color(prob_str):
    """Return color class based on HR probability."""
    if not prob_str or prob_str == "N/A":
        return "neutral"
    try:
        val = float(str(prob_str).replace("%", ""))
        if val >= 20:
            return "strong-pos"
        elif val >= 15:
            return "pos"
        else:
            return "neutral"
    except Exception:
        return "neutral"


def _build_table(df, colored_cols=None):
    """Build an HTML table with color coding."""
    colored_cols = colored_cols or {}
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            val = row[col] if pd.notna(row[col]) else "N/A"
            css_class = ""
            if col in colored_cols:
                color_fn = colored_cols[col]
                css_class = f' class="{color_fn(str(val))}"'
            cells += f"<td{css_class}>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"""
    <table>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def display_dashboard(hr_df, wins_df, hr_bets_df, run_date):
    """
    Generate and open a baseball predictions dashboard in the browser.

    hr_df       : formatted HR top predictions DataFrame
    wins_df     : formatted win predictions DataFrame
    hr_bets_df  : formatted HR bets DataFrame (edge plays only)
    run_date    : date string like '2026-05-16'
    """

    hr_colored = {
        "Edge": _edge_color,
        "Wind Out": _wind_color,
        "HR Prob": _prob_color,
        "Barrel%": lambda v: _stat_color("Barrel%", v),
        "EV": lambda v: _stat_color("EV", v),
        "HARDHIT%": lambda v: _stat_color("HARDHIT%", v),
        "SWSPOT%": lambda v: _stat_color("SWSPOT%", v),
        "HR/PA": lambda v: _stat_color("HR/PA", v),
        "FB%": lambda v: _stat_color("FB%", v),
        "Platoon": lambda v: _stat_color("HR/PA", v),  # reuse HR/PA thresholds
    }
    wins_colored = {
        "Edge (H)": _edge_color,
        "Edge (V)": _edge_color,
    }
    bets_colored = {
        "Edge": _edge_color,
        "Wind Out": _wind_color,
        "HR Prob": _prob_color,
        "Barrel%": lambda v: _stat_color("Barrel%", v),
        "EV": lambda v: _stat_color("EV", v),
        "HARDHIT%": lambda v: _stat_color("HARDHIT%", v),
        "SWSPOT%": lambda v: _stat_color("SWSPOT%", v),
        "HR/PA": lambda v: _stat_color("HR/PA", v),
        "FB%": lambda v: _stat_color("FB%", v),
        "Platoon": lambda v: _stat_color("HR/PA", v),  # reuse HR/PA thresholds
    }

    hr_table = _build_table(hr_df, hr_colored)
    wins_table = _build_table(wins_df, wins_colored)
    bets_table = (
        _build_table(hr_bets_df, bets_colored)
        if not hr_bets_df.empty
        else '<p class="no-bets">No qualifying HR bets today.</p>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoneyballVo Bets — {run_date}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500;700&display=swap');

  :root {{
      --bg:        #0d0f14;
      --surface:   #141720;
      --border:    #1e2330;
      --green:     #00e676;
      --green-dim: #00c853;
      --blue:      #40c4ff;
      --red:       #ff5252;
      --yellow:    #ffd740;
      --text:      #e0e6f0;
      --muted:     #5a6480;
      --header-bg: #0a0c10;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    min-height: 100vh;
    padding: 0 0 60px;
  }}

  header {{
    background: var(--header-bg);
    border-bottom: 1px solid var(--border);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 12px;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(8px);
  }}

  header .logo {{ font-size: 22px; }}

  header h1 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: var(--green);
    letter-spacing: 0.05em;
  }}

  header .date {{
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
  }}
  
  .header-right {{
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 16px;
  }}

  .x-link {{
      color: var(--muted);
      transition: color 0.2s;
      display: flex;
      align-items: center;
  }}

  .x-link:hover {{
      color: var(--text);
  }}

  .container {{ padding: 28px 32px; }}

  section {{ margin-bottom: 40px; }}

  .section-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  .badge {{
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 600;
    letter-spacing: 0.08em;
  }}

  .badge-hr  {{ background: rgba(64,196,255,0.12); color: var(--blue);   border: 1px solid rgba(64,196,255,0.25); }}
  .badge-win {{ background: rgba(255,215,64,0.12);  color: var(--yellow); border: 1px solid rgba(255,215,64,0.25); }}
  .badge-value {{ background: rgba(0,230,118,0.12);   color: var(--green);  border: 1px solid rgba(0,230,118,0.25); }}

  .table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
  }}

  thead tr {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }}

  th {{
    padding: 10px 14px;
    text-align: left;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    white-space: nowrap;
  }}

  td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    color: var(--text);
  }}

  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: rgba(255,255,255,0.02); }}

  /* color classes */
  td.strong-pos {{ color: var(--green); font-weight: 600; }}
  td.pos        {{ color: #69f0ae; }}
  td.neutral    {{ color: var(--text); }}
  td.neg        {{ color: var(--red); }}
  td.yellow     {{ color: var(--yellow); }}

  .no-bets {{
    padding: 20px;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    text-align: center;
    border: 1px solid var(--border);
    border-radius: 8px;
  }}

  .divider {{
    height: 1px;
    background: var(--border);
    margin: 40px 0;
  }}
</style>
</head>
<body>

<header>
  <span class="logo">⚾</span>
  <h1>MoneyballVo Bets</h1>
  <div class="header-right">
    <a href="https://x.com/MoneyballVo" target="_blank" class="x-link">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.748l7.73-8.835L1.254 2.25H8.08l4.259 5.63L18.244 2.25zm-1.161 17.52h1.833L7.084 4.126H5.117L17.083 19.77z"/>
      </svg>
    </a>
    <span class="date">{run_date}</span>
  </div>
</header>

<div class="container">

  <section>
    <div class="section-title">
      <span class="badge badge-value">VALUE</span>
      HR Edge Plays
    </div>
    <div class="table-wrap">
      {bets_table}
    </div>
  </section>

  <section>
    <div class="section-title">
      <span class="badge badge-hr">HR</span>
      Top Home Run Predictions
    </div>
    <div class="table-wrap">
      {hr_table}
    </div>
  </section>

  <div class="divider"></div>

  <section>
    <div class="section-title">
      <span class="badge badge-win">WIN</span>
      Game Winner Predictions
    </div>
    <div class="table-wrap">
      {wins_table}
    </div>
  </section>

</div>
</body>
</html>"""

    tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False)
    tmp.write(html)
    tmp.close()
    webbrowser.open(f"file://{tmp.name}")
    print(f"Dashboard opened: {tmp.name}")
