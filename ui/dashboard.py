"""
dashboard.py  —  Static HTML generator
───────────────────────────────────────
Generates a self-contained HTML file instead of running a Streamlit server.

Usage:
    from ui.dashboard import generate_html
    html = generate_html(hr_df, wins_df, run_date, warnings=[...])
    with open("dashboard.html", "w") as f:
        f.write(html)
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime

# ── league average thresholds ──────────────────────────────────────────────────
LEAGUE_AVG = {
    "Barrel%": 0.083,
    "EV": 89.1,
    "HARDHIT%": 0.397,
    "SWSPOT%": 0.334,
    "HR/PA": 0.038,
    "FB%": 0.261,
    "P-HR/BF": 0.032,
}

TEAM_LOGOS = {
    "ARI": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "AZ": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/mlb/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/mlb/500/bal.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/mlb/500/bos.png",
    "CHC": "https://a.espncdn.com/i/teamlogos/mlb/500/chc.png",
    "CHW": "https://a.espncdn.com/i/teamlogos/mlb/500/chw.png",
    "CWS": "https://a.espncdn.com/i/teamlogos/mlb/500/chw.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/mlb/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/mlb/500/cle.png",
    "COL": "https://a.espncdn.com/i/teamlogos/mlb/500/col.png",
    "DET": "https://a.espncdn.com/i/teamlogos/mlb/500/det.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/mlb/500/hou.png",
    "KCR": "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "KC": "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "LAA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",
    "LAD": "https://a.espncdn.com/i/teamlogos/mlb/500/lad.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/mlb/500/mia.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/mlb/500/mil.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/mlb/500/min.png",
    "NYM": "https://a.espncdn.com/i/teamlogos/mlb/500/nym.png",
    "NYY": "https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png",
    "ATH": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "OAK": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "LAS": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/mlb/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/mlb/500/pit.png",
    "SDP": "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "SD": "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/mlb/500/sea.png",
    "SFG": "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "SF": "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "STL": "https://a.espncdn.com/i/teamlogos/mlb/500/stl.png",
    "TBR": "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "TB": "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "TEX": "https://a.espncdn.com/i/teamlogos/mlb/500/tex.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/mlb/500/tor.png",
    "WSH": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png",
    "WSN": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --green:      #00e676;
    --green-dim:  #00c853;
    --blue:       #40c4ff;
    --red:        #ff5252;
    --yellow:     #ffd740;
    --muted:      #5a6480;
    --border:     #1e2330;
    --surface:    #141720;
    --bg:         #0d0f17;
    --text:       #e0e6f0;
}

html, body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    min-height: 100vh;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 32px 48px;
}

/* warnings */
.warning {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: rgba(255,215,64,0.06);
    border: 1px solid rgba(255,215,64,0.2);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 11px;
    color: #ffd740;
}
.warning-icon { flex-shrink: 0; margin-top: 1px; }

/* header */
.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
}
.header-left { display: flex; align-items: center; gap: 10px; }
.header-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--green);
    letter-spacing: 0.05em;
}
.header-right { text-align: right; }
.header-date { font-size: 12px; color: var(--muted); }
.header-updated { font-size: 10px; color: var(--muted); opacity: 0.6; margin-top: 2px; }
.x-icon { color: var(--muted); text-decoration: none; }
.x-icon:hover { color: var(--text); }
.header-meta { display: flex; align-items: center; justify-content: flex-end; gap: 10px; margin-bottom: 2px; }

/* divider */
.divider { height: 1px; background: var(--border); margin: 32px 0; }

/* section label */
.section-label { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.badge {
    font-size: 10px; padding: 2px 8px; border-radius: 3px;
    font-weight: 600; letter-spacing: 0.08em;
}
.badge-hr    { background: rgba(64,196,255,0.12); color: var(--blue);  border: 1px solid rgba(64,196,255,0.25); }
.badge-value { background: rgba(0,230,118,0.12);  color: var(--green); border: 1px solid rgba(0,230,118,0.25); }
.section-title {
    font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
}

/* tables */
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
.dash-table { width: 100%; border-collapse: collapse; }
.dash-table thead tr { border-bottom: 1px solid var(--border); }
.dash-table th {
    padding: 8px 10px;
    text-align: left;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    white-space: nowrap;
    background: var(--surface);
}
.dash-table td {
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    color: var(--text);
}
.dash-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.dash-table tbody tr:last-child td { border-bottom: none; }

/* color classes */
.strong-pos { color: var(--green)  !important; font-weight: 600; }
.pos        { color: #69f0ae       !important; }
.neutral    { color: var(--text)   !important; }
.neg        { color: var(--red)    !important; }
.yellow     { color: var(--yellow) !important; }

.empty-state {
    padding: 32px; text-align: center;
    font-size: 12px; color: var(--muted);
    border: 1px solid var(--border); border-radius: 8px;
}
</style>
"""

# ── color helpers ──────────────────────────────────────────────────────────────


def _stat_color(col: str, val_str: str) -> str:
    if val_str == "N/A" or col not in LEAGUE_AVG:
        return "neutral"
    try:
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

        if col == "P-HR/BF":
            diff = (val - avg) / avg
            if diff >= 0.20:
                return "strong-pos"
            elif diff >= 0.05:
                return "pos"
            elif diff >= -0.10:
                return "yellow"
            else:
                return "neg"

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


def _edge_color(val_str: str) -> str:
    if not val_str or val_str == "None" or pd.isna(val_str):
        return "neutral"
    try:
        val = float(str(val_str).replace("%", ""))
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


def _wind_color(val_str: str) -> str:
    if not val_str or val_str == "N/A":
        return "neutral"
    try:
        val = float(str(val_str).replace("+", ""))
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


def _prob_color(val_str: str) -> str:
    if not val_str or val_str == "N/A":
        return "neutral"
    try:
        val = float(str(val_str).replace("%", ""))
        if val >= 20:
            return "strong-pos"
        elif val >= 15:
            return "pos"
        else:
            return "neutral"
    except Exception:
        return "neutral"


# ── table builder ──────────────────────────────────────────────────────────────


def _format_date(val: str) -> str:
    try:
        s = str(int(val))
        date_str = s[:-1]
        return datetime.strptime(date_str, "%Y%m%d").strftime("%m/%d/%Y")
    except Exception:
        return str(val)


def _build_table(df: pd.DataFrame, colored_cols: dict | None = None) -> str:
    colored_cols = colored_cols or {}
    HIDDEN_COLS = {"Temp", "Humidity"}
    visible_cols = [c for c in df.columns if c not in HIDDEN_COLS]

    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in visible_cols:
            val = row[col] if pd.notna(row[col]) else "N/A"
            if col == "Team":
                logo = TEAM_LOGOS.get(str(row.get("Team", "")), "")
                logo_html = (
                    f'<img src="{logo}" style="height:28px;width:28px;'
                    f'background:white;border-radius:30%;padding:4px">'
                    if logo
                    else ""
                )
                cells += f"<td>{logo_html}</td>"
            elif col == "Date":
                cells += f"<td>{_format_date(val)}</td>"
            else:
                css_class = colored_cols[col](str(val)) if col in colored_cols else ""
                css_attr = f' class="{css_class}"' if css_class else ""
                cells += f"<td{css_attr}>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in visible_cols)
    return (
        '<div class="table-wrap">'
        '<table class="dash-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )


# ── main generator ─────────────────────────────────────────────────────────────


def generate_html(
    hr_df: pd.DataFrame,
    wins_df: pd.DataFrame,
    run_date: str,
    warnings: list[str] | None = None,
) -> str:
    warnings = warnings or []
    updated = datetime.now().strftime("%-I:%M %p CST")

    # warnings
    warnings_html = ""
    for w in warnings:
        warnings_html += (
            f'<div class="warning">'
            f'<span class="warning-icon">⚠️</span>'
            f"<span>{w}</span>"
            f"</div>"
        )

    # HR table
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
        "P-HR/BF": lambda v: _stat_color("P-HR/BF", v),
        "Platoon": lambda v: _stat_color("HR/PA", v),
    }
    hr_section = (
        _build_table(hr_df, hr_colored)
        if isinstance(hr_df, pd.DataFrame) and not hr_df.empty
        else '<div class="empty-state">No qualifying HR predictions yet</div>'
    )

    # wins table
    wins_colored = {
        "Edge (H)": _edge_color,
        "Edge (V)": _edge_color,
    }
    wins_section = (
        _build_table(wins_df, wins_colored)
        if isinstance(wins_df, pd.DataFrame) and not wins_df.empty
        else '<div class="empty-state">No qualifying win predictions yet</div>'
    )

    x_icon = (
        '<a href="https://x.com/MoneyballVo" target="_blank" class="x-icon">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231'
        "-5.401 6.231H2.748l7.73-8.835L1.254 2.25H8.08l4.259 5.63L18.244 2.25z"
        'm-1.161 17.52h1.833L7.084 4.126H5.117L17.083 19.77z"/>'
        "</svg></a>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚾️ MoneyballVo | MLB Analytics</title>
{CSS}
</head>
<body>
<div class="container">

  {warnings_html}

  <div class="header">
    <div class="header-left">
      <span style="font-size:24px">⚾</span>
      <span class="header-title">@MoneyballVo | MLB Analytics</span>
    </div>
    <div class="header-right">
      <div class="header-meta">
        {x_icon}
        <span class="header-date">{run_date}</span>
      </div>
      <div class="header-updated">Updated {updated}</div>
    </div>
  </div>

  <div class="section-label">
    <span class="badge badge-hr">HR</span>
    <span class="section-title">Top Home Run Predictions</span>
  </div>
  {hr_section}

  <div class="divider"></div>

  <div class="section-label">
    <span class="badge badge-value">VALUE</span>
    <span class="section-title">Game Winner Predictions</span>
  </div>
  {wins_section}

</div>
</body>
</html>"""
