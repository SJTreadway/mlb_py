"""
dashboard.py  —  Streamlit version
───────────────────────────────────
Run standalone:
    streamlit run ui/dashboard.py

Or call display_dashboard() from pipeline.py when running:
    streamlit run pipeline.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from datetime import datetime


# ── theme / global CSS ─────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500;700&display=swap');

:root {
    --green:      #00e676;
    --green-dim:  #00c853;
    --blue:       #40c4ff;
    --red:        #ff5252;
    --yellow:     #ffd740;
    --muted:      #5a6480;
    --border:     #1e2330;
    --surface:    #141720;
    --text:       #e0e6f0;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* typography */
body, .stMarkdown, td, th {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px;
}

/* tables */
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

/* section badges */
.badge {
    font-size: 10px; padding: 2px 8px; border-radius: 3px;
    font-weight: 600; letter-spacing: 0.08em;
}
.badge-hr    { background: rgba(64,196,255,0.12); color: var(--blue);  border: 1px solid rgba(64,196,255,0.25); }
.badge-value { background: rgba(0,230,118,0.12);  color: var(--green); border: 1px solid rgba(0,230,118,0.25); }

.table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
}

.divider { height: 1px; background: var(--border); margin: 32px 0; }
.empty-state {
    padding: 32px; text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: var(--muted);
}
</style>
"""

# ── league average thresholds ──────────────────────────────────────────────────
LEAGUE_AVG = {
    "Barrel%": 0.083,
    "EV": 89.1,
    "HARDHIT%": 0.397,
    "SWSPOT%": 0.334,
    "HR/PA": 0.038,
    "FB%": 0.261,
    "P-HR/BF": 0.032,  # pitcher HR allowed per batter faced
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

        # For P-HR/BF: higher = worse pitcher = better for batter (invert scale)
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


# ── HTML table builder ─────────────────────────────────────────────────────────


def _format_date(val: str) -> str:
    try:
        s = str(int(val))
        date_str = s[:-1]
        return datetime.strptime(date_str, "%Y%m%d").strftime("%m/%d/%Y")
    except Exception:
        return str(val)


def _build_table(df: pd.DataFrame, colored_cols: dict | None = None) -> str:
    colored_cols = colored_cols or {}
    rows_html = ""
    HIDDEN_COLS = {"Temp", "Humidity"}
    visible_cols = [c for c in df.columns if c not in HIDDEN_COLS]
    for _, row in df.iterrows():
        cells = ""
        for col in visible_cols:
            val = row[col] if pd.notna(row[col]) else "N/A"
            if col == "Team":
                logo = TEAM_LOGOS.get(str(row.get("Team", "")), "")
                logo_html = (
                    f'<img src="{logo}" style="height:24px;width:24px;'
                    f"vertical-align:middle;margin-right:6px;"
                    f'background:white;border-radius:50%;padding:2px;">'
                    if logo
                    else ""
                )
                cells += f"<td>{logo_html}</td>"
            elif col == "Date":
                cells += f"<td>{_format_date(val)}</td>"
            else:
                css = ""
                if col in colored_cols:
                    css_class = colored_cols[col](str(val))
                    css = f' class="{css_class}"'
                cells += f"<td{css}>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in visible_cols)
    return (
        f'<div class="table-wrap">'
        f'<table class="dash-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>"
    )


def _empty_state(msg: str) -> str:
    return f'<div class="empty-state">{msg}</div>'


# ── main display function ──────────────────────────────────────────────────────


def display_dashboard(
    hr_df: pd.DataFrame,
    wins_df: pd.DataFrame,
    run_date: str,
) -> None:
    st.set_page_config(
        page_title="MoneyballVo | MLB Analytics",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    # ── header ─────────────────────────────────────────────────────────────────
    col_logo, col_title, col_right = st.columns([0.04, 0.61, 0.35])

    with col_logo:
        st.markdown(
            '<span style="font-size:24px;line-height:1;padding-top:6px;display:block;">⚾</span>',
            unsafe_allow_html=True,
        )

    with col_title:
        st.markdown(
            "<h1 style=\"font-family:'IBM Plex Mono',monospace;font-size:18px;"
            "font-weight:600;color:#00e676;letter-spacing:0.05em;margin:0;"
            'padding-top:6px;margin-left:-16px;">MoneyballVo | MLB Analytics</h1>',
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f'<div style="text-align:right;padding-top:6px;">'
            f'<div style="display:flex;align-items:center;justify-content:flex-end;gap:10px;margin-bottom:2px;">'
            f'<a href="https://x.com/MoneyballVo" target="_blank" style="color:#5a6480;line-height:1;">'
            f'<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">'
            f'<path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.748'
            f"l7.73-8.835L1.254 2.25H8.08l4.259 5.63L18.244 2.25zm-1.161 17.52h1.833L7.084 4.126H5.117"
            f'L17.083 19.77z"/></svg></a>'
            f"<span style=\"font-family:'IBM Plex Mono',monospace;font-size:12px;color:#5a6480;\">{run_date}</span>"
            f"</div>"
            f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:10px;color:#5a6480;opacity:0.6;\">"
            f'Updated {datetime.now().strftime("%-I:%M %p CST")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── HR predictions ─────────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:14px;">'
        '<span class="badge badge-hr">HR</span>'
        "<span style=\"font-family:'IBM Plex Mono',monospace;font-size:11px;"
        "font-weight:600;letter-spacing:0.12em;text-transform:uppercase;"
        'color:#5a6480;margin-left:10px;">Top Home Run Predictions</span>'
        "</div>",
        unsafe_allow_html=True,
    )

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

    if isinstance(hr_df, pd.DataFrame) and not hr_df.empty:
        st.markdown(_build_table(hr_df, hr_colored), unsafe_allow_html=True)
    else:
        st.markdown(
            _empty_state("No qualifying HR predictions yet"), unsafe_allow_html=True
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── win predictions ────────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:14px;">'
        '<span class="badge badge-value">VALUE</span>'
        "<span style=\"font-family:'IBM Plex Mono',monospace;font-size:11px;"
        "font-weight:600;letter-spacing:0.12em;text-transform:uppercase;"
        'color:#5a6480;margin-left:10px;">Game Winner Predictions</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    wins_colored = {
        "Edge (H)": _edge_color,
        "Edge (V)": _edge_color,
    }

    if isinstance(wins_df, pd.DataFrame) and not wins_df.empty:
        st.markdown(_build_table(wins_df, wins_colored), unsafe_allow_html=True)
    else:
        st.markdown(
            _empty_state("No qualifying win predictions yet"), unsafe_allow_html=True
        )
