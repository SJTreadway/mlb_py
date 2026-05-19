import requests
import pandas as pd
import numpy as np
from datetime import date

STADIUM_COORDS = {
    "COL": (39.7559, -104.9942),
    "CHC": (41.9484, -87.6553),
    "BOS": (42.3467, -71.0972),
    "NYY": (40.8296, -73.9262),
    "LAD": (34.0739, -118.2400),
    "SFG": (37.7786, -122.3893),
    "HOU": (29.7572, -95.3555),
    "ATL": (33.8908, -84.4678),
    "PHI": (39.9061, -75.1665),
    "STL": (38.6226, -90.1928),
    "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),
    "DET": (42.3390, -83.0485),
    "CLE": (41.4962, -81.6852),
    "CIN": (39.0979, -84.5082),
    "PIT": (40.4469, -80.0057),
    "MIA": (25.7781, -80.2197),
    "WSH": (38.8730, -77.0074),
    "WSN": (38.8730, -77.0074),
    "NYM": (40.7571, -73.8458),
    "TBR": (27.7682, -82.6534),
    "TB": (27.7682, -82.6534),
    "BAL": (39.2838, -76.6218),
    "TOR": (43.6414, -79.3894),
    "KCR": (39.0517, -94.4803),
    "KC": (39.0517, -94.4803),
    "CHW": (41.8300, -87.6338),
    "CWS": (41.8300, -87.6338),
    "TEX": (32.7473, -97.0828),
    "LAA": (33.8003, -117.8827),
    "ATH": (37.7516, -122.2005),
    "OAK": (37.7516, -122.2005),
    "LAS": (36.0800, -115.1522),  # Las Vegas
    "SEA": (47.5914, -122.3325),
    "SDP": (32.7076, -117.1570),
    "SD": (32.7076, -117.1570),
    "ARI": (33.4455, -112.0667),
}

# Domes get fixed values
DOME_TEAMS = {"TBR", "TB", "HOU", "MIA", "TOR"}
DOME_TEMP = 72
DOME_HUMIDITY = 50
DOME_WIND = 0
DOME_WIND_DIR = 0

# Stadium orientation — angle in degrees where wind blows OUT to CF
# 0 = North, 90 = East, 180 = South, 270 = West
# Wind blowing OUT = wind direction matches or close to CF orientation
STADIUM_CF_BEARING = {
    "COL": 292,
    "CHC": 180,
    "BOS": 95,
    "NYY": 220,
    "LAD": 25,
    "SFG": 285,
    "ATL": 10,
    "PHI": 340,
    "STL": 340,
    "MIL": 220,
    "DET": 170,
    "CLE": 210,
    "CIN": 340,
    "PIT": 320,
    "WSH": 130,
    "WSN": 130,
    "NYM": 340,
    "BAL": 90,
    "KCR": 0,
    "KC": 0,
    "CHW": 10,
    "CWS": 10,
    "TEX": 335,
    "LAA": 5,
    "ATH": 290,
    "OAK": 290,
    "SDP": 310,
    "SD": 310,
    "LAD": 25,
}


def get_weather_for_game(game_date, lat, lon):
    """Fetch temp, humidity, wind speed and direction for a game."""
    today = date.today().strftime("%Y-%m-%d")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        if game_date < today
        else "https://api.open-meteo.com/v1/forecast"
    )
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "start_date": game_date,
        "end_date": game_date,
        "timezone": "America/New_York",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # index 18 = 6pm local as game time proxy
    temp = data["hourly"]["temperature_2m"][18]
    humidity = data["hourly"]["relativehumidity_2m"][18]
    wind_spd = data["hourly"]["windspeed_10m"][18]
    wind_dir = data["hourly"]["winddirection_10m"][18]

    return temp, humidity, wind_spd, wind_dir


def compute_wind_out(wind_spd, wind_dir, cf_bearing):
    """
    Compute effective wind blowing OUT toward CF.
    Positive = wind blowing out (helps HRs)
    Negative = wind blowing in (hurts HRs)
    """
    if wind_spd == 0:
        return 0.0
    # angle difference between wind direction and CF bearing
    angle_diff = abs(wind_dir - cf_bearing) % 360
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    # cos(0) = 1 (directly out), cos(180) = -1 (directly in)
    wind_out = wind_spd * np.cos(np.radians(angle_diff))
    return round(wind_out, 2)


def process_weather_data(df, run_date):
    temps, humidities, wind_spds, wind_outs = [], [], [], []
    today = run_date.strftime("%Y-%m-%d") or date.today().strftime("%Y-%m-%d")

    for _, row in df.iterrows():
        home_team = row["team_h"]
        if home_team in DOME_TEAMS:
            temps.append(DOME_TEMP)
            humidities.append(DOME_HUMIDITY)
            wind_spds.append(DOME_WIND)
            wind_outs.append(DOME_WIND)
        else:
            coords = STADIUM_COORDS.get(home_team)
            if not coords:
                temps.append(72)
                humidities.append(50)
                wind_spds.append(0)
                wind_outs.append(0)
                continue

            lat, lon = coords
            try:
                temp, humidity, wind_spd, wind_dir = get_weather_for_game(
                    today, lat, lon
                )
                cf_bearing = STADIUM_CF_BEARING.get(home_team, 0)
                wind_out = compute_wind_out(wind_spd, wind_dir, cf_bearing)
                temps.append(temp)
                humidities.append(humidity)
                wind_spds.append(wind_spd)
                wind_outs.append(wind_out)
            except Exception as e:
                print(f"Weather error for {home_team}: {e}")
                temps.append(72)
                humidities.append(50)
                wind_spds.append(0)
                wind_outs.append(0)

    df["temp"] = temps
    df["humidity"] = humidities
    df["wind_spd"] = wind_spds
    df["wind_out"] = wind_outs  # key feature: positive = blowing out
    return df
