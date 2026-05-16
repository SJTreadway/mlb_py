import requests
import pandas as pd
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
    "CHW": (41.8300, -87.6338),
    "CWS": (41.8300, -87.6338),
    "MIN": (44.9817, -93.2776),
    "TEX": (32.7473, -97.0828),
    "LAA": (33.8003, -117.8827),
    "ATH": (37.7516, -122.2005),
    "OAK": (37.7516, -122.2005),
    "LAS": (37.7516, -122.2005),
    "SEA": (47.5914, -122.3325),
    "SDP": (32.7076, -117.1570),
    "ARI": (33.4455, -112.0667),
}

# Domes get fixed values
DOME_TEAMS = {"TBR", "HOU", "ARI", "MIA", "SEA", "TOR", "MIN"}
DOME_TEMP = 72
DOME_HUMIDITY = 50


def get_weather_for_game(game_date, lat, lon):
    """Fetch temp and humidity for a game. Works for past dates and forecast."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m",
        "temperature_unit": "fahrenheit",
        "start_date": game_date,
        "end_date": game_date,
        "timezone": "America/New_York",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Use early evening hour (6pm local = index 18) as game time proxy
    temp = data["hourly"]["temperature_2m"][18]
    humidity = data["hourly"]["relativehumidity_2m"][18]
    return temp, humidity


def process_weather_data(df):
    temps, humidities = [], []
    today = date.today().strftime("%Y-%m-%d")
    for _, row in df.iterrows():
        home_team = row["team_h"]
        if home_team in DOME_TEAMS:
            temps.append(DOME_TEMP)
            humidities.append(DOME_HUMIDITY)
        else:
            lat, lon = STADIUM_COORDS[home_team]
            temp, humidity = get_weather_for_game(today, lat, lon)
            temps.append(temp)
            humidities.append(humidity)
    df["temp"] = temps
    df["humidity"] = humidities
    return df
