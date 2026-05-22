import argparse
import os
import pickle
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
    "SF": (37.7786, -122.3893),
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
    "AZ": (33.4455, -112.0667),
}

# Domes get fixed values — no wind/temp effect
DOME_TEAMS = {"TBR", "TB", "HOU", "MIA", "TOR", "ARI", "AZ"}
DOME_TEMP = 72
DOME_HUMIDITY = 50
DOME_WIND = 0
DOME_WIND_DIR = 0

# Stadium CF bearing — compass degrees pointing FROM home plate TOWARD CF
# Wind blowing in that direction = blowing OUT (positive wind_out = helps HRs)
# 0 = North, 90 = East, 180 = South, 270 = West
STADIUM_CF_BEARING = {
    "COL": 292,
    "CHC": 180,
    "BOS": 95,
    "NYY": 220,
    "LAD": 25,
    "SFG": 285,
    "SF": 285,
    "ATL": 10,
    "PHI": 340,
    "STL": 340,
    "MIL": 220,
    "MIN": 330,
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
    "LAS": 315,  # Las Vegas Ballpark — CF roughly NW; verify when stadium finalised
    "SEA": 5,
    "SDP": 310,
    "SD": 310,
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
    angle_diff = abs(wind_dir - cf_bearing) % 360
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
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
    df["wind_out"] = wind_outs  # positive = blowing out
    return df


# ── backfill ───────────────────────────────────────────────────────────────────


def _fetch_hourly(lat, lon, date_str):
    """Return the raw hourly dict from Open-Meteo archive for a single date."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "America/New_York",
    }
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive", params=params, timeout=15
    )
    resp.raise_for_status()
    return resp.json()["hourly"]


def _extract_at_hour(hourly, game_hour=18):
    """Pull weather values at the given local hour (default 6 pm)."""
    hours = hourly["time"]  # ["YYYY-MM-DDTHH:00", ...]
    idx = next(
        (i for i, h in enumerate(hours) if int(h[11:13]) == game_hour),
        game_hour,  # fallback: use the integer directly as an index
    )
    return (
        hourly["temperature_2m"][idx],
        hourly["relativehumidity_2m"][idx],
        hourly["windspeed_10m"][idx],
        hourly["winddirection_10m"][idx],
    )


def _fetch_game_log_from_snowflake(start_year: int, end_year: int) -> pd.DataFrame:
    """Pull distinct game-date + home-team rows from Snowflake GAME_RESULTS."""
    import snowflake.connector
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    key_path = os.path.expanduser(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"])
    passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    pw_bytes = passphrase.encode() if passphrase else None

    with open(key_path, "rb") as f:
        p_key = serialization.load_pem_private_key(
            f.read(), password=pw_bytes, backend=default_backend()
        )
    private_key = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    db = os.environ.get("SNOWFLAKE_DATABASE", "BASEBALL")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "STATCAST")

    query = f"""
        SELECT DISTINCT GAME_DATE, TEAM_H AS home_team
        FROM {db}.{schema}.GAME_RESULTS
        WHERE YEAR(GAME_DATE) BETWEEN {start_year} AND {end_year}
        ORDER BY GAME_DATE
    """

    cfg = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "private_key": private_key,
        "database": db,
        "schema": schema,
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    }
    role = os.environ.get("SNOWFLAKE_ROLE", "")
    if role:
        cfg["role"] = role

    print(f"Pulling game log from Snowflake ({start_year}-{end_year}) ...")
    conn = snowflake.connector.connect(**cfg)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        cols = [d[0].lower() for d in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=cols)
    finally:
        cursor.close()
        conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"Fetched {len(df):,} game-log rows from Snowflake")
    return df


def backfill_cache(
    cache_path: str,
    game_log_path: str | None = None,
    start_year: int = 2015,
    end_year: int = 2019,
    game_hour: int = 18,
):
    """
    Extend an existing weather cache with historical data from Open-Meteo.

    Parameters
    ----------
    cache_path : str
        Path to the existing .pkl cache. New entries are merged in and the
        file is overwritten in place.
    game_log_path : str or None
        CSV with columns: game_date, home_team[, game_hour].
        If None (default), the game log is pulled directly from Snowflake
        using the standard SNOWFLAKE_* env vars.
    start_year / end_year : int
        Inclusive year range to backfill.
    game_hour : int
        Local hour used as first-pitch proxy when the game log has no game_hour col.
    """
    # Load existing cache
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded existing cache: {len(cache):,} entries")
    except FileNotFoundError:
        cache = {}
        print("No existing cache found — starting fresh")

    # Load game log — CSV or Snowflake
    if game_log_path:
        games = pd.read_csv(game_log_path, parse_dates=["game_date"])
        games = games[games["game_date"].dt.year.between(start_year, end_year)]
    else:
        games = _fetch_game_log_from_snowflake(start_year, end_year)

    games = games.drop_duplicates(["game_date", "home_team"])
    print(f"Game-log rows to process: {len(games):,}")

    fetched = skipped = failed = 0

    for _, row in games.iterrows():
        key = (row["game_date"].date(), row["home_team"])
        if key in cache:
            skipped += 1
            continue

        home_team = row["home_team"]

        # Domes: fixed environment, no API call needed
        if home_team in DOME_TEAMS:
            cache[key] = (DOME_TEMP, DOME_HUMIDITY, DOME_WIND, DOME_WIND)
            fetched += 1
            continue

        coords = STADIUM_COORDS.get(home_team)
        if coords is None:
            print(f"  No coords for {home_team} — skipping")
            failed += 1
            continue

        date_str = row["game_date"].strftime("%Y-%m-%d")
        hour = int(row["game_hour"]) if "game_hour" in row.index else game_hour

        try:
            hourly = _fetch_hourly(*coords, date_str)
            temp, humidity, wind_spd, wind_dir = _extract_at_hour(hourly, hour)
            cf_bearing = STADIUM_CF_BEARING.get(home_team, 0)
            wind_out = compute_wind_out(wind_spd, wind_dir, cf_bearing)
            cache[key] = (temp, humidity, wind_spd, wind_out)
            fetched += 1
        except Exception as e:
            print(f"  Failed {key}: {e}")
            failed += 1

    print(f"Fetched: {fetched}  Skipped (already cached): {skipped}  Failed: {failed}")

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Cache saved → {cache_path}  (total entries: {len(cache):,})")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Backfill weather cache via Open-Meteo"
    )
    parser.add_argument(
        "--cache", required=True, help="Path to .pkl cache (updated in place)"
    )
    parser.add_argument(
        "--game-log",
        default=None,
        help="CSV with game_date + home_team (omit to pull from Snowflake)",
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--game-hour",
        type=int,
        default=18,
        help="Local hour used as first-pitch proxy",
    )
    args = parser.parse_args()

    backfill_cache(
        cache_path=args.cache,
        game_log_path=args.game_log,
        start_year=args.start_year,
        end_year=args.end_year,
        game_hour=args.game_hour,
    )
