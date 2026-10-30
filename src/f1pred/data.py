"""Descarga y guarda en caché datos de carreras de la API Jolpica (compatible con Ergast)."""
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.jolpi.ca/ergast/f1"
CACHE = Path("data/cache")
PAGE = 100


def _get(url: str) -> dict:
    for attempt in range(6):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        resp.raise_for_status()
        time.sleep(0.3)
        return resp.json()["MRData"]
    raise RuntimeError(f"rate limited: {url}")


def _fetch_races(year: int, endpoint: str, refresh: bool) -> list[dict]:
    """Descarga todas las carreras de un endpoint y une las repartidas entre páginas."""
    path = CACHE / f"{year}_{endpoint}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    key = {"results": "Results", "qualifying": "QualifyingResults"}[endpoint]
    races: dict[str, dict] = {}
    offset = 0
    while True:
        page = _get(f"{BASE}/{year}/{endpoint}.json?limit={PAGE}&offset={offset}")
        for race in page["RaceTable"]["Races"]:
            merged = races.setdefault(race["round"], {**race, key: []})
            merged[key].extend(race[key])
        offset += PAGE
        if offset >= int(page["total"]):
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    result = list(races.values())
    path.write_text(json.dumps(result))
    return result


def fetch_schedule(year: int) -> pd.DataFrame:
    races = _get(f"{BASE}/{year}.json?limit=40")["RaceTable"]["Races"]
    return pd.DataFrame(
        {
            "year": year,
            "round": int(r["round"]),
            "race": r["raceName"],
            "circuit": r["Circuit"]["circuitId"],
            "lat": float(r["Circuit"]["Location"]["lat"]),
            "lon": float(r["Circuit"]["Location"]["long"]),
            "date": r["date"],
            "sprint": "Sprint" in r,
        }
        for r in races
    )


def fetch_standings(year: int) -> pd.DataFrame:
    tables = _get(f"{BASE}/{year}/driverstandings.json")["StandingsTable"]["StandingsLists"]
    return pd.DataFrame(
        {"driver": s["Driver"]["driverId"], "code": s["Driver"]["code"], "points": float(s["points"])}
        for s in tables[0]["DriverStandings"]
    )


def _seconds(text: str | None) -> float:
    if not text:
        return float("nan")
    minutes, _, seconds = text.rpartition(":")
    return int(minutes or 0) * 60 + float(seconds)


def _rows(results: list[dict], qualifying: list[dict]) -> list[dict]:
    quali = {}
    for race in qualifying:
        for q in race["QualifyingResults"]:
            times = [_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")]
            best = min((t for t in times if t == t), default=float("nan"))
            quali[(race["round"], q["Driver"]["driverId"])] = (int(q["position"]), best)
    rows = []
    for race in results:
        for r in race["Results"]:
            driver = r["Driver"]["driverId"]
            status = r["status"]
            finished = status == "Finished" or status == "Lapped" or status.startswith("+")
            quali_pos, quali_time = quali.get((race["round"], driver), (None, float("nan")))
            rows.append(
                {
                    "year": int(race["season"]),
                    "round": int(race["round"]),
                    "date": race["date"],
                    "race": race["raceName"],
                    "circuit": race["Circuit"]["circuitId"],
                    "lat": float(race["Circuit"]["Location"]["lat"]),
                    "lon": float(race["Circuit"]["Location"]["long"]),
                    "driver": driver,
                    "code": r["Driver"].get("code", driver[:3].upper()),
                    "team": r["Constructor"]["constructorId"],
                    "grid": int(r["grid"]),
                    "quali_pos": quali_pos,
                    "quali_time": quali_time,
                    "position": int(r["position"]),
                    "dnf": not finished,
                    "points": float(r["points"]),
                }
            )
    return rows


def load_history(first: int = 2018, last: int | None = None, refresh_last: bool = True) -> pd.DataFrame:
    """Una fila por piloto y carrera; solo se vuelve a descargar la temporada más reciente."""
    last = last or date.today().year
    rows = []
    for year in range(first, last + 1):
        refresh = refresh_last and year == last
        rows += _rows(_fetch_races(year, "results", refresh), _fetch_races(year, "qualifying", refresh))
    return pd.DataFrame(rows).sort_values(["date", "position"]).reset_index(drop=True)


def fetch_qualifying(year: int, round_: int) -> pd.DataFrame:
    """Qualy de una carrera; vacía hasta que se dispute la sesión."""
    races = _get(f"{BASE}/{year}/{round_}/qualifying.json?limit=40")["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()
    rows = []
    for q in races[0]["QualifyingResults"]:
        times = [_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")]
        rows.append({
            "driver": q["Driver"]["driverId"],
            "code": q["Driver"]["code"],
            "team": q["Constructor"]["constructorId"],
            "grid": int(q["position"]),
            "quali_pos": int(q["position"]),
            "quali_time": min((t for t in times if t == t), default=float("nan")),
        })
    return pd.DataFrame(rows)


def fetch_pitstops(year: int, round_: int) -> list[dict]:
    path = CACHE / "pits" / f"{year}_{round_}.json"
    if path.exists():
        return json.loads(path.read_text())
    stops, offset = [], 0
    while True:
        page = _get(f"{BASE}/{year}/{round_}/pitstops.json?limit={PAGE}&offset={offset}")
        for race in page["RaceTable"]["Races"]:
            stops += race["PitStops"]
        offset += PAGE
        if offset >= int(page["total"]):
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stops))
    return stops


def pit_table(races: pd.DataFrame) -> pd.DataFrame:
    """Paradas y mediana del tiempo en pits por piloto y carrera."""
    rows = []
    for year, round_ in races[["year", "round"]].drop_duplicates().itertuples(index=False):
        stops = pd.DataFrame(fetch_pitstops(int(year), int(round_)))
        if stops.empty:
            continue
        stops["seconds"] = stops["duration"].map(_seconds)
        table = stops.groupby("driverId").agg(pit_stops=("stop", "count"), pit_time=("seconds", "median"))
        rows.append(table.reset_index().rename(columns={"driverId": "driver"}).assign(year=year, round=round_))
    return pd.concat(rows, ignore_index=True)


def fetch_weather(lat: float, lon: float, day: str) -> dict:
    """Clima del día de carrera: archivo para las antiguas, pronóstico para recientes o futuras."""
    path = CACHE / "weather" / f"{lat:.2f}_{lon:.2f}_{day}.json"
    if path.exists():
        return json.loads(path.read_text())
    day_date = date.fromisoformat(day)
    if day_date > date.today() + timedelta(days=14):  # fuera del horizonte de pronóstico
        return {"rain_mm": None, "temp_max": None, "wind_max": None, "elevation": None}
    recent = day_date > date.today() - timedelta(days=10)
    host = "api.open-meteo.com/v1/forecast" if recent else "archive-api.open-meteo.com/v1/archive"
    resp = requests.get(f"https://{host}", timeout=30, params={
        "latitude": lat, "longitude": lon, "start_date": day, "end_date": day, "timezone": "auto",
        "daily": "precipitation_sum,temperature_2m_max,wind_speed_10m_max",
    })
    resp.raise_for_status()
    body = resp.json()
    daily = body["daily"]
    weather = {"rain_mm": daily["precipitation_sum"][0], "temp_max": daily["temperature_2m_max"][0],
               "wind_max": daily["wind_speed_10m_max"][0], "elevation": body.get("elevation")}
    if not recent:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(weather))
    time.sleep(0.15)
    return weather


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Añade clima y pits, también a una carrera que aún no se ha corrido."""
    races = df[["year", "round", "date", "lat", "lon"]].drop_duplicates(["year", "round"])
    weather = pd.DataFrame(
        {"year": r.year, "round": r.round, **fetch_weather(r.lat, r.lon, r.date)}
        for r in races.itertuples()
    )
    pits = pit_table(df[df["position"].notna()])
    return df.merge(weather, on=["year", "round"], how="left").merge(
        pits, on=["year", "round", "driver"], how="left")
