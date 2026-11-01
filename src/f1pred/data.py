"""Descarga y guarda en caché datos de carreras de la API Jolpica (compatible con Ergast)."""
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
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


def _code(driver: dict) -> str:
    """Los reservas pueden no traer código en la API."""
    return driver.get("code", driver["familyName"][:3].upper())


def fetch_standings(year: int) -> pd.DataFrame:
    tables = _get(f"{BASE}/{year}/driverstandings.json")["StandingsTable"]["StandingsLists"]
    return pd.DataFrame(
        {"driver": s["Driver"]["driverId"], "code": _code(s["Driver"]), "points": float(s["points"])}
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
            "code": _code(q["Driver"]),
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


def _stint_laps(laps: pd.Series) -> float:
    """Longitud media de stint a partir de las vueltas de las paradas."""
    laps = np.asarray(laps.dropna().sort_values(), dtype=float)
    if laps.size == 0:
        return float("nan")
    return float(np.mean(np.diff(np.concatenate([[0.0], laps]))))


def pit_table(races: pd.DataFrame) -> pd.DataFrame:
    """Paradas por piloto y carrera, con granularidad de stint (no solo la mediana)."""
    rows = []
    for year, round_ in races[["year", "round"]].drop_duplicates().itertuples(index=False):
        stops = pd.DataFrame(fetch_pitstops(int(year), int(round_)))
        if stops.empty:
            continue
        stops["seconds"] = stops["duration"].map(_seconds)
        stops["lap"] = pd.to_numeric(stops.get("lap"), errors="coerce")
        table = stops.groupby("driverId").agg(
            pit_stops=("stop", "count"),
            pit_time=("seconds", "median"),
            pit_time_mean=("seconds", "mean"),
            pit_time_std=("seconds", lambda s: s.std(ddof=0)),
            first_stint_laps=("lap", "min"),
            stint_laps=("lap", _stint_laps),
        )
        rows.append(table.reset_index().rename(columns={"driverId": "driver"}).assign(year=year, round=round_))
    return pd.concat(rows, ignore_index=True)


def _climate_normal(lat: float, lon: float, day: str) -> dict:
    """Prior climático del circuito para una fecha: valor de un modelo, no un pronóstico.

    Se usan como prior cuando la carrera queda fuera del horizonte de
    pronóstico de 14 días, en lugar de dejar NaN.
    """
    resp = requests.get("https://climate-api.open-meteo.com/v1/climate", timeout=30, params={
        "latitude": lat, "longitude": lon, "start_date": day, "end_date": day,
        "daily": "precipitation_sum,temperature_2m_max,wind_speed_10m_max",
    })
    resp.raise_for_status()
    body = resp.json()
    daily = body["daily"]
    return {"rain_mm": daily["precipitation_sum"][0], "temp_max": daily["temperature_2m_max"][0],
            "wind_max": daily["wind_speed_10m_max"][0], "elevation": body.get("elevation")}


def fetch_weather(lat: float, lon: float, day: str) -> dict:
    """Clima del día de carrera: archivo para las antiguas, pronóstico para recientes o
    futuras, y normales climáticas como prior más allá del horizonte de pronóstico."""
    path = CACHE / "weather" / f"{lat:.2f}_{lon:.2f}_{day}.json"
    day_date = date.fromisoformat(day)
    recent = day_date > date.today() - timedelta(days=10)
    # Solo el archivo histórico es definitivo; lo demás se vuelve a pedir.
    if not recent and path.exists():
        return json.loads(path.read_text())
    if day_date > date.today() + timedelta(days=14):  # fuera del horizonte: prior climático
        prior = CACHE / "climate" / path.name  # separado para no pisar el pronóstico real
        if prior.exists():
            return json.loads(prior.read_text())
        try:
            weather = _climate_normal(lat, lon, day)
        except requests.RequestException:
            return {"rain_mm": None, "temp_max": None, "wind_max": None, "elevation": None}
        prior.parent.mkdir(parents=True, exist_ok=True)
        prior.write_text(json.dumps(weather))
        time.sleep(0.15)
        return weather
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


# --- Safety car por circuito (OpenF1, mejor esfuerzo) ------------------------
_OPENF1_CIRCUIT = {
    "Monte Carlo": "monaco", "Monza": "monza", "Silverstone": "silverstone",
    "Spa-Francorchamps": "spa", "Suzuka": "suzuka", "Interlagos": "interlagos",
    "Bahrain": "bahrain", "Jeddah": "jeddah", "Albert Park": "albert_park",
    "Miami": "miami", "Imola": "imola", "Baku": "baku", "Montreal": "montreal",
    "Spielberg": "red_bull_ring", "Hungaroring": "hungaroring", "Zandvoort": "zandvoort",
    "Singapore": "marina_bay", "Austin": "cota", "Mexico City": "rodriguez",
    "Las Vegas": "vegas", "Yas Marina": "yas_marina", "Losail": "losail",
    "Shanghai": "shanghai", "Barcelona": "catalunya", "Madring": "madring",
    "Villeneuve": "montreal",
}


def refresh_sc_rates(years: tuple[int, ...] = (2024, 2025)) -> dict:
    """Menciones de safety car/VSC por carrera y circuito (OpenF1), como prior refinable.

    Mejor esfuerzo: no es necesario para que el modelo corra. Cuenta los mensajes
    de race control que mencionan "SAFETY CAR" por sesión de carrera y promedia
    por circuito. Se guarda en ``data/cache/safety_car.json``.
    """
    path = CACHE / "safety_car.json"
    mentions: dict[str, list[float]] = {}
    for year in years:
        resp = requests.get("https://api.openf1.org/v1/sessions", timeout=30,
                            params={"year": year, "session_type": "Race"})
        if resp.status_code != 200:
            continue
        for session in resp.json():
            if session.get("session_name") != "Race":  # session_type también incluye sprints
                continue
            circuit = _OPENF1_CIRCUIT.get(session.get("circuit_short_name"))
            if circuit is None:
                continue
            rc = requests.get("https://api.openf1.org/v1/race_control", timeout=30,
                              params={"session_key": session["session_key"]})
            if rc.status_code != 200:
                continue
            count = sum(1 for row in rc.json() if "SAFETY CAR" in (row.get("message") or ""))
            mentions.setdefault(circuit, []).append(float(count))
            time.sleep(0.1)
    out = {c: sum(v) / len(v) for c, v in mentions.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


# --- Pace por vuelta (OpenF1, mejor esfuerzo) --------------------------------
def _openf1_race_sessions(year: int) -> list[dict]:
    """Sesiones de carrera de OpenF1 para un año, ordenadas por fecha de inicio."""
    path = CACHE / f"openf1_sessions_{year}.json"
    if path.exists():
        return json.loads(path.read_text())
    resp = requests.get("https://api.openf1.org/v1/sessions", timeout=30,
                        params={"year": year, "session_type": "Race"})
    resp.raise_for_status()
    sessions = sorted((s for s in resp.json() if s.get("session_name") == "Race"),
                      key=lambda s: s["date_start"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sessions))
    time.sleep(0.1)
    return sessions


def fetch_openf1_laps(year: int, round_: int) -> pd.DataFrame:
    """Ritmo por piloto (mediana de vuelta) desde OpenF1 para una carrera.

    Devuelve un DataFrame con columnas ``driver`` (código Jolpica) y ``lap_pace``
    (mediana de ``lap_duration``, NaN si no hay vueltas). Es mejor esfuerzo y
    asume que los rounds de Jolpica son continuos dentro de la temporada.
    """
    path = CACHE / "openf1" / f"{year}_{round_}_laps.csv"
    if path.exists():
        return pd.read_csv(path)
    sessions = _openf1_race_sessions(year)
    if not (1 <= round_ <= len(sessions)):
        return pd.DataFrame(columns=["driver", "lap_pace"])
    session_key = sessions[round_ - 1]["session_key"]
    laps = requests.get("https://api.openf1.org/v1/laps", timeout=30,
                        params={"session_key": session_key}).json()
    drivers = requests.get("https://api.openf1.org/v1/drivers", timeout=30,
                           params={"session_key": session_key}).json()
    code = {d["driver_number"]: d["name_acronym"] for d in drivers}
    frame = pd.DataFrame(laps)
    if frame.empty or "lap_duration" not in frame:
        return pd.DataFrame(columns=["driver", "lap_pace"])
    frame = frame[frame["is_pit_out_lap"].fillna(False).astype(bool) == False]
    frame["lap_pace"] = pd.to_numeric(frame["lap_duration"], errors="coerce")
    frame = frame.dropna(subset=["lap_pace"])
    table = frame.groupby("driver_number")["lap_pace"].median().reset_index()
    table["driver"] = table["driver_number"].map(code)
    out = table[["driver", "lap_pace"]].dropna(subset=["driver"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    time.sleep(0.1)
    return out
