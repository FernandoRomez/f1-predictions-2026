"""Download and cache race data from the Jolpica (Ergast-compatible) API."""
import json
import time
from datetime import date
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
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        time.sleep(0.3)
        return resp.json()["MRData"]
    raise RuntimeError(f"rate limited: {url}")


def _fetch_races(year: int, endpoint: str, refresh: bool) -> list[dict]:
    """Fetch every race of an endpoint, merging races split across pages."""
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
    """One row per driver per race; only the newest season is ever re-downloaded."""
    last = last or date.today().year
    rows = []
    for year in range(first, last + 1):
        refresh = refresh_last and year == last
        rows += _rows(_fetch_races(year, "results", refresh), _fetch_races(year, "qualifying", refresh))
    return pd.DataFrame(rows).sort_values(["date", "position"]).reset_index(drop=True)
