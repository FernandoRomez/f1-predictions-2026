"""Variables sin fuga: cada valor usa solo carreras anteriores a la de su fila."""
import numpy as np
import pandas as pd

# Cambios de reglamento: el rendimiento del auto apenas se conserva entre estas etapas.
ERAS = [(2018, 2021), (2022, 2025), (2026, 2100)]

LINEAGE = {
    "force_india": "aston_martin", "racing_point": "aston_martin",
    "renault": "alpine",
    "toro_rosso": "rb", "alphatauri": "rb",
    "alfa": "audi", "sauber": "audi",
}
STREET = {"monaco", "baku", "marina_bay", "jeddah", "vegas", "miami", "albert_park", "villeneuve", "madring"}

PRE_QUALI = [
    "driver_form", "driver_form_recent", "team_form", "team_form_recent", "team_quali_form",
    "driver_vs_team", "driver_circuit", "team_circuit", "driver_dnf_rate", "team_dnf_rate",
    "era_races", "street", "driver_quali_form", "quali_vs_team", "positions_gained", "wet_form",
    "wet_edge", "team_pit_time", "circuit_overtakes", "circuit_dnf", "circuit_stops", "rain_mm",
    "temp_max", "wind_max", "elevation",
]
POST_QUALI = PRE_QUALI + ["grid_pct", "quali_gap", "grid_penalty", "quali_gap_vs_form"]
OPTIONAL = ["rain_mm", "temp_max", "wind_max", "elevation", "pit_stops", "pit_time"]


def _era(year: int) -> int:
    return next(i for i, (a, b) in enumerate(ERAS) if a <= year <= b)


def _lagged_ewm(values: pd.Series, groups, alpha: float) -> pd.Series:
    return values.groupby(groups).transform(lambda s: s.shift().ewm(alpha=alpha, ignore_na=True).mean())


def _lagged_mean(values: pd.Series, groups) -> pd.Series:
    return values.groupby(groups).transform(lambda s: s.shift().expanding().mean())


def add_upcoming(history: pd.DataFrame, upcoming: dict, entrants: pd.DataFrame) -> pd.DataFrame:
    """Añade una carrera sin resultado; los inscritos pueden traer parrilla y qualy."""
    columns = [c for c in ("driver", "code", "team", "grid", "quali_pos", "quali_time") if c in entrants]
    rows = entrants[columns].assign(**upcoming, position=np.nan, dnf=np.nan, points=np.nan)
    return pd.concat([history, rows], ignore_index=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "driver"]).reset_index(drop=True).copy()
    for column in OPTIONAL:
        if column not in df:
            df[column] = np.nan
    df["team"] = df["team"].replace(LINEAGE)
    df["era"] = df["year"].map(_era)
    df["street"] = df["circuit"].isin(STREET).astype(int)
    df["race_id"] = df["year"] * 100 + df["round"]

    # Objetivo: orden de llegada entre clasificados, 0 = ganador, 1 = último.
    finished = df["dnf"].eq(False)
    rank = df["position"].where(finished).groupby(df["race_id"]).rank()
    span = (rank.groupby(df["race_id"]).transform("max") - 1).clip(lower=1)
    df["target"] = (rank - 1) / span

    field = df.groupby("race_id")["driver"].transform("count")
    grid = pd.to_numeric(df["grid"], errors="coerce").astype(float)
    grid = grid.where(grid != 0, field)  # 0 significa salida desde pits
    df["grid_pct"] = (grid - 1) / (field - 1)
    df["quali_gap"] = df["quali_time"] / df.groupby("race_id")["quali_time"].transform("min") - 1

    df["grid_penalty"] = (grid - pd.to_numeric(df["quali_pos"], errors="coerce")) / field
    dnf = df["dnf"].astype(float)
    df["driver_form"] = _lagged_ewm(df["target"], df["driver"], 0.15)
    df["driver_form_recent"] = _lagged_ewm(df["target"], df["driver"], 0.4)
    df["driver_dnf_rate"] = _lagged_ewm(dnf, df["driver"], 0.05)
    df["driver_circuit"] = _lagged_mean(df["target"], [df["driver"], df["circuit"]])

    df["driver_quali_form"] = _lagged_ewm(df["quali_gap"], df["driver"], 0.25)
    df["quali_gap_vs_form"] = df["quali_gap"] - df["driver_quali_form"]
    df["positions_gained"] = _lagged_ewm((grid - df["position"]).where(finished) / field, df["driver"], 0.1)
    df["wet_form"] = _lagged_mean(df["target"].where(df["rain_mm"] >= 1), df["driver"])
    df["wet_edge"] = df["wet_form"] - df["driver_form"]

    circuit = df.assign(moved=(grid - df["position"]).abs(), dnf_value=dnf).groupby(
        ["circuit", "race_id"], as_index=False
    ).agg(moved=("moved", "mean"), dnf_value=("dnf_value", "mean"), stops=("pit_stops", "mean")).sort_values("race_id")
    circuit["circuit_overtakes"] = _lagged_mean(circuit["moved"], circuit["circuit"])
    circuit["circuit_dnf"] = _lagged_mean(circuit["dnf_value"], circuit["circuit"])
    circuit["circuit_stops"] = _lagged_mean(circuit["stops"], circuit["circuit"])
    df = df.merge(circuit[["race_id", "circuit_overtakes", "circuit_dnf", "circuit_stops"]], on="race_id", how="left")

    team_race = (
        df.assign(dnf_value=dnf.to_numpy())
        .groupby(["era", "team", "circuit", "race_id"], as_index=False)
        .agg(target=("target", "mean"), quali_gap=("quali_gap", "mean"), dnf_value=("dnf_value", "mean"),
              pit_time=("pit_time", "mean"))
        .sort_values("race_id")
    )
    keys = [team_race["era"], team_race["team"]]
    team_race["team_form"] = _lagged_ewm(team_race["target"], keys, 0.2)
    team_race["team_form_recent"] = _lagged_ewm(team_race["target"], keys, 0.5)
    team_race["team_quali_form"] = _lagged_ewm(team_race["quali_gap"], keys, 0.3)
    team_race["team_dnf_rate"] = _lagged_ewm(team_race["dnf_value"], keys, 0.1)
    team_race["team_pit_time"] = _lagged_ewm(team_race["pit_time"], keys, 0.2)
    team_race["team_circuit"] = _lagged_mean(team_race["target"], [team_race["team"], team_race["circuit"], team_race["era"]])
    columns = ["team_form", "team_form_recent", "team_quali_form", "team_dnf_rate", "team_circuit", "team_pit_time"]
    df = df.merge(team_race[["team", "race_id", *columns]], on=["team", "race_id"], how="left")

    df["driver_vs_team"] = df["driver_form"] - df["team_form"]
    df["quali_vs_team"] = df["driver_quali_form"] - df["team_quali_form"]
    df["era_races"] = df.groupby("era")["race_id"].rank(method="dense")
    return df
