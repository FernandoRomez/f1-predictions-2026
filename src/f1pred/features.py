"""Point-in-time features: every value uses only races before the row's own race."""
import numpy as np
import pandas as pd

# Regulation resets: car performance carries over poorly across these seasons.
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
    "era_races", "street",
]
POST_QUALI = PRE_QUALI + ["grid_pct", "quali_gap"]


def _era(year: int) -> int:
    return next(i for i, (a, b) in enumerate(ERAS) if a <= year <= b)


def _lagged_ewm(values: pd.Series, groups, alpha: float) -> pd.Series:
    return values.groupby(groups).transform(lambda s: s.shift().ewm(alpha=alpha, ignore_na=True).mean())


def _lagged_mean(values: pd.Series, groups) -> pd.Series:
    return values.groupby(groups).transform(lambda s: s.shift().expanding().mean())


def add_upcoming(history: pd.DataFrame, upcoming: dict, entrants: pd.DataFrame) -> pd.DataFrame:
    """Append a race with unknown outcome so it receives features like any other."""
    rows = entrants[["driver", "code", "team"]].assign(
        **upcoming, position=np.nan, dnf=np.nan, points=np.nan,
    )
    return pd.concat([history, rows], ignore_index=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "driver"]).reset_index(drop=True).copy()
    df["team"] = df["team"].replace(LINEAGE)
    df["era"] = df["year"].map(_era)
    df["street"] = df["circuit"].isin(STREET).astype(int)
    df["race_id"] = df["year"] * 100 + df["round"]

    # Target: finishing order among classified cars, 0 = winner, 1 = last.
    finished = df["dnf"].eq(False)
    rank = df["position"].where(finished).groupby(df["race_id"]).rank()
    span = (rank.groupby(df["race_id"]).transform("max") - 1).clip(lower=1)
    df["target"] = (rank - 1) / span

    field = df.groupby("race_id")["driver"].transform("count")
    grid = pd.to_numeric(df["grid"], errors="coerce").astype(float)
    grid = grid.where(grid != 0, field)  # 0 means a pit-lane start
    df["grid_pct"] = (grid - 1) / (field - 1)
    df["quali_gap"] = df["quali_time"] / df.groupby("race_id")["quali_time"].transform("min") - 1

    dnf = df["dnf"].astype(float)
    df["driver_form"] = _lagged_ewm(df["target"], df["driver"], 0.15)
    df["driver_form_recent"] = _lagged_ewm(df["target"], df["driver"], 0.4)
    df["driver_dnf_rate"] = _lagged_ewm(dnf, df["driver"], 0.05)
    df["driver_circuit"] = _lagged_mean(df["target"], [df["driver"], df["circuit"]])

    team_race = (
        df.assign(dnf_value=dnf)
        .groupby(["era", "team", "circuit", "race_id"], as_index=False)
        .agg(target=("target", "mean"), quali_gap=("quali_gap", "mean"), dnf_value=("dnf_value", "mean"))
        .sort_values("race_id")
    )
    keys = [team_race["era"], team_race["team"]]
    team_race["team_form"] = _lagged_ewm(team_race["target"], keys, 0.2)
    team_race["team_form_recent"] = _lagged_ewm(team_race["target"], keys, 0.5)
    team_race["team_quali_form"] = _lagged_ewm(team_race["quali_gap"], keys, 0.3)
    team_race["team_dnf_rate"] = _lagged_ewm(team_race["dnf_value"], keys, 0.1)
    team_race["team_circuit"] = _lagged_mean(team_race["target"], [team_race["team"], team_race["circuit"], team_race["era"]])
    columns = ["team_form", "team_form_recent", "team_quali_form", "team_dnf_rate", "team_circuit"]
    df = df.merge(team_race[["team", "race_id", *columns]], on=["team", "race_id"], how="left")

    df["driver_vs_team"] = df["driver_form"] - df["team_form"]
    df["era_races"] = df.groupby("era")["race_id"].rank(method="dense")
    return df
