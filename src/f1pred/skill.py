"""Habilidad de piloto separada del coche, sin fuga de datos.

Antes de cada carrera se resuelve una regresión ridge con las carreras previas:

    resultado ≈ habilidad_piloto + rendimiento_equipo×temporada

El efecto del coche se reinicia cada temporada; el del piloto la sobrevive, así
que los compañeros de equipo y los cambios de equipo conectan toda la parrilla.
Valores menores son mejores. Un novato parte de 0, la media del prior.
"""
import numpy as np
import pandas as pd

COLUMNS = ["driver_skill_race", "driver_skill_quali", "driver_skill_sunday"]
QUALI_GAP_CAP = 0.03  # evita que una vuelta rota o una qualy mojada domine
TEAM_RIDGE = 0.1


def _lagged_effects(df: pd.DataFrame, value: pd.Series, half_life: float, ridge: float) -> np.ndarray:
    """Efecto de piloto de cada fila, estimado solo con carreras anteriores."""
    drivers, n_drivers = pd.factorize(df["driver"])[0], df["driver"].nunique()
    team_keys = df["year"].astype(str) + "/" + df["team"].astype(str)
    teams = pd.factorize(team_keys)[0] + n_drivers
    size = n_drivers + team_keys.nunique()
    penalty = np.r_[np.full(n_drivers, ridge), np.full(size - n_drivers, TEAM_RIDGE)]
    decay = 0.5 ** (1 / half_life)
    xtx, xty = np.zeros((size, size)), np.zeros(size)
    out = np.zeros(len(df))
    y = (value - value.groupby(df["race_id"]).transform("mean")).to_numpy(dtype=float)
    race_ids = df["race_id"].to_numpy()
    for race_id in np.sort(df["race_id"].unique()):
        rows = np.flatnonzero(race_ids == race_id)
        if xty.any():
            coef = np.linalg.solve(xtx + np.diag(penalty), xty)
            out[rows] = coef[drivers[rows]]
        xtx *= decay
        xty *= decay
        for row in rows[~np.isnan(y[rows])]:
            pair = [drivers[row], teams[row]]
            xtx[np.ix_(pair, pair)] += 1
            xty[pair] += y[row]
    return out


def add_skill(df: pd.DataFrame, half_life: float, ridge: float) -> pd.DataFrame:
    """half_life se mide en carreras. Requiere target, quali_gap y race_id."""
    out = df.copy()
    quali = (out["quali_gap"].clip(upper=QUALI_GAP_CAP)) / QUALI_GAP_CAP
    out["driver_skill_race"] = _lagged_effects(out, out["target"], half_life, ridge)
    out["driver_skill_quali"] = _lagged_effects(out, quali, half_life, ridge)
    out["driver_skill_sunday"] = out["driver_skill_race"] - out["driver_skill_quali"]
    return out
