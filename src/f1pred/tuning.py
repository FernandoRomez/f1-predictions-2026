"""Ajuste ciego de hiperparámetros y validación cruzada anidada por año.

Protocolo: todo el ajuste se hace SOLO con ``config.TUNED_ON`` (2023-2024).
Las temporadas ``config.HOLDOUT`` (2025-2026) no se usan para elegir nada.

El truco clave es separar lo caro (ajustar LightGBM para obtener el ritmo de
cada piloto) de lo barato (la simulación Monte Carlo y el blend con la
parrilla). Los hiperparámetros de simulación (``sigma``, ``chaos``,
``team_share``, ``rain_*``, ``blend_alpha``) solo afectan a la simulación, así
que el ritmo se precalcula una vez por carrera y luego se reutiliza para
evaluar cientos de combinaciones en segundos.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import circuits, config, model as m
from .backtest import _grid_win_rate


@dataclass
class RacePlan:
    """Resultado walk-forward de una carrera ya ajustado (ritmo + contexto)."""
    race_id: int
    year: int
    scores: np.ndarray
    dnf_p: np.ndarray
    teams: np.ndarray
    rain_mm: float
    sc_factor: float
    grid_prior: np.ndarray
    position: np.ndarray
    winner: np.ndarray
    podium: np.ndarray


def precompute(frame: pd.DataFrame, years: tuple[int, ...], mode: str = "post",
               season_decay: float | None = None, era_boost: float | None = None) -> list[RacePlan]:
    """Ajusta una vez por carrera (walk-forward) y devuelve el ritmo de cada una."""
    rows = []
    features = m.POST_QUALI if mode == "post" else m.PRE_QUALI
    races = frame[(frame["year"].isin(years)) & frame["position"].notna()]
    for race_id in sorted(races["race_id"].unique()):
        race = frame[frame["race_id"] == race_id]
        past = frame[frame["race_id"] < race_id]
        year, era = int(race["year"].iloc[0]), int(race["era"].iloc[0])
        models = m.fit(past, features, year, era, season_decay=season_decay, era_boost=era_boost)
        scores = m.predict(models, race, features)
        baseline = past[past["year"] >= year - 1]["dnf"].astype(float).mean()
        dnf_p = m.dnf_probability(race, baseline)
        position = race["position"].to_numpy()
        rain = race["rain_mm"].iloc[0]
        # Sin qualy no se conoce la parrilla: prior uniforme (el blend no aporta nada).
        grid_prior = _grid_win_rate(past, race) if mode == "post" else np.full(len(race), 1 / len(race))
        rows.append(RacePlan(
            race_id=race_id, year=year, scores=scores, dnf_p=dnf_p,
            teams=race["team"].to_numpy(), rain_mm=0.0 if pd.isna(rain) else float(rain),
            sc_factor=circuits.sc_chaos_factor(race["circuit"].iloc[0]),
            grid_prior=grid_prior, position=position,
            winner=position == 1, podium=position <= 3,
        ))
    return rows


def metrics(plans: list[RacePlan], *, sigma: float, chaos: float, team_share: float,
            rain_noise: float, rain_dnf: float, blend_alpha: float,
            runs: int = 2000, seed: int = 0) -> dict[str, float]:
    """Agrega winner_hit y log-loss (entre otras) sobre un conjunto de carreras."""
    hits, logloss, brier, top3_prob = [], [], [], []
    for i, plan in enumerate(plans):
        ranks = m.simulate(plan.scores, plan.dnf_p, sigma, runs, seed + i, plan.teams,
                           plan.rain_mm, chaos=chaos * plan.sc_factor, team_share=team_share,
                           rain_noise=rain_noise, rain_dnf=rain_dnf)
        win = (ranks == 0).mean(axis=0)
        podium = (ranks < 3).mean(axis=0)
        if blend_alpha is not None and blend_alpha < 1.0:
            win = m.blend_win(win, plan.grid_prior, blend_alpha)
        hits.append(float(plan.winner[np.argmax(win)]))
        logloss.append(-np.log(np.clip(win[plan.winner], 1e-3, 1)).mean())
        brier.append(((win - plan.winner) ** 2).sum())
        top3_prob.append(float(podium[plan.podium].mean()))
    return {
        "winner_hit": float(np.mean(hits)),
        "logloss": float(np.mean(logloss)),
        "brier": float(np.mean(brier)),
        "top3_prob": float(np.mean(top3_prob)),
    }


def grid_search(plans: list[RacePlan], grid: dict[str, list], objective: str = "winner_hit",
                penalty: float = 1.0) -> list[dict]:
    """Barre una rejilla de hiperparámetros de simulación y ordena por objetivo.

    ``objective`` puede ser "winner_hit" o "logloss". Con "winner_hit" se añade
    una penalización proporcional a ``penalty`` por el empeoramiento del
    log-loss respecto al mejor, para no sacrificar la calibración.
    """
    keys = list(grid)
    results = []
    best_logloss = None
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        met = metrics(plans, **cfg)
        best_logloss = met["logloss"] if best_logloss is None else min(best_logloss, met["logloss"])
        results.append({**cfg, **met})
    for r in results:
        r["penalized"] = r[objective] - penalty * max(0.0, r["logloss"] - best_logloss) \
            if objective == "winner_hit" else -r["logloss"]
    return sorted(results, key=lambda r: -r["penalized"])


def leave_one_year_out(frame: pd.DataFrame, years: tuple[int, ...], grid: dict[str, list],
                       objective: str = "winner_hit", penalty: float = 1.0,
                       mode: str = "post") -> pd.DataFrame:
    """CV anidada por año: para cada año dejado fuera, se ajusta en los demás y
    se evalúa en él. Reporta la métrica por año (para intervalos)."""
    rows = []
    for held in years:
        train_years = tuple(y for y in years if y != held)
        train_plans = precompute(frame, train_years, mode=mode)
        best = grid_search(train_plans, grid, objective, penalty)[0]
        held_plans = precompute(frame, (held,), mode=mode)
        met = metrics(held_plans, **{k: best[k] for k in grid})
        rows.append({"held_year": held, **best, **met})
    return pd.DataFrame(rows)


def freeze(best: dict, mode: str = "post") -> dict:
    """Devuelve el dict de configuración con los valores elegidos, para volcar a config.json.

    Solo toca los parámetros de simulación que barre la rejilla. ``sigma`` se
    escribe en ``sigma_post_quali`` o ``sigma_pre_quali`` según el modo.
    """
    new = config.snapshot()
    sigma_key = "sigma_post_quali" if mode == "post" else "sigma_pre_quali"
    mapping = {
        sigma_key: "sigma",
        "chaos": "chaos",
        "team_share": "team_share",
        "rain_noise": "rain_noise",
        "rain_dnf": "rain_dnf",
        "blend_alpha": "blend_alpha",
    }
    for key, src in mapping.items():
        if src in best:
            new[key] = best[src]
    return new
