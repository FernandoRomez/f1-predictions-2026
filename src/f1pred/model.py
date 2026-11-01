"""Modelo de ritmo y simulación Monte Carlo de carrera."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, circuits
from .features import POST_QUALI, PRE_QUALI

# Reexportamos la configuración congelada para no romper la API pública.
PARAMS = config.PARAMS
SEASON_DECAY = config.SEASON_DECAY
ERA_BOOST = config.ERA_BOOST
SIGMA_POST_QUALI = config.SIGMA_POST_QUALI
SIGMA_PRE_QUALI = config.SIGMA_PRE_QUALI
CHAOS = config.CHAOS
TEAM_SHARE = config.TEAM_SHARE
RAIN_NOISE = config.RAIN_NOISE
RAIN_DNF = config.RAIN_DNF
BLEND_ALPHA = config.BLEND_ALPHA
SEEDS = config.SEEDS
POINTS = config.POINTS
SPRINT_POINTS = config.SPRINT_POINTS


def _weights(train: pd.DataFrame, target_year: int, target_era: int,
             season_decay: float = SEASON_DECAY, era_boost: float = ERA_BOOST) -> np.ndarray:
    weight = season_decay ** (target_year - train["year"]).to_numpy(dtype=float)
    return weight * np.where(train["era"] == target_era, era_boost, 1.0)


def fit(train: pd.DataFrame, features: list[str], target_year: int, target_era: int,
        season_decay: float | None = None, era_boost: float | None = None) -> list[lgb.LGBMRegressor]:
    rows = train[train["target"].notna()]
    sd = SEASON_DECAY if season_decay is None else season_decay
    eb = ERA_BOOST if era_boost is None else era_boost
    models = []
    for seed in SEEDS:
        model = lgb.LGBMRegressor(**{**PARAMS, "random_state": seed})
        model.fit(rows[features], rows["target"], sample_weight=_weights(rows, target_year, target_era, sd, eb))
        models.append(model)
    return models


def predict(models: list[lgb.LGBMRegressor], rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.mean([model.predict(rows[features]) for model in models], axis=0)


def dnf_probability(race: pd.DataFrame, baseline: float) -> np.ndarray:
    """Mezcla fiabilidad de piloto y equipo, acercándola a la media reciente de la parrilla."""
    driver = race["driver_dnf_rate"].fillna(baseline).to_numpy()
    team = race["team_dnf_rate"].fillna(baseline).to_numpy()
    return np.clip(0.3 * driver + 0.4 * team + 0.3 * baseline, 0.02, 0.5)


def grid_win_rate(past: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
    """Tasa histórica de victorias por posición de salida: baseline probabilístico justo."""
    past = past[past["position"].notna() & (past["grid"] > 0)]
    wins = past[past["position"] == 1].groupby("grid").size()
    starts = past.groupby("grid").size()
    rate = (wins.reindex(race["grid"]).fillna(0).to_numpy() + 0.1) / (starts.reindex(race["grid"]).fillna(0).to_numpy() + 5)
    return rate / rate.sum()


def effective_chaos(circuit: str) -> float:
    """Caos de carrera modulado por la tasa histórica de SC/VSC del circuito."""
    return CHAOS * circuits.sc_chaos_factor(circuit)


def blend_win(model_win: np.ndarray, grid_prior: np.ndarray, alpha: float = BLEND_ALPHA) -> np.ndarray:
    """Combina la probabilidad de victoria del modelo con el prior de parrilla.

    Usa una combinación log-lineal (producto de expertos / media geométrica),
    que afila la distribución hacia el prior de parrilla —que acierta más el
    ganador— sin abandonar la calibración del modelo. ``alpha`` pondera el
    modelo; ``1 - alpha`` pondera la parrilla.
    """
    model_win = np.asarray(model_win, dtype=float)
    grid_prior = np.asarray(grid_prior, dtype=float)
    if alpha >= 1.0:
        return model_win
    if alpha <= 0.0:
        return grid_prior
    eps = 1e-6
    log_p = alpha * np.log(model_win + eps) + (1.0 - alpha) * np.log(grid_prior + eps)
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


def simulate(scores, dnf_p, sigma=SIGMA_POST_QUALI, runs=5000, seed=37, teams=None,
             rain_mm=0.0, offset=None, chaos=CHAOS, team_share=TEAM_SHARE,
             incident=None, dnf_scale=None, rain_noise=RAIN_NOISE, rain_dnf=RAIN_DNF) -> np.ndarray:
    """Devuelve una matriz simulaciones x pilotos con posiciones, 0 = ganador.

    El ruido tiene una parte por piloto, otra compartida entre compañeros y un
    multiplicador de caos por simulación que representa safety cars y cambios de
    estrategia. La lluvia lo amplía.

    ``incident`` (por simulación) sube la probabilidad de DNF de todos a la vez;
    ``dnf_scale`` (por piloto) modela un coche con problema estructural
    persistente. Ambos son opcionales y los usa la simulación del campeonato.
    """
    rng = np.random.default_rng(seed)
    scores, dnf_p = np.asarray(scores, dtype=float), np.asarray(dnf_p)
    n = len(scores)
    size = sigma * (rain_noise if rain_mm >= 2 else 1.0)
    chaos_factor = np.exp(rng.normal(0, chaos, (runs, 1)))
    noise = rng.normal(0, size * np.sqrt(1 - team_share), (runs, n))
    if teams is not None:
        codes = pd.factorize(np.asarray(teams))[0]
        noise = noise + rng.normal(0, size * np.sqrt(team_share), (runs, codes.max() + 1))[:, codes]
    noisy = scores + noise * chaos_factor + (0 if offset is None else offset)

    prob = np.clip(dnf_p * rain_dnf ** (rain_mm >= 2), 0, 1)
    if dnf_scale is not None:
        prob = prob * np.asarray(dnf_scale)
    if incident is not None:
        prob = prob * np.asarray(incident)[:, None]
    prob = np.clip(prob, 0, 1)
    noisy = noisy + 100 * (rng.random((runs, n)) < prob)
    return noisy.argsort(axis=1).argsort(axis=1)


def summarize(race: pd.DataFrame, ranks: np.ndarray, win: np.ndarray | None = None) -> pd.DataFrame:
    """``win`` sustituye la victoria simulada (p. ej. por el blend con la parrilla)."""
    out = race[["code", "team"]].reset_index(drop=True).copy()
    out["win"] = (ranks == 0).mean(axis=0) if win is None else win
    out["podium"] = (ranks < 3).mean(axis=0)
    out["points"] = (ranks < 10).mean(axis=0)
    out["exp_points"] = points_for(ranks).mean(axis=0)
    out["median_pos"] = np.median(ranks, axis=0) + 1
    return out.sort_values(["win", "exp_points"], ascending=False).reset_index(drop=True)


def points_for(ranks: np.ndarray, scale: np.ndarray = POINTS) -> np.ndarray:
    table = np.zeros(max(int(ranks.max()) + 1, len(scale)))
    table[: len(scale)] = scale
    return table[ranks]


def prepare(frame: pd.DataFrame, race_id: int) -> dict:
    """Entrena con las carreras previas a race_id; usa la qualy si existe."""
    race = frame[frame["race_id"] == race_id]
    past = frame[frame["race_id"] < race_id]
    has_quali = race["quali_gap"].notna().mean() > 0.5
    features, sigma = (POST_QUALI, SIGMA_POST_QUALI) if has_quali else (PRE_QUALI, SIGMA_PRE_QUALI)
    year = int(race["year"].iloc[0])
    models = fit(past, features, year, int(race["era"].iloc[0]))
    baseline = past[past["year"] >= year - 1]["dnf"].astype(float).mean()
    rain = race["rain_mm"].iloc[0]
    circuit = race["circuit"].iloc[0]
    return {
        "race": race, "features": features, "sigma": sigma, "models": models,
        "scores": predict(models, race, features), "dnf_p": dnf_probability(race, baseline),
        "teams": race["team"].to_numpy(), "rain_mm": 0.0 if pd.isna(rain) else float(rain),
        "chaos": effective_chaos(circuit),
        # Solo con qualy: antes de ella la parrilla de la carrera no se conoce.
        "grid_prior": grid_win_rate(past, race) if has_quali else None,
    }


def forecast(frame: pd.DataFrame, race_id: int, runs=5000, seed=37):
    plan = prepare(frame, race_id)
    ranks = simulate(plan["scores"], plan["dnf_p"], plan["sigma"], runs, seed, plan["teams"],
                     plan["rain_mm"], chaos=plan["chaos"])
    win = (ranks == 0).mean(axis=0)
    if plan["grid_prior"] is not None:
        win = blend_win(win, plan["grid_prior"])
    return plan["race"], ranks, plan["models"], plan["features"], win
