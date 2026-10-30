"""Modelo de ritmo y simulación Monte Carlo de carrera."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import POST_QUALI, PRE_QUALI

PARAMS = dict(
    n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=25,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
    random_state=37, n_jobs=-1, verbose=-1,
)
SEASON_DECAY = 0.7  # multiplicador de peso por temporada de antigüedad
ERA_BOOST = 2.0  # peso extra para carreras con el reglamento vigente
SIGMA_POST_QUALI = 0.05  # elegido solo con 2023-24
SIGMA_PRE_QUALI = 0.06
CHAOS = 0.3
TEAM_SHARE = 0.25
RAIN_NOISE = 1.5
RAIN_DNF = 1.3
SEEDS = (37, 38, 39)
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
SPRINT_POINTS = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)


def _weights(train: pd.DataFrame, target_year: int, target_era: int) -> np.ndarray:
    weight = SEASON_DECAY ** (target_year - train["year"]).to_numpy(dtype=float)
    return weight * np.where(train["era"] == target_era, ERA_BOOST, 1.0)


def fit(train: pd.DataFrame, features: list[str], target_year: int, target_era: int) -> list[lgb.LGBMRegressor]:
    rows = train[train["target"].notna()]
    models = []
    for seed in SEEDS:
        model = lgb.LGBMRegressor(**{**PARAMS, "random_state": seed})
        model.fit(rows[features], rows["target"], sample_weight=_weights(rows, target_year, target_era))
        models.append(model)
    return models


def predict(models: list[lgb.LGBMRegressor], rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.mean([model.predict(rows[features]) for model in models], axis=0)


def dnf_probability(race: pd.DataFrame, baseline: float) -> np.ndarray:
    """Mezcla fiabilidad de piloto y equipo, acercándola a la media reciente de la parrilla."""
    driver = race["driver_dnf_rate"].fillna(baseline).to_numpy()
    team = race["team_dnf_rate"].fillna(baseline).to_numpy()
    return np.clip(0.3 * driver + 0.4 * team + 0.3 * baseline, 0.02, 0.5)


def simulate(scores, dnf_p, sigma=SIGMA_POST_QUALI, runs=5000, seed=37, teams=None, rain_mm=0.0, offset=None) -> np.ndarray:
    """Devuelve una matriz simulaciones x pilotos con posiciones, 0 = ganador.

    El ruido tiene una parte por piloto, otra compartida entre compañeros y un
    multiplicador de caos por simulación que representa safety cars y cambios de
    estrategia. La lluvia lo amplía.
    """
    rng = np.random.default_rng(seed)
    scores, dnf_p = np.asarray(scores, dtype=float), np.asarray(dnf_p)
    n = len(scores)
    size = sigma * (RAIN_NOISE if rain_mm >= 2 else 1.0)
    chaos = np.exp(rng.normal(0, CHAOS, (runs, 1)))
    noise = rng.normal(0, size * np.sqrt(1 - TEAM_SHARE), (runs, n))
    if teams is not None:
        codes = pd.factorize(np.asarray(teams))[0]
        noise = noise + rng.normal(0, size * np.sqrt(TEAM_SHARE), (runs, codes.max() + 1))[:, codes]
    noisy = scores + noise * chaos + (0 if offset is None else offset)
    noisy = noisy + 100 * (rng.random((runs, n)) < np.clip(dnf_p * RAIN_DNF ** (rain_mm >= 2), 0, 1))
    return noisy.argsort(axis=1).argsort(axis=1)


def summarize(race: pd.DataFrame, ranks: np.ndarray) -> pd.DataFrame:
    out = race[["code", "team"]].reset_index(drop=True).copy()
    out["win"] = (ranks == 0).mean(axis=0)
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
    return {
        "race": race, "features": features, "sigma": sigma, "models": models,
        "scores": predict(models, race, features), "dnf_p": dnf_probability(race, baseline),
        "teams": race["team"].to_numpy(), "rain_mm": 0.0 if pd.isna(rain) else float(rain),
    }


def forecast(frame: pd.DataFrame, race_id: int, runs=5000, seed=37):
    plan = prepare(frame, race_id)
    ranks = simulate(plan["scores"], plan["dnf_p"], plan["sigma"], runs, seed, plan["teams"], plan["rain_mm"])
    return plan["race"], ranks, plan["models"], plan["features"]
