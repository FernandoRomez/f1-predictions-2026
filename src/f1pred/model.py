"""Pace model plus Monte Carlo race simulation."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import POST_QUALI, PRE_QUALI

PARAMS = dict(
    n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=25,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
    random_state=37, n_jobs=-1, verbose=-1,
)
SEASON_DECAY = 0.7  # weight multiplier per season of age
ERA_BOOST = 2.0  # extra weight for races under the current regulations
DEFAULT_SIGMA = 0.2
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)


def _weights(train: pd.DataFrame, target_year: int, target_era: int) -> np.ndarray:
    weight = SEASON_DECAY ** (target_year - train["year"]).to_numpy(dtype=float)
    return weight * np.where(train["era"] == target_era, ERA_BOOST, 1.0)


def fit(train: pd.DataFrame, features: list[str], target_year: int, target_era: int) -> lgb.LGBMRegressor:
    rows = train[train["target"].notna()]
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(rows[features], rows["target"], sample_weight=_weights(rows, target_year, target_era))
    return model


def dnf_probability(race: pd.DataFrame, baseline: float) -> np.ndarray:
    """Blend driver and team reliability, shrunk toward the recent field average."""
    driver = race["driver_dnf_rate"].fillna(baseline).to_numpy()
    team = race["team_dnf_rate"].fillna(baseline).to_numpy()
    return np.clip(0.3 * driver + 0.4 * team + 0.3 * baseline, 0.02, 0.5)


def simulate(scores, dnf_p, sigma=DEFAULT_SIGMA, runs=5000, seed=37) -> np.ndarray:
    """Return a runs x drivers matrix of finishing ranks, 0 = winner."""
    rng = np.random.default_rng(seed)
    scores, dnf_p = np.asarray(scores), np.asarray(dnf_p)
    noisy = scores + rng.normal(0, sigma, (runs, len(scores)))
    noisy = noisy + 100 * (rng.random((runs, len(scores))) < dnf_p)  # retirements sort last
    return noisy.argsort(axis=1).argsort(axis=1)


def summarize(race: pd.DataFrame, ranks: np.ndarray) -> pd.DataFrame:
    out = race[["code", "team"]].reset_index(drop=True).copy()
    out["win"] = (ranks == 0).mean(axis=0)
    out["podium"] = (ranks < 3).mean(axis=0)
    out["points"] = (ranks < 10).mean(axis=0)
    out["exp_points"] = points_for(ranks).mean(axis=0)
    out["median_pos"] = np.median(ranks, axis=0) + 1
    return out.sort_values(["win", "exp_points"], ascending=False).reset_index(drop=True)


def points_for(ranks: np.ndarray) -> np.ndarray:
    table = np.zeros(max(int(ranks.max()) + 1, len(POINTS)))
    table[: len(POINTS)] = POINTS
    return table[ranks]


def forecast(frame: pd.DataFrame, race_id: int, sigma=DEFAULT_SIGMA, runs=5000, seed=37):
    """Train on all races before race_id and simulate it; uses quali data when present."""
    race = frame[frame["race_id"] == race_id]
    past = frame[frame["race_id"] < race_id]
    has_quali = race["quali_gap"].notna().mean() > 0.5
    features = POST_QUALI if has_quali else PRE_QUALI
    model = fit(past, features, int(race["year"].iloc[0]), int(race["era"].iloc[0]))
    scores = model.predict(race[features])
    baseline = past[past["year"] >= race["year"].iloc[0] - 1]["dnf"].astype(float).mean()
    return race, simulate(scores, dnf_probability(race, baseline), sigma, runs, seed), model, features
