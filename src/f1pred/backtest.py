"""Walk-forward evaluation: each race is predicted using only earlier races."""
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr

from . import model as m
from .features import PRE_QUALI


def _race_metrics(race: pd.DataFrame, ranks: np.ndarray, win: np.ndarray | None = None) -> dict:
    position = race["position"].to_numpy()
    win = (ranks == 0).mean(axis=0) if win is None else win
    podium = (ranks < 3).mean(axis=0)
    expected = ranks.mean(axis=0)
    winner = position == 1
    actual_top3 = position <= 3
    return {
        "spearman": spearmanr(expected, position).statistic,
        "mae": np.abs(expected.argsort().argsort() + 1 - position).mean(),
        "winner_hit": float(winner[np.argmax(win)]),
        "top3_overlap": actual_top3[np.argsort(-podium)[:3]].mean(),
        "win_logloss": -np.log(np.clip(win[winner], 1e-3, 1)).item(),
        "win_brier": ((win - winner) ** 2).sum(),
    }


def _ordering(values: np.ndarray, runs: int = 1) -> np.ndarray:
    """Deterministic forecast as a rank matrix."""
    return np.tile(values.argsort().argsort(), (runs, 1))


def _grid_win_rate(past: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
    """Historical win rate of each grid slot: a fair probabilistic baseline."""
    past = past[past["position"].notna() & (past["grid"] > 0)]
    wins = past[past["position"] == 1].groupby("grid").size()
    starts = past.groupby("grid").size()
    rate = (wins.reindex(race["grid"]).fillna(0).to_numpy() + 0.1) / (starts.reindex(race["grid"]).fillna(0).to_numpy() + 5)
    return rate / rate.sum()


def run(frame: pd.DataFrame, first_year=2023, sigmas=(0.2,), mode="post") -> pd.DataFrame:
    rows = []
    races = frame[(frame["year"] >= first_year) & frame["position"].notna()]
    for race_id in sorted(races["race_id"].unique()):
        race = frame[frame["race_id"] == race_id]
        past = frame[frame["race_id"] < race_id]
        year, era = int(race["year"].iloc[0]), int(race["era"].iloc[0])
        features = m.POST_QUALI if mode == "post" else PRE_QUALI
        scores = m.fit(past, features, year, era).predict(race[features])
        baseline = past[past["year"] >= year - 1]["dnf"].astype(float).mean()
        dnf_p = m.dnf_probability(race, baseline)
        for sigma in sigmas:
            ranks = m.simulate(scores, dnf_p, sigma, runs=2000, seed=int(race_id))
            rows.append({"race_id": race_id, "year": year, "method": f"model_s{sigma}", **_race_metrics(race, ranks)})
        rows.append({"race_id": race_id, "year": year, "method": "grid",
                     **_race_metrics(race, _ordering(race["grid_pct"].to_numpy()), _grid_win_rate(past, race))})
        form = race["driver_form"].fillna(1).to_numpy()
        rows.append({"race_id": race_id, "year": year, "method": "driver_form",
                     **_race_metrics(race, _ordering(form), softmax(-form * 8))})
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> pd.DataFrame:
    columns = ["spearman", "mae", "winner_hit", "top3_overlap", "win_logloss", "win_brier"]
    return results.groupby("method")[columns].mean().round(3)
