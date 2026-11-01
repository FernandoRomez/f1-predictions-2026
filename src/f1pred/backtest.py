"""Evaluación walk-forward: cada carrera se predice solo con carreras anteriores."""
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr

from . import model as m
from .features import PRE_QUALI

# Métricas que se promedian en el resumen (todas son por carrera).
METRIC_COLUMNS = [
    "spearman", "mae", "winner_hit", "top3_overlap", "top3_prob",
    "logloss", "brier", "podium_logloss",
]


def _race_metrics(race: pd.DataFrame, ranks: np.ndarray, win: np.ndarray | None = None,
                  podium: np.ndarray | None = None) -> dict:
    """Métricas probabilísticas y de orden para una carrera.

    ``logloss`` y ``brier`` son los scores multiclase completos de la clase
    "ganador" (la variable objetivo es one-hot en el ganador), por lo que
    ``logloss = -log(p_ganador)`` y ``brier = sum_j (p_j - y_j)^2``.
    """
    position = race["position"].to_numpy()
    win = (ranks == 0).mean(axis=0) if win is None else np.asarray(win)
    podium = (ranks < 3).mean(axis=0) if podium is None else np.asarray(podium)
    expected = ranks.mean(axis=0)
    winner = position == 1
    actual_top3 = position <= 3
    return {
        "spearman": spearmanr(expected, position).statistic,
        "mae": np.abs(expected.argsort().argsort() + 1 - position).mean(),
        "winner_hit": float(winner[np.argmax(win)]),
        "top3_overlap": actual_top3[np.argsort(-podium)[:3]].mean(),
        "top3_prob": float(podium[actual_top3].mean()),
        "logloss": -np.log(np.clip(win[winner], 1e-3, 1)).item(),
        "brier": ((win - winner) ** 2).sum(),
        "podium_logloss": -(actual_top3 * np.log(np.clip(podium, 1e-3, 1))
                            + (1 - actual_top3) * np.log(np.clip(1 - podium, 1e-3, 1))).mean(),
        "win_probs": win,
        "won": winner,
    }


def _ordering(values: np.ndarray, runs: int = 1) -> np.ndarray:
    """Pronóstico determinista como matriz de posiciones."""
    return np.tile(values.argsort().argsort(), (runs, 1))


_grid_win_rate = m.grid_win_rate


def run(frame: pd.DataFrame, first_year=2023, sigmas=(0.2,), mode="post",
        blend_alpha: float | None = None) -> pd.DataFrame:
    """Backtest walk-forward. Añade el método ``blend`` (parrilla como prior)."""
    rows = []
    races = frame[(frame["year"] >= first_year) & frame["position"].notna()]
    for race_id in sorted(races["race_id"].unique()):
        race = frame[frame["race_id"] == race_id]
        past = frame[frame["race_id"] < race_id]
        year, era = int(race["year"].iloc[0]), int(race["era"].iloc[0])
        features = m.POST_QUALI if mode == "post" else PRE_QUALI
        scores = m.predict(m.fit(past, features, year, era), race, features)
        baseline = past[past["year"] >= year - 1]["dnf"].astype(float).mean()
        dnf_p = m.dnf_probability(race, baseline)
        chaos = m.effective_chaos(race["circuit"].iloc[0])
        rain = race["rain_mm"].iloc[0]
        rain = 0.0 if pd.isna(rain) else rain
        for sigma in sigmas:
            ranks = m.simulate(scores, dnf_p, sigma, 2000, int(race_id), race["team"].to_numpy(),
                               rain, chaos=chaos)
            rows.append({"race_id": race_id, "year": year, "method": f"model_s{sigma}", **_race_metrics(race, ranks)})
        if blend_alpha is not None and mode == "post":  # sin qualy no hay parrilla que mezclar
            grid_prior = _grid_win_rate(past, race)
            rows.append({"race_id": race_id, "year": year, "method": "blend",
                         **_race_metrics(race, ranks, win=m.blend_win((ranks == 0).mean(0), grid_prior, blend_alpha))})
        rows.append({"race_id": race_id, "year": year, "method": "grid",
                     **_race_metrics(race, _ordering(race["grid_pct"].to_numpy()), _grid_win_rate(past, race))})
        form = race["driver_form"].fillna(1).to_numpy()
        rows.append({"race_id": race_id, "year": year, "method": "driver_form",
                     **_race_metrics(race, _ordering(form), softmax(-form * 8))})
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("method")[METRIC_COLUMNS].mean().round(4)


def calibration(results: pd.DataFrame, method: str = "blend", n_bins: int = 10) -> pd.DataFrame:
    """Curva de calibración sobre todos los pares piloto-carrera.

    Agrupa por probabilidad de victoria pronosticada y compara la media con la
    frecuencia con la que esos pilotos ganaron de verdad.
    """
    sub = results[results["method"] == method]
    if sub.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_pred", "win_rate"])
    probs = np.concatenate(sub["win_probs"].to_numpy())
    won = np.concatenate(sub["won"].to_numpy()).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    slots = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    out = [{"bin": (edges[i] + edges[i + 1]) / 2, "n": int((slots == i).sum()),
            "mean_pred": probs[slots == i].mean(), "win_rate": won[slots == i].mean()}
           for i in range(n_bins) if (slots == i).any()]
    return pd.DataFrame(out)


def expected_calibration_error(results: pd.DataFrame, method: str = "blend", n_bins: int = 10) -> float:
    table = calibration(results, method, n_bins)
    if table.empty:
        return float("nan")
    return float((table["n"] / table["n"].sum() * (table["mean_pred"] - table["win_rate"]).abs()).sum())


def bootstrap_ci(results: pd.DataFrame, n: int = 1000, ci: tuple[float, float] = (0.05, 0.95),
                 seed: int = 0) -> pd.DataFrame:
    """Intervalos de confianza por bootstrap sobre carreras, para cada método y métrica."""
    rng = np.random.default_rng(seed)
    out = []
    for method, group in results.groupby("method"):
        for metric in METRIC_COLUMNS:
            values = group[metric].to_numpy()
            means = []
            for _ in range(n):
                idx = rng.integers(0, len(values), len(values))
                means.append(values[idx].mean())
            means = np.array(means)
            out.append({"method": method, "metric": metric, "mean": values.mean(),
                        "lo": np.quantile(means, ci[0]), "hi": np.quantile(means, ci[1])})
    return pd.DataFrame(out)


def paired_bootstrap(results: pd.DataFrame, a: str, b: str, n: int = 2000,
                     ci: tuple[float, float] = (0.05, 0.95), seed: int = 0) -> pd.DataFrame:
    """Diferencia ``a - b`` remuestreando las mismas carreras en ambos métodos."""
    rng = np.random.default_rng(seed)
    left = results[results["method"] == a].set_index("race_id")
    right = results[results["method"] == b].set_index("race_id")
    races = left.index.intersection(right.index)
    out = []
    for metric in METRIC_COLUMNS:
        diff = (left.loc[races, metric] - right.loc[races, metric]).to_numpy(dtype=float)
        means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(axis=1)
        out.append({"metric": metric, "diff": diff.mean(), "lo": np.quantile(means, ci[0]),
                    "hi": np.quantile(means, ci[1])})
    return pd.DataFrame(out).assign(a=a, b=b)
