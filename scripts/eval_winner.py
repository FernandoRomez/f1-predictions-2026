"""Prueba ciega de objetivos centrados en el ganador.

  calcular: python scripts/eval_winner.py fit
  elegir:   python scripts/eval_winner.py select    (solo muestra 2023-2024)
  evaluar:  python scripts/eval_winner.py holdout "clf" 0.5    (una sola vez)
"""
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax

from f1pred import data, features, model as m

CACHE = Path(sys.argv[2]) if sys.argv[1] in ("fit", "select") and len(sys.argv) > 2 else Path("/tmp/none")
ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
TEMPERATURES = [0.5, 1.0, 2.0]


def clf_scores(past, race, cols, year, era):
    rows = past[past["position"].notna()]
    probs = []
    for seed in m.SEEDS:
        model = lgb.LGBMClassifier(**{**m.PARAMS, "random_state": seed})
        model.fit(rows[cols], (rows["position"] == 1).astype(int), sample_weight=m._weights(rows, year, era))
        probs.append(model.predict_proba(race[cols])[:, 1])
    p = np.mean(probs, axis=0)
    return p / p.sum()


def rank_scores(past, race, cols, year, era):
    rows = past[past["position"].notna()].sort_values("race_id")
    label = (4 - rows["position"]).clip(lower=0).astype(int)
    group = rows.groupby("race_id", sort=True).size().to_numpy()
    out = []
    for seed in m.SEEDS:
        model = lgb.LGBMRanker(**{**m.PARAMS, "random_state": seed}, objective="lambdarank")
        model.fit(rows[cols], label, group=group, sample_weight=m._weights(rows, year, era))
        out.append(model.predict(race[cols]))
    return np.mean(out, axis=0)


def fit(path):
    frame = features.build_features(data.enrich(data.load_history(last=2026, refresh_last=False)))
    cols = features.POST_QUALI
    out = []
    races = frame[(frame["year"] >= 2023) & frame["position"].notna()]
    for race_id in sorted(races["race_id"].unique()):
        race = frame[frame["race_id"] == race_id]
        past = frame[frame["race_id"] < race_id]
        year, era = int(race["year"].iloc[0]), int(race["era"].iloc[0])
        baseline = past[past["year"] >= year - 1]["dnf"].astype(float).mean()
        rain = race["rain_mm"].iloc[0]
        ranks = m.simulate(m.predict(m.fit(past, cols, year, era), race, cols), m.dnf_probability(race, baseline),
                           m.SIGMA_POST_QUALI, 5000, int(race_id), race["team"].to_numpy(),
                           0.0 if pd.isna(rain) else rain, chaos=m.effective_chaos(race["circuit"].iloc[0]))
        out.append({"race_id": race_id, "year": year, "won": race["position"].to_numpy() == 1,
                    "grid": m.grid_win_rate(past, race), "reg": (ranks == 0).mean(axis=0),
                    "clf": clf_scores(past, race, cols, year, era),
                    "rank": rank_scores(past, race, cols, year, era)})
        print(race_id, flush=True)
    path.write_bytes(pickle.dumps(out))


def probabilities(race, name, alpha, temp=1.0):
    p = softmax(race["rank"] / temp) if name == "rank" else race[name]
    return m.blend_win(p, race["grid"], alpha)


def score(races, name, alpha, temp=1.0):
    wins = [probabilities(r, name, alpha, temp) for r in races]
    hit = [float(r["won"][np.argmax(p)]) for r, p in zip(races, wins)]
    loss = [-np.log(np.clip(p[r["won"]], 1e-3, 1)).item() for r, p in zip(races, wins)]
    return np.array(hit), np.array(loss)


def select(path):
    races = [r for r in pickle.loads(path.read_bytes()) if r["year"] in (2023, 2024)]
    rows = []
    for name in ("reg", "clf", "rank"):
        for temp in TEMPERATURES if name == "rank" else [1.0]:
            for alpha in ALPHAS:
                hit, loss = score(races, name, alpha, temp)
                rows.append(dict(name=name, temp=temp, alpha=alpha, hit=hit.mean(), logloss=loss.mean()))
    table = pd.DataFrame(rows).sort_values(["hit", "logloss"], ascending=[False, True])
    print(len(races), "carreras 2023-24")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def holdout(path, name, alpha, temp):
    races = [r for r in pickle.loads(path.read_bytes()) if r["year"] >= 2025]
    new_hit, new_loss = score(races, name, alpha, temp)
    old_hit, old_loss = score(races, "reg", 1.0)
    grid_hit, grid_loss = score(races, "grid", 1.0)
    print(f"{len(races)} carreras")
    print(f"elegido hit={new_hit.mean():.3f} logloss={new_loss.mean():.3f}")
    print(f"actual  hit={old_hit.mean():.3f} logloss={old_loss.mean():.3f}")
    print(f"parrilla hit={grid_hit.mean():.3f} logloss={grid_loss.mean():.3f}")
    rng = np.random.default_rng(0)
    for label, base_hit, base_loss in (("actual", old_hit, old_loss), ("parrilla", grid_hit, grid_loss)):
        idx = rng.integers(0, len(races), (5000, len(races)))
        for metric, diff in (("hit", new_hit - base_hit), ("logloss", new_loss - base_loss)):
            lo, hi = np.quantile(diff[idx].mean(axis=1), [0.05, 0.95])
            print(f"vs {label} {metric}: {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(CACHE)
    elif sys.argv[1] == "select":
        select(CACHE)
    else:
        holdout(Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
