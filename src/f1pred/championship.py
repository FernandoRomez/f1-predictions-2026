"""Monte Carlo of the drivers' title from current standings and remaining races."""
import numpy as np
import pandas as pd

from . import model as m


def simulate_title(frames: list[tuple[pd.DataFrame, int]], standings: pd.DataFrame, runs=20000, seed=37) -> pd.DataFrame:
    """frames holds (feature frame, race_id) for each race left on the calendar."""
    codes = standings["code"].tolist()
    totals = np.tile(standings["points"].to_numpy(), (runs, 1))
    for i, (frame, race_id) in enumerate(frames):
        race, ranks, _, _ = m.forecast(frame, race_id, runs=runs, seed=seed + i)
        points = pd.DataFrame(m.points_for(ranks), columns=race["code"].tolist())
        totals += points.reindex(columns=codes, fill_value=0.0).to_numpy()
    jitter = np.random.default_rng(seed).random(totals.shape) * 1e-6  # countback is not modelled
    champion = (totals + jitter).argmax(axis=1)
    return pd.DataFrame({
        "code": codes,
        "points_now": standings["points"].to_numpy(),
        "title": np.bincount(champion, minlength=len(codes)) / runs,
        "exp_points": totals.mean(axis=0),
    }).sort_values("title", ascending=False).reset_index(drop=True)
