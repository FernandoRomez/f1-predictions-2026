"""Monte Carlo del título de pilotos desde la clasificación actual y las carreras restantes."""
import numpy as np
import pandas as pd

from . import model as m

# Deriva de ritmo: la forma de hoy no queda fija para las carreras restantes.
DRIVER_DRIFT = 0.03
TEAM_DRIFT = 0.04


def simulate_title(frames: list[tuple[pd.DataFrame, int, bool]], standings: pd.DataFrame, wins: dict, runs=20000, seed=37) -> pd.DataFrame:
    """frames contiene (variables, race_id, es_fin_de_semana_sprint) por carrera restante."""
    codes = standings["code"].tolist()
    totals = pd.DataFrame(np.tile(standings["points"].to_numpy(), (runs, 1)), columns=codes)
    victories = pd.DataFrame(np.tile([wins.get(c, 0) for c in codes], (runs, 1)), columns=codes, dtype=float)
    rng = np.random.default_rng(seed)
    driver_drift = pd.DataFrame(rng.normal(0, DRIVER_DRIFT, (runs, len(codes))), columns=codes)
    team_drift: dict[str, np.ndarray] = {}
    for i, (frame, race_id, sprint) in enumerate(frames):
        plan = m.prepare(frame, race_id)
        race_codes = plan["race"]["code"].tolist()
        offset = driver_drift.reindex(columns=race_codes, fill_value=0.0).to_numpy()
        for j, team in enumerate(plan["teams"]):
            offset[:, j] += team_drift.setdefault(team, rng.normal(0, TEAM_DRIFT, runs))
        args = (plan["scores"], plan["dnf_p"], plan["sigma"], runs)
        extra = dict(teams=plan["teams"], rain_mm=plan["rain_mm"], offset=offset)
        ranks = m.simulate(*args, seed + i, **extra)
        earned = pd.DataFrame(m.points_for(ranks), columns=race_codes)
        if sprint:
            sprint_ranks = m.simulate(*args, seed + 1000 + i, **extra)
            earned += m.points_for(sprint_ranks, m.SPRINT_POINTS)
        totals += earned.reindex(columns=codes, fill_value=0.0)
        victories += pd.DataFrame(ranks == 0, columns=race_codes).reindex(columns=codes, fill_value=False).to_numpy()
    tiebreak = victories.to_numpy() * 1e-3 + rng.random(totals.shape) * 1e-6  # desempate por victorias
    champion = (totals.to_numpy() + tiebreak).argmax(axis=1)
    return pd.DataFrame({
        "code": codes,
        "points_now": standings["points"].to_numpy(),
        "title": np.bincount(champion, minlength=len(codes)) / runs,
        "exp_points": totals.to_numpy().mean(axis=0),
    }).sort_values(["title", "exp_points"], ascending=False).reset_index(drop=True)
