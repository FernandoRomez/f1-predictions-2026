"""Monte Carlo del título de pilotos desde la clasificación actual y las carreras restantes."""
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from . import config, model as m

# Cola de riesgo: DNF correlacionados (incidente por carrera) y problemas
# estructurales persistentes de COCHE (por piloto). Valores congelados en config.json.
DRIVER_DRIFT = config.DRIVER_DRIFT
TEAM_DRIFT = config.TEAM_DRIFT
INCIDENT_RATE = config.INCIDENT_RATE
INCIDENT_BOOST = config.INCIDENT_BOOST
STRUCTURAL_RATE = config.STRUCTURAL_RATE
STRUCTURAL_BOOST = config.STRUCTURAL_BOOST
STRUCTURAL_PERSIST = config.STRUCTURAL_PERSIST
STRUCTURAL_PACE = config.STRUCTURAL_PACE


def _title_ci(counts: np.ndarray, runs: int, ci: tuple[float, float] = (0.05, 0.95)) -> tuple[np.ndarray, np.ndarray]:
    """Error de muestreo Monte Carlo (Beta). No incluye la incertidumbre del modelo."""
    lo = beta_dist.ppf(ci[0], 1 + counts, 1 + runs - counts)
    hi = beta_dist.ppf(ci[1], 1 + counts, 1 + runs - counts)
    return lo, hi


def simulate_title(frames: list[tuple[pd.DataFrame, int, bool]], standings: pd.DataFrame, wins: dict,
                   runs=200000, seed=37) -> pd.DataFrame:
    """``frames`` contiene (variables, race_id, es_fin_de_semana_sprint) por carrera restante.

    Además de la deriva de ritmo, modela (a) un factor de incidente por carrera
    que sube el DNF de todos a la vez y (b) un estado estructural por coche
    (piloto) que persiste varias carreras y lastra fiabilidad y ritmo. Devuelve
    un intervalo de credibilidad sobre la probabilidad de título en vez de solo
    el punto, y usa suficientes runs para muestrear la cola.
    """
    codes = standings["code"].tolist()
    totals = pd.DataFrame(np.tile(standings["points"].to_numpy(), (runs, 1)), columns=codes)
    victories = pd.DataFrame(np.tile([wins.get(c, 0) for c in codes], (runs, 1)), columns=codes, dtype=float)
    rng = np.random.default_rng(seed)
    driver_drift = pd.DataFrame(rng.normal(0, DRIVER_DRIFT, (runs, len(codes))), columns=codes)
    team_drift: dict[str, np.ndarray] = {}

    # Estado estructural por coche (fila = simulación, columna = piloto). Persiste.
    driver_index = {c: i for i, c in enumerate(codes)}
    structural = np.zeros((runs, len(codes)), dtype=bool)
    recover_p = 1.0 / max(STRUCTURAL_PERSIST, 1.0)

    for i, (frame, race_id, sprint) in enumerate(frames):
        plan = m.prepare(frame, race_id)
        race_codes = plan["race"]["code"].tolist()

        # (b) evoluciona el problema estructural de cada coche.
        bad_to_ok = structural & (rng.random((runs, len(codes))) < recover_p)
        ok_to_bad = (~structural) & (rng.random((runs, len(codes))) < STRUCTURAL_RATE)
        structural = (structural & ~bad_to_ok) | ok_to_bad
        race_idx = np.array([driver_index.get(c, -1) for c in race_codes])
        bad = np.zeros((runs, len(race_codes)), dtype=bool)
        valid = race_idx >= 0
        bad[:, valid] = structural[:, race_idx[valid]]
        dnf_scale = np.where(bad, STRUCTURAL_BOOST, 1.0)

        # (a) incidente global de carrera: sube el DNF de todos a la vez.
        incident = np.where(rng.random(runs) < INCIDENT_RATE, INCIDENT_BOOST, 1.0)

        offset = driver_drift.reindex(columns=race_codes, fill_value=0.0).to_numpy(copy=True)
        for j, team in enumerate(plan["teams"]):
            offset[:, j] += team_drift.setdefault(team, rng.normal(0, TEAM_DRIFT, runs))
        # Un coche con problema estructural también pierde ritmo (no solo fiabilidad).
        offset += bad * STRUCTURAL_PACE

        args = (plan["scores"], plan["dnf_p"], plan["sigma"], runs)
        extra = dict(teams=plan["teams"], rain_mm=plan["rain_mm"], offset=offset, chaos=plan["chaos"],
                     incident=incident, dnf_scale=dnf_scale)
        ranks = m.simulate(*args, seed + i, **extra)
        earned = pd.DataFrame(m.points_for(ranks), columns=race_codes)
        if sprint:
            sprint_ranks = m.simulate(*args, seed + 1000 + i, **extra)
            earned += m.points_for(sprint_ranks, m.SPRINT_POINTS)
        totals += earned.reindex(columns=codes, fill_value=0.0)
        victories += pd.DataFrame(ranks == 0, columns=race_codes).reindex(columns=codes, fill_value=False).to_numpy()

    tiebreak = victories.to_numpy() * 1e-3 + rng.random(totals.shape) * 1e-6  # desempate por victorias
    champion = (totals.to_numpy() + tiebreak).argmax(axis=1)
    counts = np.bincount(champion, minlength=len(codes))
    lo, hi = _title_ci(counts, runs)
    return pd.DataFrame({
        "code": codes,
        "points_now": standings["points"].to_numpy(),
        "title": counts / runs,
        "title_lo": lo,
        "title_hi": hi,
        "exp_points": totals.to_numpy().mean(axis=0),
    }).sort_values(["title", "exp_points"], ascending=False).reset_index(drop=True)
