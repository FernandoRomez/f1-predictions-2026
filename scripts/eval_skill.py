"""Prueba ciega de driver_skill frente al modelo actual.

1. Elige half_life y ridge solo con 2023-2024.
2. Evalúa esa única opción en 2025-2026 con bootstrap pareado, con y sin qualy.
"""
import pandas as pd

from f1pred import backtest, data, features, model as m, skill, tuning

GRID = [(20, 1.0), (20, 5.0), (60, 1.0), (60, 5.0), (150, 1.0), (150, 5.0)]
SIM = dict(sigma=m.SIGMA_POST_QUALI, chaos=m.CHAOS, team_share=m.TEAM_SHARE,
           rain_noise=m.RAIN_NOISE, rain_dnf=m.RAIN_DNF, blend_alpha=m.BLEND_ALPHA)
BASE_POST, BASE_PRE = list(m.POST_QUALI), list(m.PRE_QUALI)


def use(extra: list[str]) -> None:
    m.POST_QUALI = BASE_POST + extra
    m.PRE_QUALI = BASE_PRE + extra
    backtest.PRE_QUALI = m.PRE_QUALI


def blind_logloss(frame: pd.DataFrame) -> float:
    return tuning.metrics(tuning.precompute(frame, (2023, 2024)), **SIM)["logloss"]


def main() -> None:
    base = features.build_features(data.enrich(data.load_history(last=2026, refresh_last=False)))
    use([])
    reference = blind_logloss(base)
    print(f"2023-24 actual: {reference:.4f}", flush=True)
    use(skill.COLUMNS)
    scores = {}
    for half_life, ridge in GRID:
        scores[half_life, ridge] = blind_logloss(skill.add_skill(base, half_life, ridge))
        print(f"2023-24 half_life={half_life} ridge={ridge}: {scores[half_life, ridge]:.4f}", flush=True)
    best = min(scores, key=scores.get)
    print("elegido:", best, flush=True)

    frame = skill.add_skill(base, *best)
    last = frame[frame["race_id"] == frame["race_id"].max()].sort_values("driver_skill_race")
    print(last[["code", *skill.COLUMNS]].head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for mode, sigma in (("post", m.SIGMA_POST_QUALI), ("pre", m.SIGMA_PRE_QUALI)):
        method = f"model_s{sigma}"
        use([])
        old = backtest.run(base, sigmas=(sigma,), mode=mode)
        use(skill.COLUMNS)
        new = backtest.run(frame, sigmas=(sigma,), mode=mode)
        both = pd.concat([old[old["method"] == method].assign(method="old"),
                          new[new["method"] == method].assign(method="new")])
        both = both[both["year"] >= 2025]
        print(f"\n--- {mode} ({both['race_id'].nunique()} carreras)")
        print(backtest.summary(both).to_string())
        print(backtest.paired_bootstrap(both, "new", "old").to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
