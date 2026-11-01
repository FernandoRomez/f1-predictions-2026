"""Ajuste ciego de hiperparámetros de simulación y blend, solo con 2023-2024.

Ejecuta:
  .venv/bin/python scripts/tune.py

No toca 2025-2026. Precalcula el ritmo de cada piloto una vez (LightGBM) y
luego barre la rejilla de simulación reutilizándolo. Escribe el resultado en
``src/f1pred/config.json`` solo si se pasa ``--write``.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from f1pred import config, data, features, tuning

YEARS = (2023, 2024)

GRID = {
    "sigma": [0.03, 0.04, 0.05, 0.06, 0.07],
    "chaos": [0.2, 0.3, 0.4],
    "team_share": [0.2, 0.25, 0.3],
    "blend_alpha": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "rain_noise": [1.5],
    "rain_dnf": [1.3],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="escribe el mejor en config.json")
    parser.add_argument("--mode", default="post", choices=["post", "pre"])
    args = parser.parse_args()

    frame = features.build_features(data.enrich(data.load_history(last=2026, refresh_last=False)))
    plans = tuning.precompute(frame, YEARS, mode=args.mode)
    print(f"planes 2023-24: {len(plans)}")

    results = tuning.grid_search(plans, GRID, objective="winner_hit", penalty=1.0)
    best = results[0]
    print("\nTop 10 (objetivo winner_hit, penaliza empeorar log-loss):")
    for r in results[:10]:
        print(f"  sigma={r['sigma']:.3f} chaos={r['chaos']:.2f} team_share={r['team_share']:.2f} "
              f"blend={r['blend_alpha']:.2f} -> hit={r['winner_hit']:.3f} logloss={r['logloss']:.3f} "
              f"brier={r['brier']:.3f} score={r['penalized']:.3f}")

    # La mejor config sin blend (alpha=1) como referencia del modelo puro.
    pure = [r for r in results if r["blend_alpha"] == 1.0]
    if pure:
        p = max(pure, key=lambda r: r["penalized"])
        print(f"\nMejor sin blend: hit={p['winner_hit']:.3f} logloss={p['logloss']:.3f} brier={p['brier']:.3f}")

    print("\nCV anidada por año (dejar un año fuera):")
    cv = tuning.leave_one_year_out(frame, YEARS, GRID, mode=args.mode)
    print(cv[["held_year", "winner_hit", "logloss", "brier", "blend_alpha", "sigma"]].to_string(index=False))

    if args.write:
        frozen = tuning.freeze(best, mode=args.mode)
        frozen["version"] = f"{config.VERSION}+tuned"
        path = Path(__file__).resolve().parents[1] / "src" / "f1pred" / "config.json"
        path.write_text(json.dumps(frozen, indent=2) + "\n")
        print(f"\nEscrito {path}")


if __name__ == "__main__":
    main()
