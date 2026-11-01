"""Barrido del peso del blend sobre el hold-out 2025-26 (solo reporte, no ajuste).

Precalcula el ritmo una vez y evalúa varias alphas para ver el efecto de anclar
la parrilla como prior en winner_hit y log-loss.
"""
import numpy as np

from f1pred import data, features, tuning

ALPHAS = [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5, 0.0]
HOLDOUT = (2025, 2026)


def main() -> None:
    frame = features.build_features(data.enrich(data.load_history(last=2026, refresh_last=False)))
    plans = tuning.precompute(frame, HOLDOUT, mode="post")
    print(f"carreras hold-out: {len(plans)}\n")

    print(f"{'alpha':>6} {'winner_hit':>11} {'logloss':>9} {'brier':>7} {'top3_prob':>10}")
    grid_only = tuning.metrics(plans, sigma=0.04, chaos=0.4, team_share=0.2,
                               rain_noise=1.5, rain_dnf=1.3, blend_alpha=0.0)
    for alpha in ALPHAS:
        met = tuning.metrics(plans, sigma=0.04, chaos=0.4, team_share=0.2,
                             rain_noise=1.5, rain_dnf=1.3, blend_alpha=alpha)
        tag = "grid puro" if alpha == 0.0 else ("modelo puro" if alpha == 1.0 else "")
        print(f"{alpha:>6.2f} {met['winner_hit']:>11.3f} {met['logloss']:>9.3f} "
              f"{met['brier']:>7.3f} {met['top3_prob']:>10.3f}  {tag}")


if __name__ == "__main__":
    main()
