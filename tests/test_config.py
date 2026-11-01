import numpy as np

from f1pred import circuits, config, model as m
from f1pred.championship import _title_ci
from f1pred.data import _stint_laps


def test_config_is_versioned_and_blind():
    assert config.VERSION
    assert config.TUNED_ON == "2023-2024"
    assert config.HOLDOUT == "2025-2026"
    for key in ("params", "season_decay", "era_boost", "sigma_post_quali", "sigma_pre_quali",
                "chaos", "team_share", "rain_noise", "rain_dnf", "blend_alpha"):
        assert key in config.CONFIG


def test_snapshot_is_a_deep_copy():
    snap = config.snapshot()
    snap["chaos"] = 99.0
    assert config.CHAOS != 99.0


def test_safety_car_factor_orders_circuits():
    assert circuits.sc_chaos_factor("monaco") > 1.0
    assert circuits.sc_chaos_factor("monza") < 1.0
    assert np.isclose(circuits.sc_chaos_factor("nonexistent"), 1.0)


def test_effective_chaos_scales_global_chaos():
    assert m.effective_chaos("monaco") > m.CHAOS
    assert m.effective_chaos("monza") < m.CHAOS


def test_title_ci_brackets_the_point_estimate():
    counts = np.array([15000.0, 5000.0])
    lo, hi = _title_ci(counts, 20000)
    p = counts / 20000
    assert (lo <= p).all() and (hi >= p).all()
    assert (lo >= 0).all() and (hi <= 1).all()


def test_stint_laps_is_mean_inter_stop_lap():
    import pandas as pd
    laps = pd.Series([10, 25, 40])
    assert np.isclose(_stint_laps(laps), np.mean([10, 15, 15]))


def test_calibration_uses_every_driver_race_pair():
    import pandas as pd
    from f1pred import backtest
    results = pd.DataFrame({"method": ["blend"] * 2, "race_id": [1, 2],
                            "win_probs": [np.array([0.9, 0.1]), np.array([0.9, 0.1])],
                            "won": [np.array([True, False]), np.array([False, True])]})
    table = backtest.calibration(results, "blend")
    assert table["n"].sum() == 4
    assert np.isclose(table.loc[table["mean_pred"] > 0.5, "win_rate"].iloc[0], 0.5)


def test_paired_bootstrap_detects_constant_gap():
    import pandas as pd
    from f1pred import backtest
    rows = [{"method": k, "race_id": r, **{c: float(k == "a") for c in backtest.METRIC_COLUMNS}}
            for k in "ab" for r in range(5)]
    table = backtest.paired_bootstrap(pd.DataFrame(rows), "a", "b")
    assert (table["lo"] == 1).all() and (table["hi"] == 1).all()


def test_structural_problem_is_rare():
    p = config.STRUCTURAL_RATE
    assert p / (p + 1 / config.STRUCTURAL_PERSIST) < 0.15
