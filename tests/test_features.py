import numpy as np
import pandas as pd

from f1pred.features import POST_QUALI, add_upcoming, build_features


def history(races=8):
    rng = np.random.default_rng(1)
    rows = []
    for r in range(1, races + 1):
        order = rng.permutation(6)
        for pos, d in enumerate(order, 1):
            rows.append({
                "year": 2026, "round": r, "date": f"2026-{r:02d}-01", "race": f"R{r}",
                "circuit": "monza", "driver": f"d{d}", "code": f"D{d}", "team": f"t{d // 2}",
                "grid": int(d) + 1, "quali_pos": int(d) + 1, "quali_time": 80.0 + d,
                "position": pos, "dnf": False, "points": 0.0,
            })
    return pd.DataFrame(rows)


def test_changing_a_race_result_does_not_change_its_own_or_earlier_features():
    base = history()
    changed = base.copy()
    last = changed["round"] == 8
    changed.loc[last, "position"] = changed.loc[last, "position"].to_numpy()[::-1]
    columns = [c for c in POST_QUALI if c not in {"grid_pct", "quali_gap"}]
    a, b = build_features(base), build_features(changed)
    pd.testing.assert_frame_equal(a[columns], b[columns])


def test_target_runs_from_winner_to_last_and_skips_retirements():
    df = history(1)
    df.loc[df["position"] == 6, "dnf"] = True
    out = build_features(df)
    assert out.loc[out["position"] == 1, "target"].item() == 0
    assert out.loc[out["position"] == 5, "target"].item() == 1
    assert out.loc[out["position"] == 6, "target"].isna().all()


def test_upcoming_race_keeps_its_qualifying_data():
    base = history()
    entrants = base[base["round"] == 8][["driver", "code", "team", "grid", "quali_pos", "quali_time"]]
    fields = {"year": 2026, "round": 9, "date": "2026-09-01", "race": "R9", "circuit": "monza"}
    out = build_features(add_upcoming(base, fields, entrants))
    race = out[out["round"] == 9]
    assert race["quali_gap"].notna().all() and race["position"].isna().all()
