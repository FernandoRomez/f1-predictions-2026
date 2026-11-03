import numpy as np
import pandas as pd

from f1pred.skill import COLUMNS, add_skill


def seasons():
    """"ace" es mejor que sus compañeros aunque cambie a un coche peor."""
    rng = np.random.default_rng(3)
    rows = []
    for year, ace_team in ((2024, "fast"), (2025, "slow")):
        for r in range(1, 13):
            race_id = year * 100 + r
            line = [("ace", ace_team), ("mate_a", "fast"), ("mate_b", "slow"), ("mate_c", "mid"), ("mate_d", "mid")]
            for driver, team in line:
                car = {"fast": 0.1, "mid": 0.5, "slow": 0.9}[team]
                target = car - (0.2 if driver == "ace" else 0) + rng.normal(0, 0.02)
                rows.append({"year": year, "race_id": race_id, "driver": driver, "team": team,
                             "target": target, "quali_gap": max(target, 0) * 0.02})
    return pd.DataFrame(rows)


def test_skill_is_separated_from_the_car():
    df = add_skill(seasons(), half_life=40, ridge=1.0)
    last = df[df["race_id"] == df["race_id"].max()].set_index("driver")
    assert last.loc["ace", "driver_skill_race"] == last["driver_skill_race"].min()


def test_skill_does_not_use_its_own_or_later_races():
    base = seasons()
    changed = base.copy()
    cut = base["race_id"].max()
    changed.loc[changed["race_id"] == cut, ["target", "quali_gap"]] = 0.0, 0.03
    a, b = add_skill(base, 20, 1.0), add_skill(changed, 20, 1.0)
    pd.testing.assert_frame_equal(a[COLUMNS], b[COLUMNS])


def test_first_race_and_unseen_data_stay_at_the_prior():
    df = add_skill(seasons(), 20, 1.0)
    assert (df.loc[df["race_id"] == df["race_id"].min(), COLUMNS] == 0).all().all()
