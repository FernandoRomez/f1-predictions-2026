import numpy as np

from f1pred.model import points_for, simulate


def test_faster_driver_wins_more_and_each_run_is_a_valid_order():
    ranks = simulate([0.1, 0.5, 0.9], [0.05, 0.05, 0.05], sigma=0.2, runs=4000)
    assert (ranks == 0).mean(axis=0).argmax() == 0
    assert (np.sort(ranks, axis=1) == np.arange(3)).all()


def test_retirement_probability_moves_a_driver_to_the_back():
    ranks = simulate([0.0, 0.5], [1.0, 0.0], sigma=0.01, runs=200)
    assert (ranks[:, 0] == 1).all()


def test_points_follow_the_finishing_order():
    assert points_for(np.array([[0, 9, 10]])).tolist() == [[25.0, 1.0, 0.0]]


def test_team_mates_share_noise():
    ranks = simulate([0.5] * 4, [0.0] * 4, sigma=0.2, runs=4000, teams=["a", "a", "b", "b"])
    together = (np.abs(ranks[:, 0] - ranks[:, 1]) == 1).mean()
    assert together > 0.53  # 0.5 si fueran independientes
