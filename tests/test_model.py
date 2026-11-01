import numpy as np

from f1pred.model import blend_win, points_for, simulate


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


def test_blend_win_is_a_distribution_and_interpolates():
    model = np.array([0.8, 0.15, 0.05])
    grid = np.array([0.4, 0.5, 0.1])
    b = blend_win(model, grid, 0.5)
    assert np.isclose(b.sum(), 1.0)
    assert (b > 0).all()
    # alpha=1 devuelve el modelo, alpha=0 la parrilla.
    assert np.allclose(blend_win(model, grid, 1.0), model)
    assert np.allclose(blend_win(model, grid, 0.0), grid)


def test_incident_factor_raises_retirements_for_everyone():
    base = simulate([0.0, 0.5, 0.9], [0.05] * 3, sigma=0.01, runs=8000)
    with_incident = simulate([0.0, 0.5, 0.9], [0.05] * 3, sigma=0.01, runs=8000,
                             incident=np.full(8000, 3.0))
    assert (base == 0).mean(axis=0)[0] > (with_incident == 0).mean(axis=0)[0]


def test_structural_dnf_scale_raises_one_driver_dnf():
    lo = simulate([0.0, 0.5, 0.5], [0.05] * 3, sigma=0.01, runs=8000,
                  dnf_scale=np.array([1.0, 1.0, 1.0]))
    hi = simulate([0.0, 0.5, 0.5], [0.05] * 3, sigma=0.01, runs=8000,
                  dnf_scale=np.array([10.0, 1.0, 1.0]))
    assert (hi[:, 0] == 2).mean() > (lo[:, 0] == 2).mean()


def test_grid_prior_follows_history():
    import pandas as pd
    from f1pred.model import grid_win_rate
    past = pd.DataFrame({"position": [1, 2, 1, 2], "grid": [1, 2, 1, 2]})
    prior = grid_win_rate(past, pd.DataFrame({"grid": [1, 2]}))
    assert prior[0] > prior[1] and np.isclose(prior.sum(), 1)
