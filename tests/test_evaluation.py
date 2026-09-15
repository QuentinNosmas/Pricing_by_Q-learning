from dataclasses import replace

import numpy as np

from src.config import DEFAULT_CONFIG
from src.evaluation import (
    fit_loglog_slope,
    q_error_2,
    q_error_2_visited,
    q_error_inf,
    q_error_inf_visited,
    reachable_mask,
    simulate_policy_value,
    state_coverage,
)

SMALL_CFG = replace(DEFAULT_CONFIG, C=5, T=8, K=4)


def test_reachable_mask_excludes_absorbing_rows():
    mask = reachable_mask(SMALL_CFG)
    assert not mask[0, :].any()
    assert not mask[:, 0].any()


def test_reachable_mask_excludes_impossible_states():
    # depuis (C,T), on ne peut pas avoir perdu plus d'unités que de pas de temps écoulés
    mask = reachable_mask(SMALL_CFG)
    C, T = SMALL_CFG.C, SMALL_CFG.T
    # à t = T (0 pas écoulé), seul c = C est atteignable
    assert mask[C, T]
    assert not mask[C - 1, T]


def test_q_error_metrics_zero_when_equal():
    Q = np.random.default_rng(0).normal(size=(6, 6, 4))
    mask = reachable_mask(replace(DEFAULT_CONFIG, C=5, T=5, K=4))
    assert q_error_inf(Q, Q, mask) == 0.0
    assert q_error_2(Q, Q, mask) == 0.0


def test_state_coverage_bounds():
    cfg = SMALL_CFG
    mask = reachable_mask(cfg)
    visits = np.zeros((cfg.C + 1, cfg.T + 1, cfg.K), dtype=np.int64)
    assert state_coverage(visits, mask) == 0.0
    visits[mask] = 1
    assert state_coverage(visits, mask) == 1.0


def test_visited_error_ignores_unvisited_pairs():
    cfg = SMALL_CFG
    mask = reachable_mask(cfg)
    rng = np.random.default_rng(0)
    Q_star = rng.normal(size=(cfg.C + 1, cfg.T + 1, cfg.K))
    Q = np.zeros_like(Q_star)
    visits = np.zeros_like(Q_star, dtype=np.int64)

    # seul (c=2,t=3,a=1) est "visité" et appris exactement -> erreur visitée nulle,
    # alors que l'erreur globale reste dominée par tout ce qui n'a jamais été visité.
    visits[2, 3, 1] = 5
    Q[2, 3, 1] = Q_star[2, 3, 1]

    assert q_error_inf_visited(Q, Q_star, mask, visits) == 0.0
    assert q_error_2_visited(Q, Q_star, mask, visits) == 0.0
    assert q_error_inf(Q, Q_star, mask) > 0.0


def test_visited_error_is_nan_when_nothing_visited():
    cfg = SMALL_CFG
    mask = reachable_mask(cfg)
    Q = np.zeros((cfg.C + 1, cfg.T + 1, cfg.K))
    visits = np.zeros_like(Q, dtype=np.int64)
    assert np.isnan(q_error_inf_visited(Q, Q, mask, visits))
    assert np.isnan(q_error_2_visited(Q, Q, mask, visits))


def test_simulate_policy_value_matches_dp_for_optimal_policy():
    from src.dp import solve_dp

    cfg = SMALL_CFG
    Q_star, V_star, pi_idx = solve_dp(cfg)
    rng = np.random.default_rng(0)
    v_mc = simulate_policy_value(cfg, pi_idx, rng, n_episodes=20_000)
    v_true = V_star[cfg.C, cfg.T]
    assert abs(v_mc - v_true) / max(v_true, 1e-9) < 0.05


def test_fit_loglog_slope_recovers_known_exponent():
    x = np.arange(1, 1000)
    y = 3.0 * x ** (-0.5)
    slope = fit_loglog_slope(x, y)
    assert abs(slope - (-0.5)) < 1e-6
