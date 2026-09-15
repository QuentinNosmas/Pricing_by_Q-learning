from dataclasses import replace

import numpy as np

from src.config import DEFAULT_CONFIG
from src.dp import (
    check_concavity_in_c,
    check_price_monotonic_in_c,
    check_price_monotonic_in_t,
    check_value_monotonic_in_c,
    check_value_monotonic_in_t,
    solve_dp,
    validate_dp,
)

SMALL_CFG = replace(DEFAULT_CONFIG, C=6, T=10, K=5)


def test_shapes():
    Q, V, pi_idx = solve_dp(SMALL_CFG)
    assert Q.shape == (SMALL_CFG.C + 1, SMALL_CFG.T + 1, SMALL_CFG.K)
    assert V.shape == (SMALL_CFG.C + 1, SMALL_CFG.T + 1)
    assert pi_idx.shape == (SMALL_CFG.C + 1, SMALL_CFG.T + 1)


def test_boundary_conditions_are_zero():
    _, V, _ = solve_dp(SMALL_CFG)
    assert np.allclose(V[0, :], 0.0)
    assert np.allclose(V[:, 0], 0.0)


def test_structural_properties_hold_on_default_config():
    _, V, pi_idx = solve_dp(DEFAULT_CONFIG)
    checks = validate_dp(V, pi_idx, DEFAULT_CONFIG)
    assert all(checks.values())


def test_structural_properties_hold_on_small_config():
    _, V, pi_idx = solve_dp(SMALL_CFG)
    assert check_value_monotonic_in_c(V)
    assert check_value_monotonic_in_t(V)
    assert check_concavity_in_c(V)
    assert check_price_monotonic_in_c(SMALL_CFG.prices, pi_idx)
    assert check_price_monotonic_in_t(SMALL_CFG.prices, pi_idx)


def test_value_matches_bellman_equation_by_direct_recomputation():
    """Recalcule Q*(c,t,.) à la main pour un (c,t) arbitraire et compare."""
    cfg = SMALL_CFG
    Q, V, _ = solve_dp(cfg)
    from src.demand import lambda_p

    c, t = 4, 7
    prices = cfg.prices
    lam = lambda_p(prices, cfg)
    expected = lam * (prices + cfg.GAMMA * V[c - 1, t - 1]) + (1 - lam) * cfg.GAMMA * V[c, t - 1]
    assert np.allclose(Q[c, t], expected)


def test_v_star_never_exceeds_theoretical_bound():
    _, V, _ = solve_dp(DEFAULT_CONFIG)
    bound = DEFAULT_CONFIG.P_MAX / (1 - DEFAULT_CONFIG.GAMMA)
    assert np.all(V <= bound + 1e-6)
