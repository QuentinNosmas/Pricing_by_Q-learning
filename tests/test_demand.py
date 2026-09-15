from dataclasses import replace

import numpy as np

from src.config import DEFAULT_CONFIG
from src.demand import assert_interior_maximum, has_interior_maximum, lambda_p


def test_lambda_is_decreasing():
    prices = DEFAULT_CONFIG.prices
    lam = lambda_p(prices, DEFAULT_CONFIG)
    assert np.all(np.diff(lam) < 0)


def test_lambda_is_a_valid_probability():
    prices = DEFAULT_CONFIG.prices
    lam = lambda_p(prices, DEFAULT_CONFIG)
    assert np.all(lam > 0) and np.all(lam < 1)


def test_default_config_has_interior_maximum():
    interior, argmax = has_interior_maximum(DEFAULT_CONFIG)
    assert interior
    assert 0 < argmax < DEFAULT_CONFIG.K - 1
    assert_interior_maximum(DEFAULT_CONFIG)  # ne doit pas lever


def test_degenerate_config_is_detected():
    # ALPHA très petit -> revenu quasi croissant sur toute la grille -> max au bord
    degenerate = replace(DEFAULT_CONFIG, ALPHA=1e-6)
    interior, _ = has_interior_maximum(degenerate)
    assert not interior
    try:
        assert_interior_maximum(degenerate)
        assert False, "aurait dû lever ValueError"
    except ValueError:
        pass
