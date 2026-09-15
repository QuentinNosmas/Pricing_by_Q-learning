from dataclasses import replace

import numpy as np

from src.config import DEFAULT_CONFIG
from src.dp import solve_dp
from src.evaluation import make_evaluator, reachable_mask, q_error_inf
from src.qlearning import (
    constant_epsilon,
    constant_step,
    epsilon_greedy_action,
    linear_epsilon,
    polynomial_step,
    run_qlearning,
    sample_transition,
)

SMALL_CFG = replace(DEFAULT_CONFIG, C=5, T=8, K=4)


def test_polynomial_step_decreases_and_is_summable_squared_when_omega_1():
    step = polynomial_step(alpha0=1.0, omega=1.0)
    values = np.array([step(n) for n in range(1000)])
    assert np.all(np.diff(values) <= 0)
    # sum alpha_n^2 doit converger pour omega=1 (comparaison à une somme partielle stable)
    assert np.sum(values ** 2) < 10


def test_constant_step_is_constant():
    step = constant_step(alpha0=0.3)
    assert all(step(n) == 0.3 for n in [0, 1, 100, 10_000])


def test_linear_epsilon_bounds():
    eps = linear_epsilon(1000, eps_start=1.0, eps_end=0.05)
    assert eps(0) == 1.0
    assert abs(eps(999) - 0.05) < 1e-9
    assert eps(500) < 1.0 and eps(500) > 0.05


def test_sample_transition_is_bernoulli_consistent():
    rng = np.random.default_rng(0)
    cfg = SMALL_CFG
    n = 20_000
    sales = 0
    price_idx = 0  # prix le plus bas -> probabilité de vente la plus haute
    from src.demand import lambda_p

    expected_p = lambda_p(cfg.prices[price_idx], cfg)
    for _ in range(n):
        c_next, t_next, reward, done = sample_transition(rng, 3, 5, price_idx, cfg)
        sales += reward > 0
    assert abs(sales / n - expected_p) < 0.02


def test_sample_transition_terminates_correctly():
    rng = np.random.default_rng(0)
    cfg = SMALL_CFG
    # t=1 -> t_next=0 -> toujours done
    _, t_next, _, done = sample_transition(rng, 3, 1, 0, cfg)
    assert t_next == 0 and done


def test_constant_epsilon_is_constant():
    eps = constant_epsilon(0.2)
    assert all(eps(n) == 0.2 for n in [0, 1, 500])


def test_epsilon_greedy_is_greedy_when_eps_zero():
    rng = np.random.default_rng(0)
    Q = np.zeros((6, 6, 4))
    Q[2, 3, 2] = 5.0
    assert epsilon_greedy_action(rng, Q, 2, 3, eps=0.0) == 2


def test_run_qlearning_updates_visited_states_only():
    cfg = SMALL_CFG
    rng = np.random.default_rng(0)
    step_fn = polynomial_step(1.0, 1.0)
    eps_fn = linear_epsilon(200, 1.0, 0.05)
    Q, visits, history = run_qlearning(cfg, 200, step_fn, eps_fn, rng)
    assert visits.sum() > 0
    assert np.all((Q != 0) <= (visits > 0))  # jamais visité -> jamais mis à jour


def test_run_qlearning_reduces_q_error_towards_q_star():
    cfg = replace(DEFAULT_CONFIG, C=5, T=8, K=4, N_EPISODES=6000)
    Q_star, V_star, _ = solve_dp(cfg)
    mask = reachable_mask(cfg)

    rng = np.random.default_rng(1)
    step_fn = polynomial_step(1.0, 1.0)
    eps_fn = linear_epsilon(cfg.N_EPISODES, 1.0, 0.05)
    evaluator = make_evaluator(cfg, Q_star, V_star, rng, n_mc_episodes=100)

    Q, visits, history = run_qlearning(
        cfg, cfg.N_EPISODES, step_fn, eps_fn, rng, eval_every=cfg.N_EPISODES // 2, eval_fn=evaluator
    )
    errors = [h["q_error_inf"] for h in history]
    assert errors[-1] < errors[0]
    assert history[-1]["coverage"] > 0.8
