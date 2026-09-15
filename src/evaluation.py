"""Métriques de convergence, évaluation Monte-Carlo, agrégation multi-graines.

Rien ici ne lit lambda(p) directement pour l'agent : la fonction `simulate_policy_value`
simule l'environnement vrai (elle a le droit, ce n'est pas l'agent) pour estimer par
Monte-Carlo la valeur d'une politique gloutonne apprise, seule façon d'obtenir le
regret de politique puisque V^{pi_n} n'est pas connu en forme close.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.config import Config, DEFAULT_CONFIG
from src.demand import lambda_p
from src.qlearning import EvalFn


def reachable_mask(cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """États (c,t) non absorbants atteignables depuis (C,T) en au plus T-t pas.

    Exclut aussi les lignes absorbantes c=0 et t=0, où aucune action Q-learning
    n'est jamais prise (Q y reste à sa valeur d'initialisation par construction).
    """
    c = np.arange(cfg.C + 1)[:, None]
    t = np.arange(cfg.T + 1)[None, :]
    mask = (cfg.T - t) >= (cfg.C - c)
    mask &= (c >= 1) & (t >= 1)
    return mask


def q_error_inf(Q: np.ndarray, Q_star: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(Q - Q_star)
    return float(diff[mask].max())


def q_error_2(Q: np.ndarray, Q_star: np.ndarray, mask: np.ndarray) -> float:
    diff = (Q - Q_star)[mask]
    return float(np.sqrt(np.sum(diff ** 2)))


def _visited_mask(mask: np.ndarray, visit_counts: np.ndarray) -> np.ndarray:
    """Triplets (c,t,p) à la fois atteignables et visités au moins une fois.

    Distinct de `mask` seul : certains coins de l'espace d'états ne sont
    atteignables qu'au prix d'une longue suite d'événements rares (plusieurs
    ventes consécutives), et restent à Q=0 même après beaucoup d'épisodes. Le
    mesurer séparément permet de distinguer "dette de couverture" (traitée en
    (c)) de la vitesse de convergence proprement dite sur ce qui a été visité.
    """
    return np.broadcast_to(mask[:, :, None], visit_counts.shape) & (visit_counts > 0)


def q_error_inf_visited(Q: np.ndarray, Q_star: np.ndarray, mask: np.ndarray, visit_counts: np.ndarray) -> float:
    visited = _visited_mask(mask, visit_counts)
    if not visited.any():
        return float("nan")
    return float(np.abs(Q - Q_star)[visited].max())


def q_error_2_visited(Q: np.ndarray, Q_star: np.ndarray, mask: np.ndarray, visit_counts: np.ndarray) -> float:
    visited = _visited_mask(mask, visit_counts)
    if not visited.any():
        return float("nan")
    diff = (Q - Q_star)[visited]
    return float(np.sqrt(np.sum(diff ** 2)))


def state_coverage(visit_counts: np.ndarray, mask: np.ndarray) -> float:
    """Fraction des triplets (c,t,p) atteignables visités au moins une fois."""
    mask3 = np.broadcast_to(mask[:, :, None], visit_counts.shape)
    visited = (visit_counts > 0) & mask3
    return float(visited.sum() / mask3.sum())


def simulate_policy_value(
    cfg: Config,
    price_idx_policy: np.ndarray,
    rng: np.random.Generator,
    n_episodes: int = 300,
) -> float:
    """Valeur Monte-Carlo de la politique déterministe `price_idx_policy` depuis (C,T).

    Vectorisé sur les n_episodes trajectoires simultanément (au plus T pas).
    """
    C, T = cfg.C, cfg.T
    prices = cfg.prices

    c = np.full(n_episodes, C, dtype=int)
    t = np.full(n_episodes, T, dtype=int)
    returns = np.zeros(n_episodes)
    discount = np.ones(n_episodes)
    active = np.ones(n_episodes, dtype=bool)

    for _ in range(T):
        if not active.any():
            break
        idx = np.where(active)[0]
        a = price_idx_policy[c[idx], t[idx]]
        a = np.clip(a, 0, cfg.K - 1)  # sécurité si politique non définie (ne devrait pas arriver)
        p = prices[a]
        sale = rng.random(idx.shape[0]) < lambda_p(p, cfg)
        reward = np.where(sale, p, 0.0)

        returns[idx] += discount[idx] * reward
        discount[idx] *= cfg.GAMMA

        c_new = c[idx] - sale.astype(int)
        t_new = t[idx] - 1
        c[idx] = c_new
        t[idx] = t_new

        done = (c_new == 0) | (t_new == 0)
        active[idx[done]] = False

    return float(returns.mean())


def make_evaluator(
    cfg: Config,
    Q_star: np.ndarray,
    V_star: np.ndarray,
    rng: np.random.Generator,
    n_mc_episodes: int = 300,
) -> EvalFn:
    """Construit le callback d'évaluation passé à `run_qlearning`."""
    mask = reachable_mask(cfg)
    v_opt = float(V_star[cfg.C, cfg.T])

    def evaluate(Q: np.ndarray, visit_counts: np.ndarray) -> dict[str, float]:
        pi_idx = np.argmax(Q, axis=2)
        v_pi = simulate_policy_value(cfg, pi_idx, rng, n_mc_episodes)
        return {
            "q_error_inf": q_error_inf(Q, Q_star, mask),
            "q_error_2": q_error_2(Q, Q_star, mask),
            "q_error_inf_visited": q_error_inf_visited(Q, Q_star, mask, visit_counts),
            "q_error_2_visited": q_error_2_visited(Q, Q_star, mask, visit_counts),
            "coverage": state_coverage(visit_counts, mask),
            "regret": v_opt - v_pi,
        }

    return evaluate


def run_multi_seed(
    cfg: Config,
    n_seeds: int,
    n_episodes: int,
    step_fn: Callable[[int], float],
    eps_fn: Callable[[int], float],
    Q_star: np.ndarray,
    V_star: np.ndarray,
    eval_every: int,
    n_mc_episodes: int = 300,
    seed0: int = 0,
) -> dict:
    """Répète `run_qlearning` sur n_seeds graines et agrège moyenne/écart-type.

    Returns a dict {"episodes": array, metric_name: {"mean", "std", "raw"}}.
    step_fn et eps_fn sont des fonctions pures de l'indice de visite / de
    l'épisode : elles sont sans état et peuvent être partagées entre graines.
    """
    from src.qlearning import run_qlearning  # import local pour éviter un cycle

    histories = []
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        evaluator = make_evaluator(cfg, Q_star, V_star, rng, n_mc_episodes)
        _, _, history = run_qlearning(cfg, n_episodes, step_fn, eps_fn, rng, eval_every, evaluator)
        histories.append(history)

    episodes = np.array([h["episode"] for h in histories[0]])
    metric_names = [k for k in histories[0][0] if k != "episode"]

    agg: dict = {"episodes": episodes}
    for name in metric_names:
        arr = np.array([[h[name] for h in hist] for hist in histories])  # (n_seeds, n_checkpoints)
        agg[name] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0), "raw": arr}
    return agg


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Pente de régression log(y) ~ log(x), ignorant les valeurs non positives."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = (x > 0) & (y > 0)
    slope, _ = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
    return float(slope)
