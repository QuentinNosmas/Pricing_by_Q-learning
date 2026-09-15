"""Q-learning tabulaire.

L'agent n'accède à lambda(p) qu'à travers `sample_transition`, qui simule
l'environnement par un tirage de Bernoulli. Aucune autre fonction de ce module
ne lit `lambda(p)` : la mise à jour Q-learning n'utilise que la récompense et
l'état suivant échantillonnés.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.config import Config, DEFAULT_CONFIG
from src.demand import lambda_p

StepFn = Callable[[int], float]
EpsFn = Callable[[int], float]
EvalFn = Callable[[np.ndarray, np.ndarray], dict[str, float]]


# --------------------------------------------------------------------------
# Schémas de pas, injectables
# --------------------------------------------------------------------------


def polynomial_step(alpha0: float = 1.0, omega: float = 1.0) -> StepFn:
    """alpha_n = alpha0 / (1+n)^omega. omega=1 -> conditions de Robbins-Monro."""

    def step(n: int) -> float:
        return alpha0 / (1.0 + n) ** omega

    return step


def constant_step(alpha0: float = 0.1) -> StepFn:
    """Pas constant : ne satisfait pas somme(alpha_n^2) < infini, pas de convergence p.s."""

    def step(n: int) -> float:
        return alpha0

    return step


def linear_epsilon(n_episodes: int, eps_start: float = 1.0, eps_end: float = 0.05) -> EpsFn:
    """Epsilon décroissant linéairement de eps_start à eps_end sur n_episodes."""

    def eps(episode: int) -> float:
        frac = min(episode / max(n_episodes - 1, 1), 1.0)
        return eps_start + frac * (eps_end - eps_start)

    return eps


def constant_epsilon(eps_value: float) -> EpsFn:
    """Exploration epsilon-gloutonne à taux constant (pas de décroissance)."""

    def eps(episode: int) -> float:
        return eps_value

    return eps


# --------------------------------------------------------------------------
# Environnement (seul point de contact avec lambda)
# --------------------------------------------------------------------------


def sample_transition(
    rng: np.random.Generator, c: int, t: int, price_idx: int, cfg: Config = DEFAULT_CONFIG
) -> tuple[int, int, float, bool]:
    """Échantillonne une transition par tirage de Bernoulli(lambda(p)).

    C'est le seul endroit où lambda(p) est utilisé : l'agent ne l'observe
    jamais directement, seulement à travers la vente ou non-vente réalisée.
    """
    p = cfg.prices[price_idx]
    sale = rng.random() < lambda_p(p, cfg)
    c_next = c - 1 if sale else c
    t_next = t - 1
    reward = float(p) if sale else 0.0
    done = (c_next == 0) or (t_next == 0)
    return c_next, t_next, reward, done


def epsilon_greedy_action(rng: np.random.Generator, Q: np.ndarray, c: int, t: int, eps: float) -> int:
    if rng.random() < eps:
        return int(rng.integers(Q.shape[2]))
    return int(np.argmax(Q[c, t]))


# --------------------------------------------------------------------------
# Boucle d'apprentissage
# --------------------------------------------------------------------------


def run_qlearning(
    cfg: Config,
    n_episodes: int,
    step_fn: StepFn,
    eps_fn: EpsFn,
    rng: np.random.Generator,
    eval_every: int | None = None,
    eval_fn: EvalFn | None = None,
    Q_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Entraîne un agent Q-learning tabulaire depuis l'état (C, T).

    Si `eval_fn` est fourni, il est appelé tous les `eval_every` épisodes
    (et à l'épisode 0 et n_episodes) avec (Q, visit_counts) et doit retourner
    un dict de métriques scalaires ; ces dicts sont accumulés dans `history`.

    Returns
    -------
    Q : (C+1, T+1, K) table Q apprise.
    visit_counts : (C+1, T+1, K) nombre de mises à jour par couple état-action.
    history : liste de dicts {"episode": n, **metrics}.
    """
    C, T, K = cfg.C, cfg.T, cfg.K
    Q = np.zeros((C + 1, T + 1, K)) if Q_init is None else Q_init.copy()
    visit_counts = np.zeros((C + 1, T + 1, K), dtype=np.int64)
    history: list[dict] = []

    def maybe_eval(episode: int) -> None:
        if eval_fn is not None and eval_every is not None and episode % eval_every == 0:
            history.append({"episode": episode, **eval_fn(Q, visit_counts)})

    maybe_eval(0)

    for episode in range(1, n_episodes + 1):
        c, t = C, T
        eps = eps_fn(episode - 1)
        while c > 0 and t > 0:
            a = epsilon_greedy_action(rng, Q, c, t, eps)
            c_next, t_next, reward, done = sample_transition(rng, c, t, a, cfg)

            n = visit_counts[c, t, a]
            alpha_n = step_fn(int(n))
            bootstrap = 0.0 if done else cfg.GAMMA * np.max(Q[c_next, t_next])
            target = reward + bootstrap
            Q[c, t, a] += alpha_n * (target - Q[c, t, a])
            visit_counts[c, t, a] += 1

            c, t = c_next, t_next

        maybe_eval(episode)

    return Q, visit_counts, history
