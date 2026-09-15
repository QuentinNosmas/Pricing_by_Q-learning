"""Programmation dynamique exacte (vérité terrain) et tests structurels.

Le MDP est acyclique : t décroît strictement à chaque transition, donc une seule
passe arrière sur t = 1..T suffit (pas d'itération de valeur jusqu'à point fixe).
Tout est vectorisé sur c et p ; seule la boucle sur t est explicite.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.config import Config, DEFAULT_CONFIG
from src.demand import lambda_p


def solve_dp(cfg: Config = DEFAULT_CONFIG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Résout le MDP exactement par récursion arrière.

    Returns
    -------
    Q_star : (C+1, T+1, K) valeur état-action optimale.
    V_star : (C+1, T+1) valeur d'état optimale, V*(c,0) = V*(0,t) = 0.
    pi_idx : (C+1, T+1) indice de prix optimal (int), -1 aux états absorbants
             (c=0 ou t=0) où aucune action n'est prise.
    """
    C, T, K = cfg.C, cfg.T, cfg.K
    prices = cfg.prices
    lam = lambda_p(prices, cfg)  # (K,)
    gamma = cfg.GAMMA

    V = np.zeros((C + 1, T + 1))
    Q = np.zeros((C + 1, T + 1, K))
    pi_idx = np.full((C + 1, T + 1), -1, dtype=int)

    for t in range(1, T + 1):
        v_sale = V[0:C, t - 1]  # V*(c-1, t-1) pour c = 1..C
        v_no_sale = V[1:C + 1, t - 1]  # V*(c, t-1) pour c = 1..C

        q = lam[None, :] * (prices[None, :] + gamma * v_sale[:, None]) + (
            1.0 - lam[None, :]
        ) * gamma * v_no_sale[:, None]  # (C, K)

        Q[1:C + 1, t, :] = q
        V[1:C + 1, t] = q.max(axis=1)
        pi_idx[1:C + 1, t] = q.argmax(axis=1)

    return Q, V, pi_idx


# --------------------------------------------------------------------------
# Tests structurels : la théorie du revenue management prédit ces propriétés.
# Une violation signale un bug dans la récursion, pas une curiosité empirique.
# --------------------------------------------------------------------------


def check_value_monotonic_in_c(V: np.ndarray, tol: float = 1e-6) -> bool:
    """V*(c,t) croissante en c (plus de stock ne peut pas nuire)."""
    return bool(np.all(np.diff(V, axis=0) >= -tol))


def check_value_monotonic_in_t(V: np.ndarray, tol: float = 1e-6) -> bool:
    """V*(c,t) croissante en t (plus de temps ne peut pas nuire)."""
    return bool(np.all(np.diff(V, axis=1) >= -tol))


def check_concavity_in_c(V: np.ndarray, tol: float = 1e-6) -> bool:
    """Valeur marginale delta(c,t) = V*(c,t) - V*(c-1,t) décroissante en c."""
    delta = np.diff(V, axis=0)  # delta[c-1, t] = V(c,t) - V(c-1,t), c = 1..C
    second_diff = np.diff(delta, axis=0)
    return bool(np.all(second_diff <= tol))


def check_price_monotonic_in_c(prices: np.ndarray, pi_idx: np.ndarray, tol: float = 1e-9) -> bool:
    """pi*(c,t) décroissant en c à t fixé (plus de stock -> prix plus bas)."""
    pi_p = prices[np.clip(pi_idx, 0, None)]
    sub = pi_p[1:, 1:]  # restreint aux états non absorbants c>=1, t>=1
    return bool(np.all(np.diff(sub, axis=0) <= tol))


def check_price_monotonic_in_t(prices: np.ndarray, pi_idx: np.ndarray, tol: float = 1e-9) -> bool:
    """pi*(c,t) croissant en t à c fixé (plus de temps -> on peut être plus cher)."""
    pi_p = prices[np.clip(pi_idx, 0, None)]
    sub = pi_p[1:, 1:]
    return bool(np.all(np.diff(sub, axis=1) >= -tol))


def validate_dp(
    V: np.ndarray, pi_idx: np.ndarray, cfg: Config = DEFAULT_CONFIG, tol: float = 1e-6
) -> dict[str, bool]:
    """Lance tous les tests structurels ; lève AssertionError si l'un échoue."""
    prices = cfg.prices
    checks = {
        "V* croissante en c": check_value_monotonic_in_c(V, tol),
        "V* croissante en t": check_value_monotonic_in_t(V, tol),
        "V* concave en c": check_concavity_in_c(V, tol),
        "pi* décroissant en c": check_price_monotonic_in_c(prices, pi_idx, tol),
        "pi* croissant en t": check_price_monotonic_in_t(prices, pi_idx, tol),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(
            "Tests structurels DP échoués (signe probable d'un bug dans la "
            f"récursion) : {failed}"
        )
    return checks


def plot_price_heatmap(pi_idx: np.ndarray, cfg: Config = DEFAULT_CONFIG, savepath: str | None = None) -> Figure:
    """Carte de chaleur du prix optimal pi*(c,t) : la figure centrale du projet."""
    prices = cfg.prices
    pi_p = np.where(pi_idx >= 0, prices[np.clip(pi_idx, 0, None)], np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pi_p.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[0, cfg.C, 0, cfg.T],
    )
    ax.set_xlabel("stock restant c")
    ax.set_ylabel("temps restant t")
    ax.set_title(r"Prix optimal $\pi^*(c,t)$")
    fig.colorbar(im, ax=ax, label="prix optimal")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
