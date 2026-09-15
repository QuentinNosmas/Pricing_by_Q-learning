"""Courbe de demande lambda(p) et vérification qu'un arbitrage prix existe.

Cette vérification est une étape obligatoire avant tout le reste : si le revenu
espéré p*lambda(p) est monotone sur la grille de prix, il n'y a aucun compromis
prix/probabilité de vente à apprendre et le MDP tout entier est dégénéré.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.config import Config, DEFAULT_CONFIG


def lambda_p(p: np.ndarray | float, cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """Probabilité de vente à prix p, strictement décroissante en p."""
    return cfg.LAMBDA0 * np.exp(-cfg.ALPHA * np.asarray(p, dtype=float))


def expected_revenue(p: np.ndarray | float, cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """Revenu espéré par période p * lambda(p)."""
    return np.asarray(p, dtype=float) * lambda_p(p, cfg)


def has_interior_maximum(cfg: Config = DEFAULT_CONFIG) -> tuple[bool, int]:
    """Teste si l'argmax du revenu espéré est strictement à l'intérieur de la grille."""
    rev = expected_revenue(cfg.prices, cfg)
    argmax = int(np.argmax(rev))
    interior = 0 < argmax < cfg.K - 1
    return interior, argmax


def assert_interior_maximum(cfg: Config = DEFAULT_CONFIG) -> int:
    """Échoue bruyamment si le revenu espéré n'a pas de maximum intérieur."""
    interior, argmax = has_interior_maximum(cfg)
    if not interior:
        raise ValueError(
            "Le revenu espéré p*lambda(p) est monotone sur la grille de prix "
            f"(argmax à l'indice {argmax}/{cfg.K - 1}) : aucun arbitrage prix/probabilité "
            "de vente n'existe, le MDP est dégénéré. Recalibrer LAMBDA0 et ALPHA, par "
            "exemple en choisissant ALPHA ~ 1/p_cible avec p_cible au centre de "
            "[P_MIN, P_MAX]."
        )
    return argmax


def plot_demand_curve(cfg: Config = DEFAULT_CONFIG, savepath: str | None = None) -> Figure:
    """Trace lambda(p) et p*lambda(p) sur la grille de prix."""
    prices = cfg.prices
    lam = lambda_p(prices, cfg)
    rev = expected_revenue(prices, cfg)
    interior, argmax = has_interior_maximum(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(prices, lam, marker="o")
    axes[0].set_xlabel("prix p")
    axes[0].set_ylabel(r"$\lambda(p)$")
    axes[0].set_title("Probabilité de vente")
    axes[0].set_ylim(0, 1)

    axes[1].plot(prices, rev, marker="o")
    axes[1].axvline(prices[argmax], color="red", ls="--", label=f"argmax = {prices[argmax]:.1f}")
    axes[1].set_xlabel("prix p")
    axes[1].set_ylabel(r"$p \cdot \lambda(p)$")
    axes[1].set_title("Revenu espéré par période")
    axes[1].legend()

    title = "Maximum intérieur trouvé" if interior else "PAS de maximum intérieur — MDP dégénéré !"
    fig.suptitle(title)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
