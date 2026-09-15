

"""Configuration centrale du problème de pricing dynamique.

Toute la paramétrisation du MDP et des expériences vit ici, dans une unique
dataclass. Créer une config alternative (pour une expérience de sensibilité,
p. ex. sur gamma) se fait avec `dataclasses.replace(DEFAULT_CONFIG, GAMMA=0.99)`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    # --- MDP ---
    C: int = 50  # stock initial
    T: int = 200  # horizon (nombre de périodes)
    K: int = 20  # taille de la grille de prix
    P_MIN: float = 50.0
    P_MAX: float = 145.0
    # lambda(p) = LAMBDA0 * exp(-ALPHA * p) ; calibré pour que p*lambda(p)
    # ait un maximum intérieur en p* = 1/ALPHA = 90, à l'intérieur de [P_MIN, P_MAX].
    LAMBDA0: float = 1.5
    ALPHA: float = 1.0 / 90.0
    GAMMA: float = 0.95

    # --- Q-learning ---
    EPS_START: float = 1.0
    EPS_END: float = 0.05
    STEP_ALPHA0: float = 1.0
    STEP_OMEGA: float = 1.0  # exposant par défaut du pas polynomial

    # --- Expériences ---
    N_EPISODES: int = 20_000
    EVAL_EVERY: int = 200
    N_MC_EPISODES: int = 300  # épisodes Monte-Carlo pour estimer V^{pi_n}
    N_SEEDS: int = 10
    SEED0: int = 0

    @property
    def prices(self) -> np.ndarray:
        """Grille de prix régulièrement espacée, taille K."""
        return np.linspace(self.P_MIN, self.P_MAX, self.K)


DEFAULT_CONFIG = Config()
