# Pricing dynamique par Q-learning

Projet pédagogique et expérimental de *revenue management* : vendre un stock
périssable en ajustant le prix dans le temps. Le problème est résolu deux fois — une
fois exactement par programmation dynamique (vérité terrain), une fois par un agent
Q-learning tabulaire qui n'observe jamais le modèle — pour valider empiriquement les
garanties théoriques du Q-learning sur un MDP qui satisfait exactement ses hypothèses
(états et actions finis, stationnarité, récompenses bornées).

Dépendances : NumPy, Matplotlib, tqdm (et pytest pour les tests). Aucun framework de
RL, pas de PyTorch, pas de Gym.

## Le modèle

Un vendeur dispose de `C` unités périssables à écouler sur `T` périodes. À chaque
période il affiche un prix `p` choisi dans une grille finie de `K` prix ; une vente
survient avec probabilité `lambda(p)`, strictement décroissante en `p`.

- **États** : `s = (c, t)`, stock restant et temps restant,
  `S = {0,...,C} x {0,...,T}`.
- **Actions** : `A = P`, la grille finie de prix.
- **Transitions** : avec probabilité `lambda(p)`, `(c, t) -> (c-1, t-1)` (vente) ;
  sinon `(c, t) -> (c, t-1)` (pas de vente).
- **Récompense** : `p` si vente, `0` sinon ; en espérance `lambda(p) * p`.
- **Absorption** : l'épisode se termine dès que `c = 0` (stock épuisé) ou `t = 0`
  (horizon atteint).

`lambda(p) = lambda0 * exp(-alpha * p)`, avec `lambda0` et `alpha` calibrés pour
que le revenu espéré par période `p * lambda(p)` ait un maximum **intérieur** à la
grille de prix — sans cet arbitrage prix/probabilité de vente, le problème serait
dégénéré (voir `src/demand.py::assert_interior_maximum`, appelé en tout premier
dans `notebooks/01_dp_exacte.ipynb`).

Paramètres par défaut (`src/config.py`) : `C=50`, `T=200`, `K=20` prix régulièrement
espacés sur `[50, 145]`, `lambda0=1.5`, `alpha=1/90` (maximum de revenu en `p*=90`,
au centre de la grille), `gamma=0.95`.

### Pourquoi c'est bien un MDP stationnaire fini

- **Markov** : la probabilité de vente ne dépend que du prix affiché, pas de
  l'historique ; en incluant le temps restant `t` dans l'état, la dynamique
  (probabilité de transition, récompense) ne dépend que de `(c, t, p)` — le
  processus est markovien même si `lambda(p)` seul ne l'est pas dans le temps.
- **Stationnarité obtenue en gonflant l'état** : le problème sous-jacent n'est pas
  stationnaire (il reste `t` périodes, un budget qui s'épuise), mais la
  transformation standard de l'horizon fini — mettre `t` dans l'état plutôt que de
  le traiter comme un indice externe — rend la dynamique `(c,t) -> (c',t')`
  elle-même stationnaire : la fonction de transition ne change pas d'une période à
  l'autre, ce qui est la condition requise par le Q-learning tabulaire (une seule
  table `Q(c,t,p)`, pas une table par période traitée séparément... en fait si, une
  ligne par période, mais dans le même tableau et avec la même règle de mise à
  jour).
- **Cardinalité finie** : `|S| = (C+1)(T+1)`, `|A| = K` ; avec les valeurs par
  défaut, `|S| x |A| ≈ 205 000` couples état-action — largement dans le régime où
  une table explicite est praticable et où les garanties de convergence du
  Q-learning tabulaire (Watkins & Dayan, 1992 ; conditions de Robbins-Monro sur le
  pas) s'appliquent sans approximation de fonction.
- **Acyclicité** : `t` décroît strictement à chaque transition, donc le graphe
  d'états est un DAG. Cela permet de résoudre le MDP exactement par une seule
  récursion arrière sur `t`, sans itération de valeur jusqu'à un point fixe.

## Structure du dépôt

```
src/
  config.py        paramètres du MDP et des expériences, dans une seule dataclass
  demand.py        lambda(p), vérification du maximum intérieur, tracés
  dp.py            programmation dynamique exacte + tests structurels
  qlearning.py     agent tabulaire, schémas de pas et d'exploration injectables
  evaluation.py    métriques, évaluation Monte-Carlo, agrégation multi-graines
notebooks/
  01_dp_exacte.ipynb     résolution DP, validation structurelle, heatmap de prix
  02_qlearning.ipynb     démonstration d'un agent unique, comparaison à la DP
  03_experiences.ipynb   l'expérience centrale (section suivante)
tests/             tests unitaires (pytest) de src/
outputs/           figures générées par les notebooks (gitignored)
```

Les notebooks n'importent que depuis `src/` et ne contiennent que l'orchestration et
les figures ; toute la logique (récursion DP, mise à jour Q-learning, métriques) est
dans `src/`, testée indépendamment dans `tests/`.

## Utilisation

```bash
pip install numpy matplotlib tqdm pytest
python -m pytest tests/ -q
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dp_exacte.ipynb
```

## L'expérience centrale (notebook 3)

Le point de tout le projet : `Q*` étant connue exactement, on peut mesurer
directement l'erreur de l'agent Q-learning à la vérité terrain pendant
l'apprentissage — une mesure normalement impossible en RL, où l'on ne dispose que
d'un proxy (la récompense cumulée). Toutes les courbes sont moyennées sur au moins
10 graines avec bande d'écart-type.

À chaque point d'évaluation on mesure :
- `||Q_n - Q*||_inf` et `||Q_n - Q*||_2`, restreintes aux états atteignables depuis
  `(C,T)` (les états non atteignables ou absorbants ne sont jamais mis à jour et
  fausseraient la mesure) ;
- le **regret de politique** `V^{pi*}(C,T) - V^{pi_n}(C,T)`, où `V^{pi_n}` est estimé
  par simulation Monte-Carlo de la politique gloutonne courante — seule façon de
  l'obtenir puisqu'elle n'a pas de forme close ;
- la fraction de couples `(c,t,p)` atteignables visités au moins une fois.

Quatre questions testables :

- **(a) Vitesse de convergence.** L'approximation stochastique prédit
  `O(n^{-1/2})` pour `||Q_n-Q*||_inf` sous un pas de type `1/n`. On ajuste une pente
  sur le tracé log-log et on la compare à `-0.5` — un écart modéré est attendu (la
  borne est asymptotique et les états rares convergent plus lentement), une absence
  totale de décroissance signalerait un bug.
- **(b) Schémas de pas.** `omega=1` satisfait les deux conditions de Robbins-Monro
  (`somme alpha_n = infini`, `somme alpha_n^2 < infini`) ; `omega=0.5` ne satisfait
  que la première ; le pas constant ne satisfait ni l'une ni l'autre. Prédiction
  testée : le pas constant ne converge pas mais oscille dans un voisinage de rayon
  `O(sqrt(alpha))`.
- **(c) Effet de epsilon.** Les états jamais visités gardent `Q=0`. On relie
  visuellement le taux d'exploration à la couverture de l'espace état-action et au
  regret final : trop peu d'exploration plafonne la couverture (et le regret), trop
  d'exploration ralentit l'exploitation pendant l'entraînement.
- **(d) Effet de gamma.** `||Q*||_inf` est borné par `p_max/(1-gamma)` : l'échelle de
  l'erreur absolue croît mécaniquement avec `gamma`. Une fois normalisée par cette
  borne, on observe aussi une dégradation de la vitesse de convergence quand
  `gamma -> 1`, cohérente avec un module de contraction de Bellman qui se rapproche
  de 1.

Pour que l'ensemble s'exécute en quelques minutes, le notebook 3 utilise un MDP plus
petit (`C`, `T`, `K` réduits) que celui des notebooks 1 et 2 : les phénomènes testés
ne dépendent pas de la taille de l'espace d'états, seule la vitesse d'exécution en
dépend.
