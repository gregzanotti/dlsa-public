# Deep Learning Statistical Arbitrage : explication du dépôt

Ce dépôt est le code officiel du papier *Deep Learning Statistical Arbitrage*
(Guijarro-Ordonez, Pelger, Zanotti ; [arXiv:2106.04028](https://arxiv.org/abs/2106.04028)).
Ce document explique ce que fait le code, comment il est organisé et comment le lancer sur un
Mac Apple Silicon avec l'environnement préparé pour Python 3.13.

## 1. L'idée

L'arbitrage statistique consiste à acheter des actifs temporairement « bon marché » et vendre des
actifs « chers » relativement à des actifs similaires, en pariant sur un retour à l'équilibre.
Le papier découpe le problème en trois étapes :

```
rendements des actions ──► (1) modèle factoriel ──► résidus ε(t, i)
                                                       │
                          (2) extraction du signal ◄───┘  fenêtre de L=30 jours
                                       │
                          (3) allocation ──► poids w(t, i) ──► rendement du portefeuille
```

1. **Construire des portefeuilles d'arbitrage (résidus).** On retire à chaque action la part de
   son rendement expliquée par K facteurs de risque. Le résidu `ε(t, i)` est le rendement d'un
   portefeuille « action i moins son exposition aux facteurs ». Il est (en principe) peu corrélé au
   marché et a tendance à revenir vers sa moyenne. Trois familles de facteurs sont utilisées :
   - **Fama-French** : facteurs observés (marché, taille, valeur, profitabilité, investissement,
     momentum, reversals court et long terme). Les betas sont estimés par régression glissante
     sur 60 jours.
   - **PCA** : facteurs latents, issus des composantes principales de la matrice de corrélation
     des rendements (fenêtre de 252 jours).
   - **IPCA** (*Instrumented PCA*) : facteurs latents dont les betas dépendent des
     caractéristiques des entreprises (46 caractéristiques mensuelles).

   Pour chaque famille, on utilise K ∈ {0, 1, 3, 5, 8, 10, 15} facteurs (K = 0 : rendement brut).
   Tout est calculé **hors échantillon** : à la date t, seules les données passées servent.

2. **Extraire un signal temporel.** Pour chaque résidu, on regarde la trajectoire cumulée des
   `lookback` = 30 derniers jours et on en tire une représentation :
   - `preprocess_ou` : on ajuste un processus d'**Ornstein-Uhlenbeck** (AR(1) sur la série cumulée)
     et on garde 4 caractéristiques : dernière valeur, moyenne d'équilibre μ, volatilité σ et R².
     C'est la version paramétrique classique (Avellaneda & Lee, 2010) ;
   - `preprocess_fourier` : coefficients de Fourier (parties réelle et imaginaire) de la fenêtre ;
   - `preprocess_cumsum` : la trajectoire cumulée brute, que le modèle apprend à lire lui-même.

3. **Allouer.** Un réseau de neurones prend le signal de chaque résidu et renvoie un poids. Les
   poids du jour sont normalisés pour que `Σ|w| = 1` (exposition brute unitaire). Le rendement du
   portefeuille est `Σ w(t-1, i) · ε(t, i)`.

**Point clé du papier :** le réseau est entraîné **directement sur l'objectif financier**, pas sur une
prévision de rendement. La perte est le ratio de Sharpe négatif du portefeuille sur la fenêtre
d'entraînement (`objective: "sharpe"`). Les alternatives sont `meanvar` (moyenne-variance) et
`sqrtMeanSharpe`. Des coûts de transaction (`trans_cost`, sur le turnover) et de détention
de positions courtes (`hold_cost`) peuvent être retranchés.

Le modèle principal du papier est le **CNN+Transformer** : convolutions pour détecter des motifs
locaux (tendances, retournements) dans la trajectoire, puis attention pour combiner ces motifs
sur toute la fenêtre. Les autres modèles servent de comparaisons.

## 2. Organisation du code

| Fichier / dossier | Rôle |
|---|---|
| `run_train_test.py` | Point d'entrée : lit une config YAML, charge les résidus, appelle `test()` ou `estimate()`, sauvegarde les résultats (`results/<modèle>/*.pickle`) et les logs (`logs/`). |
| `train_test.py` | Cœur du code. `train()` = boucle d'entraînement sur l'objectif Sharpe ; `get_returns()` = évaluation hors échantillon ; `test()` = backtest glissant complet ; `estimate()` = un seul entraînement/test (exploration d'hyperparamètres). |
| `preprocess.py` | Les 3 extractions de signal (OU, Fourier, cumsum). Entrée `T×N`, sortie `(T-L)×N×features` + masque des actifs utilisables. |
| `models/` | Les 4 modèles d'allocation (`nn.Module` PyTorch), voir ci-dessous. |
| `factor_models/` | Construction des résidus (PCA, IPCA, Fama-French) à partir des données brutes, **non fournies** (CRSP/Compustat sous licence). |
| `run_factor_model.py` | Lance un modèle factoriel pour régénérer les résidus. Inutilisable sans les données brutes. |
| `residuals/` | Résidus déjà calculés, compressés en `.npy.gz` (21 fichiers ≈ 40 Mo chacun) et `superMask.npy`. |
| `configs/` | Une config par modèle d'allocation, avec les hyperparamètres du papier. |
| `data.py` | `perturb()` : ajout de bruit gaussien aux résidus (test de robustesse). |
| `utils.py` | Logs, chargement numpy avec progression, sélection de GPU (`nvidia-smi`), notifications SMS Twilio. |

### Les modèles (`models/`)

| Modèle | Préprocessing | Architecture |
|---|---|---|
| `OUFFN` | `preprocess_ou` (4 features) | Petit réseau dense (4→4→4→4→1, sigmoïde) : allocation non linéaire sur le signal OU paramétrique. |
| `FourierFFN` | `preprocess_fourier` (30 features) | Réseau dense (30→16→8→4→1, ReLU) sur les coefficients de Fourier. |
| `RawFFN` | `preprocess_cumsum` (30 features) | Réseau dense sur la trajectoire brute. |
| `CNNTransformer` | `preprocess_cumsum` (30 pas de temps) | Blocs convolutifs résiduels 1D (1→8 filtres, normalisation d'instance), un `TransformerEncoderLayer` (4 têtes) sur la dimension temps, puis une couche linéaire sur le dernier pas de temps. |

Chaque modèle traite **chaque actif indépendamment**, avec les mêmes paramètres partagés par tous
les actifs. L'entrée est `(N_actifs, features)` et la sortie un poids par actif.

### Les données

Chaque fichier de résidus est une matrice `float64` de **4781 jours × 9483 actifs**
(jours de bourse du 2 janvier 1998 au 30 décembre 2016). Un `0` code une donnée manquante :
l'actif n'est pas dans l'univers ce jour-là. L'univers retenu (`cap_proportion: 0.01`) contient
les actions dont la capitalisation dépasse 0,01 % de la capitalisation totale.

Le nom des fichiers encode les paramètres du modèle factoriel, par exemple
`AvPCA_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_252_covWindow_0.01_Cap`
= PCA, 5 facteurs, hors échantillon à partir de 1998, régression sur 60 jours, covariance sur 252 jours.

Les **matrices de composition** des résidus (`*Matrix*`, qui donnent l'exposition de chaque
résidu aux actions réelles) **ne sont pas fournies**. Elles servent à calculer le turnover et la
part de positions courtes en espace « actions » (`use_residual_weights: True`), ce qui est
nécessaire pour reproduire exactement les chiffres du papier. Sans elles, ces métriques sont
calculées en espace « résidus ».

## 3. Le backtest glissant (`train_test.test`)

```
jours :  0 ─────────── 1000 ── 1125 ── 1250 ── ... ── 4781
         [ entraînement ][ test ]
                 [ entraînement ][ test ]
                         [ entraînement ][ test ]   ...
```

- On ne garde que les actifs ayant au moins `lookback` observations non nulles.
- Pour chaque sous-période t (≈ 31 sous-périodes avec les valeurs par défaut) :
  - entraînement sur `length_training` = 1000 jours (≈ 4 ans), pendant `num_epochs` = 100 époques,
    par mini-lots de `batch_size` = 125 jours consécutifs (Adam, lr = 0,001) ;
  - test sur les `retrain_freq` = 125 jours suivants (≈ 6 mois), avec un nouveau modèle à chaque
    sous-période si `rolling_retrain: True`.
- On concatène les rendements de test, soit environ 15 ans de rendements quotidiens hors échantillon
  (2002 à 2016).
- Les métriques sont annualisées : Sharpe × √252, rendement moyen × 252, volatilité × √252.
  On calcule aussi le turnover moyen et la part de positions courtes.

Sorties dans `results/<modèle>/` :
- `results_<tag>_<results_tag>.pickle` : rendements quotidiens, Sharpe, rendement, volatilité,
  turnover, part courte et config, pour chaque jeu de résidus ;
- `<model_tag>_cumulative-returns.png`, `_turnover.png`, `_short-proportion.png` ;
- `WeightsComplete_<model_tag>.npy` : poids quotidiens (jours de test × actifs) ;
- `Checkpoint-*.tar` : paramètres du modèle pour chaque sous-période.

## 4. Lancer le projet sur un Mac Apple Silicon

### Environnement

Le `requirements.txt` d'origine (numpy 1.21, torch 1.9, etc.) date de 2021, cible Python ≤ 3.9
et CUDA, et ne s'installe pas avec Python 3.13 sur ARM64. Il est conservé tel quel. Un fichier
séparé, `requirements-macos-arm64.txt`, fixe des versions compatibles :

```bash
uv venv --python 3.13 .venv
uv pip install -r requirements-macos-arm64.txt --python .venv/bin/python
```

| Dépendance | Original | macOS ARM64 / Py 3.13 | Remarque |
|---|---|---|---|
| numpy | 1.21.2 | ≥ 2.1, < 3 | `np.bool` (utilisé dans `train_test.py`) existe de nouveau depuis numpy 2.0 |
| pandas | 1.3.2 | ≥ 2.2.3, < 3 | pandas 3 évité (copy-on-write, dtype string) |
| torch | 1.9.0 | ≥ 2.5 | CPU et MPS (GPU Apple) |
| matplotlib | 3.4.3 | ≥ 3.9, < 3.11 | `plt.plot_date` supprimé en 3.11 |
| scikit-learn | 0.24.2 | ≥ 1.5.2 | |
| PyYAML | 5.4.1 | ≥ 6.0.2 | |
| psutil | 5.8.0 | ≥ 6.1 | |
| gpuinfo | 1.0.0a7 | identique | importé par `utils.py`, inutile sans `nvidia-smi` |
| twilio | 6.63.1 | ≥ 9 | optionnel (notifications SMS) |

### Lanceur local (`local/run_local.py`)

`run_train_test.py` exige CUDA : `device='cuda:N'`, `nn.DataParallel` sur plusieurs GPU et
détection des GPU via `nvidia-smi`. Pour ne pas modifier le code d'origine,
`local/run_local.py` remplace ces appels au moment de l'exécution, puis appelle
`run_train_test.run()` sans modification :

- `test()` / `estimate()` sont appelés avec `device="cpu"` ou `"mps"` et `parallelize=False` ;
- le fichier de dates Fama-French (`data/F-F_Research_Data_5_Factors_2x3_daily.CSV`) est
  préparé à l'avance : le fichier publié par Kenneth French a changé de format (préambule plus
  long, ligne de copyright finale) et le parseur d'origine (`header=2`) échoue ;
- la détection d'anomalies autograd (activée à l'import dans le code d'origine, très lente) est
  désactivée par défaut (`--detect-anomaly` pour la garder) ; elle ne change pas les résultats ;
- `use_residual_weights: True` est refusé, car les matrices ne sont pas disponibles.

```bash
# config complète du papier, sur CPU
.venv/bin/python local/run_local.py -c configs/ouffn-full.yaml

# essai rapide : 1 époque, un seul jeu de résidus
.venv/bin/python local/run_local.py -c configs/ouffn-full.yaml --num-epochs 1 --factor-model PCA:5

# GPU Apple, autres résidus
.venv/bin/python local/run_local.py -c configs/cnntransformer-full.yaml --device mps --factor-model IPCA:5
```

Au premier passage, `run()` décompresse chaque `.npy.gz` utilisé en un `.npy` à côté
(≈ 363 Mo chacun, ignorés par git). Les fichiers `.gz` d'origine ne sont pas modifiés.

### Notebook

`notebooks/dlsa_walkthrough.ipynb` déroule tout le pipeline : exploration des résidus (univers,
dispersion, autocorrélations), illustration des 3 préprocessings, architectures et vitesse CPU/MPS des
4 modèles, lancement ou rechargement du backtest, métriques (Sharpe, Sortino, drawdown, Calmar…),
rendements cumulés, Sharpe glissant, heatmaps annuelles, analyse des poids, sensibilité aux coûts et
comparaison avec des stratégies de référence. Les paramètres sont dans la première cellule
(`CONFIG`, `RUN_BACKTEST`, `NUM_EPOCHS`, `FACTOR_MODELS`, `DEVICE`).

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/dlsa_walkthrough.ipynb
# ou ouvrir dans VS Code / Jupyter avec le kernel .venv
```

### Tests

```bash
.venv/bin/python -m pytest tests
```

| Fichier | Contenu |
|---|---|
| `tests/test_imports.py` | Les 13 modules s'importent ; `np.bool` et `plt.plot_date` sont disponibles. |
| `tests/test_models.py` | Forward + backward des 4 modèles, sur CPU et MPS. |
| `tests/test_preprocess.py` | Dimensions, exclusion des données manquantes, exactitude de la fenêtre cumsum. |
| `tests/test_residuals.py` | Les 21 fichiers de résidus : en-tête `.npy` valide, 9483 actifs, `float64`. |
| `tests/test_train_test.py` | Mini backtest glissant sur données synthétiques (OUFFN, RawFFN) ; rechargement d'un checkpoint avec `torch.load` récent. |

## 5. Résultats obtenus en local

Run complet de `configs/ouffn-full.yaml` (OUFFN, 100 époques, objectif Sharpe, sans coûts), CPU Apple M4,
avec `local/run_local.py -c configs/ouffn-full.yaml --results-tag full-cpu`. Durée : environ 36 min
(IPCA 5 ≈ 5 min, PCA 5 ≈ 5 min, Fama-French 5 ≈ 26 min, sur un univers plus large). Test hors échantillon
du 26/12/2001 au 30/12/2016 (3781 jours). Résultats dans `results/OUFFN/results_safe-tetra_full-cpu*.pickle`.

Le tableau compare le modèle à des stratégies de référence calculées dans le notebook
(section 9) sur les mêmes résidus, la même période et les mêmes features OU :

| Résidus | Stratégie | Sharpe | Rend. ann. (%) | Vol. ann. (%) | Max DD (%) | Turnover | Part short |
|---|---|---:|---:|---:|---:|---:|---:|
| IPCA 5 | **OUFFN** | **0,64** | 1,28 | 2,01 | −5,23 | 0,05 | 0,10 |
| | OU seuil ±1,25 | 0,80 | 3,18 | 4,00 | −8,00 | 0,95 | 0,51 |
| | OU linéaire | 0,59 | 1,58 | 2,68 | −5,67 | 0,63 | 0,51 |
| | Reversal 30 j | 0,69 | 3,77 | 5,43 | −10,16 | 0,31 | 0,52 |
| | Long 1/N | 0,49 | 0,98 | 2,01 | −5,23 | 0,05 | 0,00 |
| PCA 5 | **OUFFN** | **0,85** | 0,62 | 0,73 | −1,97 | 0,04 | 0,00 |
| | OU seuil ±1,25 | 1,10 | 2,84 | 2,59 | −8,40 | 0,94 | 0,51 |
| | OU linéaire | 0,87 | 1,42 | 1,64 | −3,13 | 0,63 | 0,51 |
| | Reversal 30 j | 0,67 | 2,02 | 3,02 | −10,14 | 0,31 | 0,51 |
| | Long 1/N | 0,84 | 0,61 | 0,73 | −1,97 | 0,04 | 0,00 |
| Fama-French 5 | **OUFFN** | **0,19** | 0,67 | 3,51 | −12,65 | 0,05 | 0,00 |
| | OU seuil ±1,25 | 0,44 | 1,54 | 3,53 | −10,76 | 0,94 | 0,51 |
| | OU linéaire | 0,32 | 0,75 | 2,32 | −6,85 | 0,63 | 0,51 |
| | Reversal 30 j | 0,49 | 2,30 | 4,70 | −11,47 | 0,31 | 0,51 |
| | Long 1/N | 0,19 | 0,67 | 3,50 | −12,65 | 0,05 | 0,00 |

**Lecture :**

- **OUFFN a convergé vers une solution dégénérée.** Sur PCA 5 et Fama-French 5, ses rendements ont une
  corrélation de 1,00 avec « Long 1/N » (0,92 sur IPCA 5) et il ne prend presque aucune position courte.
  Le réseau (65 paramètres, sigmoïdes, dropout 0,25) donne à peu près le même poids positif à tous les
  résidus au lieu d'exploiter le signal OU. Un essai à 1 époque donnait d'ailleurs déjà un Sharpe de 0,84
  sur PCA 5 : l'entraînement n'apporte presque rien ici.
- Les stratégies OU simples font mieux en Sharpe brut, mais avec un turnover 10 à 20 fois plus élevé.
  Avec des coûts de transaction réalistes, l'écart se réduit fortement (voir la section 8 du notebook).
- Ces chiffres **ne reproduisent pas** ceux du papier : une seule graine, pas de matrices de composition
  (`use_residual_weights: False`), et OUFFN n'est qu'un modèle de comparaison. Le modèle principal du papier
  est le CNN+Transformer (`configs/cnntransformer-full.yaml`), nettement plus coûteux : environ 146 ms par
  itération sur CPU contre 16 ms sur MPS pour 2000 actifs. Il faut donc le lancer avec `--device mps`.


## 6. Limites et points d'attention

- **Données brutes absentes** : impossible de régénérer les résidus (`run_factor_model.py`). On
  ne peut que réutiliser ceux fournis, de 1998 à 2016.
- **Matrices de composition absentes** : le turnover et la part courte ne sont pas comparables à
  ceux du papier ; il faut garder `use_residual_weights: False`.
- **Cas limite dans `test()`** : si `(T − length_training)` est un multiple exact de
  `retrain_freq`, la dernière sous-période est vide et le code plante. Ce n'est pas le cas avec
  les configs fournies (3781 = 30 × 125 + 31).
- **Coût de calcul** : OUFFN est minuscule et tourne vite sur CPU. Le CNN+Transformer sur
  `preprocess_cumsum` est bien plus lourd (fenêtres de 1000 × N × 30), avec environ 1 Go de
  mémoire par sous-période ; MPS peut alors aider.
- **Avertissements sans conséquence** : `plot_date` déprécié, `marker` redondant dans les
  graphiques.
- **Code d'origine non modifié** : tout l'outillage ajouté est dans
  `requirements-macos-arm64.txt`, `local/`, `tests/`, `notebooks/` et ce fichier. Les sorties
  (`results/`, `logs/`, `data/`) ne sont pas ignorées par le `.gitignore` d'origine, qui n'a pas été modifié.
