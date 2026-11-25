# Benchmark de Qsaria V1

Le benchmark compare le backend LightGBM de Qsaria à une référence naïve et à Random Forest sur AqSolDB et BBB_Martins. Il utilise la standardisation de Qsaria et une empreinte Morgan binaire de rayon 2 et de 2 048 bits pour les modèles qui consomment des descripteurs. Le test final suit les partitions du groupe TDC ADMET. [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/)

## Données

Le manifeste verrouille l’archive ADMET publiée dans le dépôt TDC de Harvard Dataverse, fichier `4426004`. Pour obtenir les mêmes octets :

```bash
mkdir -p .files/publication/datasets
curl --fail --location https://dataverse.harvard.edu/api/access/datafile/4426004 \
  --output .files/publication/datasets/tdc-admet-4426004.zip
```

Le script vérifie le SHA-256 défini dans `benchmarks/qsaria-v1.toml`. L’archive et les prédictions par molécule restent dans le répertoire d’artefacts local. Le dépôt Dataverse de TDC est référencé sous CC0 par le catalogue NLM. Pour réutiliser les données sources séparément, consulter aussi les conditions du jeu d’origine. [Dépôt TDC](https://doi.org/10.7910/DVN/21LKWG) · [Droits TDC dans le catalogue NLM](https://datasetcatalog.nlm.nih.gov/dataset?q=0002132185) · [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1) · [BBB_Martins](https://tdcommons.ai/benchmark/admet_group/07bbb/)

## Reproduction

Depuis le dépôt Qsaria, avec Python 3.12 et l’environnement verrouillé :

```bash
uv sync --locked --python 3.12 --extra prediction --extra mcp
export USE_S3=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
uv run --no-sync python scripts/benchmark_qsaria_v1.py \
  --config benchmarks/qsaria-v1.toml \
  --archive .files/publication/datasets/tdc-admet-4426004.zip \
  --output .files/publication/reproduction-development
```

Sur macOS, LightGBM doit pouvoir charger `libomp`. Dans l’environnement testé, PyTorch fournit la bibliothèque, accessible avec `DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib"`.

La commande ci-dessus lit seulement `train_val.csv`. Pour rejouer le test à la révision exacte de l’article, lancer ces commandes depuis la copie qui contient le paquet de soumission. Le worktree et le répertoire de sortie doivent être neufs.

```bash
qsaria_root="$PWD"
git worktree add --detach .files/publication/replay-v1 \
  1179b63d42eea5be044bdc041a0f12c35983019c
(
  cd .files/publication/replay-v1 || exit
  uv sync --locked --python 3.12 --extra prediction --extra mcp || exit
  if [ "$(uname -s)" = Darwin ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  fi
  uv run --no-sync python scripts/benchmark_qsaria_v1.py \
    --phase test --config benchmarks/qsaria-v1.toml \
    --archive "$qsaria_root/.files/publication/datasets/tdc-admet-4426004.zip" \
    --output "$qsaria_root/.files/publication/reproduction-test"
)
```

Avec les scripts de restitution du paquet de soumission, régénérer les tables et la figure à partir des prédictions du test :

```bash
uv run --no-sync python scripts/report_qsaria_v1_benchmark.py \
  --run .files/publication/reproduction-test \
  --output .files/publication/reproduction-report
```

Le script de restitution recalcule les scores et les écarts ponctuels depuis les prédictions. Le tableau et la figure utilisent aussi les bornes bootstrap enregistrées par le benchmark. [Manuscrit de relecture](qsaria-v1-manuscript.md) · [Résultats agrégés](results/qsaria-v1/scores.csv)

## Lecture des résultats

- La validation de développement maintient ensemble les lignes de même identité standardisée. Le mode test réutilise les partitions publiées et calcule aussi les scores sur les identités absentes de l’entraînement lorsqu’il existe un chevauchement.
- Les lignes dont la structure ne produit pas d’empreinte sont consignées. Les métriques portent sur les lignes featurisables. Les sorties conservent le numéro de ligne dans l’archive pour relier les scores aux partitions d’origine.
- Les budgets sont définis avant le test. LightGBM et Random Forest utilisent chacun 80 arbres par graine, sur un cœur, avec leurs autres réglages fixes. Les performances des modèles sont rapportées par graine. L’intervalle à 95 % estime l’écart LightGBM moins Random Forest pour la première graine par bootstrap apparié des identités chimiques. Un écart négatif favorise LightGBM pour la MAE et un écart positif pour l’AUROC.
- Le domaine d’applicabilité est la boîte à bornes strictes de Qsaria ajustée aux seules empreintes d’entraînement. Rapporter les effectifs avec les scores de chaque classe de domaine. Une AUROC est calculable si les deux classes sont présentes.
- `manifest.json` donne le commit, les empreintes de l’archive, de `uv.lock` et de la configuration, ainsi que les versions des bibliothèques. Les fichiers `predictions.csv` et `summary.json` fournissent l’évidence ligne par ligne et les métriques recalculées correspondantes.

## Comparaison AutoML à budget égal

Rejouer le comparatif depuis sa révision archivée, avec l’archive préparée dans le dépôt principal. Utiliser un worktree et des répertoires de sortie neufs. Les deux méthodes disposent de quatre candidats, de 80 arbres par ajustement et d’un cœur. Les durées mesurées sont dans les rapports, avec les trois refits de chaque gagnant sur le test officiel.

```bash
qsaria_publication="$PWD/.files/publication"
git worktree add --detach "$qsaria_publication/replay-automl-v1" \
  89d02af3760d495a280b924f254be0c3e9721239 && (
  set -e
  cd "$qsaria_publication/replay-automl-v1"
  uv sync --locked --python 3.12 --extra prediction --extra mcp
  export USE_S3=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
  if [ "$(uname -s)" = Darwin ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  fi
  qsaria_archive="$qsaria_publication/datasets/tdc-admet-4426004.zip"
  qsaria_development="$qsaria_publication/reproduction-automl-development"
  uv run --no-sync python scripts/benchmark_qsaria_v1_automl.py \
    --config benchmarks/qsaria-v1-automl.toml --archive "$qsaria_archive" \
    select --output "$qsaria_development"
  qsaria_selection_hash=444830fe103819888c6a95ce4c8153bec5d67b6c30256028868166ddd7c2d21a
  uv run --no-sync python scripts/benchmark_qsaria_v1_automl.py \
    --config benchmarks/qsaria-v1-automl.toml --archive "$qsaria_archive" \
    evaluate --selection "$qsaria_development/selection.json" \
    --selection-sha256 "$qsaria_selection_hash" \
    --output "$qsaria_publication/reproduction-automl-test"
)
```

L’évaluation vérifie les empreintes de la sélection, de l’archive et des fichiers du protocole avant d’ouvrir les partitions de test. Le rapport distingue les identités exactes des voisins non identiques à Tanimoto ≥ 0,95 et ≥ 0,99. Le test reprend les partitions publiques du benchmark à réglages fixes. [Rapport développement](results/qsaria-v1/automl-development.json) · [Rapport test](results/qsaria-v1/automl-evaluation.json)

## Vérification des parcours

La [fixture synthétique](fixtures/qsaria-v1-synthetic.csv) contient les 40 premières structures de `tests/data/reference_regression.csv` de la PR 5 avec leur cLogP calculé. Elle vérifie les parcours des outils. Les mesures prédictives du papier utilisent les jeux du groupe ADMET. [PR 5](https://github.com/abrahambrisset/Qsaria/pull/5)

Les deux essais d’entraînement dans Terra ont utilisé le pont de validation, qui lance le serveur MCP en stdio et journalise ses réponses dans `.files/publication/agentic/tool_calls/`. Un troisième chat Terra a confirmé les appels natifs `qsaria_bootstrap` et `qsaria_list_experiments` en lecture seule dans Codex Desktop, depuis la racine du dépôt avec le plugin installé.

```bash
uv run --no-sync python scripts/call_qsaria_v1_mcp.py qsaria_bootstrap \
  --select status,profile,llm_policy

USE_S3=false CS_COPILOT_STORAGE_ROOT="$PWD/.files/publication/chainlit-check-data" \
  uv run --no-sync python -m scripts.check_qsaria_v1_chainlit
```

La seconde commande vérifie la délégation réelle dans l’équipe construite par Chainlit avec un transport de modèle simulé en local. Elle produit des prédictions LightGBM et vérifie le résultat renvoyé par le rapport.

[Essais Terra](results/qsaria-v1/agentic.json) · [Essai Chainlit](results/qsaria-v1/chainlit.json) · [Empreintes du rejeu](results/qsaria-v1/reproduction.json)
