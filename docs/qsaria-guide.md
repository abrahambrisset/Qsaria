# QSARIA Guide

Ce document resume comment utiliser QSARIA au quotidien: architecture,
lancement avec les differents containers, configuration des modeles LLM,
workflows QSAR disponibles, exemples de prompts, fichiers generes et
depannage.

Nom du projet: le depot GitHub actuel est `QSARIA`. Une partie du code et des
anciens fichiers garde encore le nom historique `ChemSpace Copilot` ou
`cs_copilot`.

## 1. Vue d'ensemble

QSARIA est une application Chainlit qui expose un systeme multi-agents pour la
chemoinformatique et les workflows QSAR.

Le sous-systeme QSAR est isole du reste de l'application. Il gere:

- curation de datasets QSAR;
- entrainement de modeles Chemprop, LightGBM et TabICL;
- benchmark explicite;
- registre et catalogue de modeles;
- inference avec modeles persistants;
- ensembles de consensus;
- rapports QSAR standardises;
- domaine d'applicabilite;
- activity cliffs avec SALI;
- export LaTeX via `@latex`.

Le point d'entree utilisateur est le chat Chainlit.

## 2. Architecture fonctionnelle

Architecture simplifiee:

```text
Chainlit UI
  |
  |-- routeur principal
      |
      |-- equipe generale
      |     |-- ChEMBL downloader
      |     |-- GTM
      |     |-- chemoinformatician
      |     |-- report generator
      |     |-- autoencoder
      |     |-- peptide WAE
      |     |-- retrosynthesis
      |
      |-- equipe QSAR
            |-- dataset_curation_agent
            |-- qsar_training_agent
            |-- model_registry_agent
            |-- model_inference_agent
            |-- qsar_report_agent
```

Architecture QSAR:

```text
Agents QSAR
  |
  |-- DatasetCurationToolkit
  |-- QSARTrainingToolkit
  |     |-- MolecularFeatureToolkit (interne)
  |     |-- ActivityCliffToolkit (interne)
  |     |-- ChempropToolkit -> ChempropBackend (interne)
  |     |-- LightGBMToolkit -> LightGBMBackend (interne)
  |     |-- TabICLToolkit   -> TabICLBackend (interne)
  |-- ModelRegistryToolkit
  |-- PredictionInferenceToolkit
  |-- BenchmarkToolkit
  |-- EnsembleToolkit
  |-- qsar_report_agent
```

Regle importante: les agents ne doivent pas appeler directement les moteurs
backend ni les outils de features internes. Ils passent par les façades
publiques, surtout `QSARTrainingToolkit` pour l'entraînement.
pour l'entrainement.

## 3. Configuration modele LLM

La configuration se fait par variables d'environnement ou par `.modelconf`.

Priorite:

```text
variables d'environnement > .modelconf > valeurs par defaut
```

Fournisseurs supportes:

- `deepseek`
- `openrouter`
- `ollama`

Variables principales:

```bash
MODEL_PROVIDER=deepseek|openrouter|ollama
MODEL_ID=...
MODEL_MAX_TOKENS=8192
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
OLLAMA_HOST=http://localhost:11434
CS_COPILOT_AGENT_TEAM=qsar
```

Exemple OpenRouter:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_ID="deepseek/deepseek-v4-flash"
export OPENROUTER_API_KEY="..."
export MODEL_MAX_TOKENS=8192
```

Exemple DeepSeek direct:

```bash
export MODEL_PROVIDER=deepseek
export MODEL_ID="deepseek-chat"
export DEEPSEEK_API_KEY="..."
```

Exemple Ollama local:

```bash
export MODEL_PROVIDER=ollama
export MODEL_ID="qwen2.5:72b"
export OLLAMA_HOST="http://localhost:11434"
```

## 4. Lancer QSARIA

### 4.1 Docker Compose

Cas standard Linux, macOS ou Windows avec Docker Desktop:

```bash
docker compose build chainlit-app
./docker-start.sh
```

Le script choisit un port libre entre `8000` et `8010`.

Acces:

```text
http://localhost:8000
```

Mode production direct:

```bash
docker compose up -d
docker compose logs -f chainlit-app
```

Mode developpement avec hot reload:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Arreter:

```bash
docker compose down
```

Reset complet Docker:

```bash
docker compose down -v --remove-orphans
docker compose build chainlit-app
```

Note OpenRouter avec Docker: `docker-start.sh` est historiquement oriente
DeepSeek pour la saisie interactive de cle. Pour OpenRouter, preferer un
fichier `.env` ou des variables exportees avant `docker compose up`.

Exemple:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_ID="deepseek/deepseek-v4-flash"
export OPENROUTER_API_KEY="..."
export CS_COPILOT_AGENT_TEAM=qsar
export MODEL_MAX_TOKENS=4096
docker compose up -d
```

### 4.2 Apple Container

Prerequis: Apple Container CLI installe.

Lancement avec build:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_MAX_TOKENS=4096
export MODEL_ID="deepseek/deepseek-v4-flash"
export OPENROUTER_API_KEY="..."
scripts/apple-container.sh
```

Relancer sans rebuild si l'image existe deja:

```bash
export APPLE_CONTAINER_SKIP_BUILD=1
scripts/apple-container.sh
```

Regler CPU/RAM:

```bash
export APPLE_CONTAINER_CPUS=8
export APPLE_CONTAINER_MEMORY=12G
scripts/apple-container.sh
```

Logs:

```bash
container logs cs-copilot-apple
```

Stop:

```bash
container stop cs-copilot-apple
```

Apple Container monte plusieurs dossiers du repo dans `/app`, notamment:

- `src`
- `public`
- `examples`
- `.files`
- `data`
- `models`
- `chainlit_app.py`

Cela permet de tester beaucoup de changements Python sans reconstruire l'image.
Un rebuild reste necessaire si les dependances ou le Dockerfile changent.

### 4.3 Apptainer

Cas recommande pour machine Linux GPU.

Construire l'image:

```bash
apptainer build chemspacecopilot.sif scripts/chemspacecopilot.def
```

Lancer avec DeepSeek:

```bash
export DEEPSEEK_API_KEY="..."
scripts/run_deepseek.sh
```

Lancer avec OpenRouter:

```bash
export OPENROUTER_API_KEY="..."
export MODEL_ID="deepseek/deepseek-v4-flash"
export MODEL_MAX_TOKENS=4096
scripts/run_openrouter.sh
```

Lancer manuellement:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_ID="deepseek/deepseek-v4-flash"
export OPENROUTER_API_KEY="..."
export MODEL_MAX_TOKENS=4096
scripts/run_apptainer.sh
```

Le script `run_apptainer.sh` monte le repo dans `/app` et lance:

```text
chainlit run chainlit_app.py --host 0.0.0.0 --port 8000
```

Il utilise `apptainer exec --nv`, donc le GPU NVIDIA est expose si disponible.

## 5. Stockage et fichiers generes

Par defaut, les artefacts locaux sont ecrits dans:

```text
.files/
```

Les modeles persistants sont ecrits dans:

```text
data/model_assets/internal/
```

Les checkpoints externes peuvent etre dans:

```text
data/model_assets/checkpoints/
```

Les outputs typiques d'un entrainement QSAR:

- dataset cure;
- rapport de curation JSON;
- fichiers de features tabulaires si LightGBM ou TabICL;
- summary JSON d'entrainement;
- predictions de test;
- plots;
- domaine d'applicabilite;
- activity cliffs;
- bundle zip complet;
- modeles persistants dans le catalogue.

## 6. Raccourcis de routage

Raccourcis supportes en debut de message:

```text
@qsaria
@qsar
@latex
```

Exemples:

```text
@qsaria Entraine un modele LightGBM standard_qsar pour pEC50.
```

```text
@latex
```

`@qsaria` force le routage vers l'equipe QSAR. Il ne cree pas encore de menu
deroulant dans l'interface; c'est un token texte reconnu par le backend.

`@latex` exporte le dernier rapport QSAR compatible en version LaTeX quand un
etat de prediction/rapport existe.

## 7. Workflows QSAR disponibles

### 7.1 Curation

Objectif: transformer un CSV brut en dataset QSAR pret pour entrainement.

La curation gere:

- identification des colonnes SMILES et cible;
- standardisation ChEMBL;
- extraction de parent structure;
- diagnostics ChEMBL checker;
- suppression des organometalliques;
- suppression ou aggregation de doublons selon l'identite QSAR;
- conversion numerique de la cible;
- detection d'outliers;
- rapport de curation.

Prompt exemple:

```text
@qsaria Cure ce dataset QSAR. La colonne SMILES est SMILES et la cible est pEC50.
```

### 7.2 Entrainement standard

`standard_qsar` est l'unique protocole nomme. Il applique un split random
80/10/10. Chemprop utilise son graphe moleculaire, LightGBM utilise `rdkit_all`
avec 50 essais Optuna/TPE par defaut, et TabICL utilise `rdkit_all`. L'analyse
post-selection des outliers est executee lorsqu'elle est applicable.

Prompts:

```text
@qsaria Entraine un modele Chemprop pour pEC50.
```

```text
@qsaria Entraine un modele QSAR LightGBM standard_qsar pour predire pEC50.
```

```text
@qsaria Entraine un modele TabICL standard_qsar pour pEC50.
```

Pour une demande generale d'entrainement, QSARIA doit utiliser
`standard_qsar`. Il ne doit pas inventer de repeated holdout, scaffold, cluster
ou cross-validation sans demande explicite.

### 7.3 Holdout personnalise

Utiliser quand l'utilisateur demande explicitement un split et un ratio.

Prompts:

```text
@qsaria Entraine un modele LightGBM pour pEC50 avec RDKit all. Utilise un split random 60/20/20.
```

```text
@qsaria Entraine un modele LightGBM pour pEC50 avec Morgan count. Utilise un split scaffold 60/20/20.
```

```text
@qsaria Entraine un modele Chemprop pour pEC50 avec un split scaffold 70/15/15.
```

### 7.4 Repeated holdout

Utiliser quand l'utilisateur demande explicitement plusieurs repetitions.

Prompts:

```text
@qsaria Entraine un modele LightGBM pour pEC50 avec les fingerprints Morgan binaires. Utilise une validation repeated random holdout avec 3 repetitions en 70/15/15.
```

```text
@qsaria Entraine un modele Chemprop pour pEC50 avec repeated scaffold holdout, 3 repetitions, 70/15/15.
```

Comportement attendu:

- une seed differente par repetition sauf demande contraire;
- un modele persistable par repetition;
- metriques individuelles et agregation moyenne plus ecart-type;
- pire repetition prise en compte pour la gouvernance.

### 7.5 Benchmark

Le benchmark est explicite uniquement. QSARIA ne doit pas lancer un benchmark
si l'utilisateur demande seulement un entrainement standard.

Prompt:

```text
@qsaria Lance un benchmark QSAR pour pEC50.
```

Benchmark standard:

- Chemprop graph;
- LightGBM sur `morgan_only`, `rdkit_all`, `morgan_count_only`;
- TabICL sur `morgan_only`, `rdkit_all`, `morgan_count_only`;
- protocole `standard_qsar` en l'absence de strategie explicite.

Une strategie avancee se demande explicitement et s'applique a tous les candidats:

```text
@qsaria Lance un benchmark QSAR pour pEC50 avec un repeated random holdout de 3 repetitions.
```

### 7.6 Inference

Predire avec un modele catalogue:

```text
@qsaria Liste les modeles QSAR catalogues pour pEC50.
```

```text
@qsaria Predis pEC50 pour ce fichier avec le meilleur modele catalogue disponible.
```

Predire avec un modele specifique:

```text
@qsaria Predis pEC50 avec le modele pxr_challenge_train_...
```

### 7.7 Ensembles

Creer un ensemble de consensus depuis les modeles catalogues:

```text
@qsaria Cree un ensemble QSAR pour pEC50 a partir des meilleurs modeles catalogues.
```

Evaluer explicitement un ensemble:

```text
@qsaria Evalue cet ensemble QSAR sur ce dataset externe.
```

Sans demande d'evaluation, QSARIA doit seulement creer/resumer l'ensemble.

## 8. Representations tabulaires

Representations automatiques modernes:

- `morgan_only`
- `rdkit_all`
- `morgan_count_only`

Representations avancees explicites:

- `morgan_binary_count_rdkit_all`
- `morgan_rdkit_all`

Prompts:

```text
@qsaria Entraine un modele LightGBM pour pEC50 avec les fingerprints Morgan count.
```

```text
@qsaria Entraine un modele TabICL pour pEC50 avec RDKit all.
```

```text
@qsaria Entraine explicitement LightGBM avec morgan_binary_count_rdkit_all pour pEC50.
```

## 9. Backends QSAR

### Chemprop

- entree: CSV avec SMILES et cible;
- representation: graphe moleculaire;
- bon pour: modele graph neural network;
- pas de features tabulaires;
- utilise un CSV Chemprop minimal et un `splits_file` natif;
- dans QSARIA, `num_replicates=1` par split; la robustesse vient des splits,
  pas des replicats Chemprop.

### LightGBM

- entree: dataset tabulaire avec features moleculaires;
- representations: Morgan, Morgan count, RDKit all;
- bon pour: baseline rapide et robuste;
- utilise `n_jobs` selon le profil compute;
- force CPU par defaut dans ce projet pour eviter le temps perdu sur GPU
  LightGBM quand OpenCL n'est pas disponible.

### TabICL

- entree: dataset tabulaire;
- utilise le checkpoint TabICL configure;
- plus sensible a la RAM que LightGBM;
- a tester avec prudence sur gros jeux de donnees et representations larges.

### Ensemble

- backend de prediction seulement;
- combine des modeles catalogues;
- fournit une incertitude simple via desaccord entre composants.

## 10. Gouvernance des modeles

Statuts canoniques:

- `experimental`
- `workflow_demo`
- `validated`
- `robust_validated`

Gates principales:

- Dataset Gate;
- Hardest Split Gate;
- Robustness Gap Gate;
- Random Stability Gate.

Regle pratique:

- si le modele est utile mais ne passe pas les seuils forts, il est persiste en
  `workflow_demo`;
- `validated` exige des gates passees;
- `robust_validated` exige assez de repetitions/splits et des gates passees;
- un modele existant meilleur dans le catalogue n'empeche pas de persister un
  nouveau run comme `workflow_demo`.

## 11. Ce qui n'est pas actif actuellement

Cross-validation et nested cross-validation ne sont pas le chemin actif. Elles
ont ete retirees du workflow courant en attendant une implementation propre
basee sur scikit-learn.

Le vocabulaire de validation actuellement supporte:

- `holdout`
- `repeated_holdout`
- `split_family`: `random`, `scaffold`, `cluster`
- `split_sizes`: `[train, validation, test]`
- `n_repeats`
- `selection_metric`

## 12. Exemples de prompts utiles

### Inspection

```text
@qsaria Decris les backends QSAR disponibles.
```

```text
@qsaria Liste les modeles QSAR catalogues pour pEC50.
```

### Curation

```text
@qsaria Cure le dataset uploade pour une tache de regression QSAR. La cible est pEC50.
```

### Training simple

```text
@qsaria Entraine un modele Chemprop pour pEC50.
```

```text
@qsaria Entraine un modele LightGBM standard_qsar pour pEC50.
```

```text
@qsaria Entraine un modele TabICL standard_qsar pour pEC50.
```

### Training personnalise

```text
@qsaria Entraine un modele LightGBM pour pEC50 avec RDKit all. Utilise un split random 60/20/20.
```

```text
@qsaria Entraine un modele Chemprop pour pEC50 avec un split scaffold 60/20/20.
```

```text
@qsaria Entraine un modele LightGBM pour pEC50 avec Morgan count. Utilise repeated random holdout avec 3 repetitions en 70/15/15.
```

### Benchmark

```text
@qsaria Lance un benchmark QSAR pour pEC50.
```

```text
@qsaria Lance un benchmark QSAR pour pEC50 avec un repeated random holdout de 3 repetitions.
```

### Prediction

```text
@qsaria Predis pEC50 pour les molecules de ce fichier avec le meilleur modele catalogue.
```

```text
@qsaria Predis pEC50 avec le modele <model_id>.
```

### Ensemble

```text
@qsaria Cree un ensemble QSAR pour pEC50 avec les meilleurs modeles catalogues.
```

```text
@qsaria Evalue cet ensemble QSAR sur ce dataset externe.
```

### Export

```text
@latex
```

## 13. Tests rapides apres modification

Checks Python:

```bash
python -m py_compile \
  src/cs_copilot/agents/prompts.py \
  src/cs_copilot/tools/prediction/qsar_training_toolkit.py \
  src/cs_copilot/tools/prediction/chemprop_toolkit.py \
  src/cs_copilot/tools/prediction/chemprop_backend.py
```

Tests unitaires QSAR cibles:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv run pytest \
  tests/unit/test_prediction_backend.py \
  tests/unit/test_qsar_training_toolkit.py \
  tests/unit/test_qsar_validation_strategy.py \
  tests/unit/test_chemprop_adapter.py \
  -q
```

Verification Git:

```bash
git diff --check
git status --short
```

## 14. Depannage

### Le rapport n'arrive pas apres un training long

Ca peut venir du provider LLM ou du refresh du serveur.

Verifier:

```bash
docker compose logs -f chainlit-app
container logs cs-copilot-apple
```

En mode dev, le hot reload peut redemarrer l'app pendant un training.
Pour tester un training long, preferer le mode production.

### OpenRouter renvoie "input too long"

Le modele choisi via OpenRouter a une limite de contexte inferieure au contexte
envoye par l'app. Changer de modele ou reduire le contexte genere.

Variables utiles:

```bash
export MODEL_PROVIDER=openrouter
export MODEL_ID="..."
export MODEL_MAX_TOKENS=8192
```

`MODEL_MAX_TOKENS` limite la sortie, pas toujours l'entree. La limite d'entree
depend du modele et du provider OpenRouter effectif.

### Chemprop recoit un mauvais argument CLI

Verifier que les logs ne contiennent pas d'argument QSAR interne tel que:

```text
--validation-strategy
```

Les strategies QSAR doivent etre converties en `chemprop_splits.json`, puis
passees a Chemprop via:

```text
--splits-file <path>
```

### LightGBM semble n'utiliser qu'un CPU

Verifier le profil compute dans le rapport et les `n_jobs` effectifs.

Politique actuelle:

```text
local_light      n_jobs = min(cpu_count, 8)
local_standard   n_jobs = min(cpu_count, 24)
heavy_validation n_jobs = min(cpu_count, 48)
```

### TabICL consomme trop de RAM

Utiliser d'abord une representation simple:

```text
rdkit_all
morgan_only
morgan_count_only
```

Eviter les representations combinees larges sur machines a RAM limitee.

### Le bundle ne contient pas les bons fichiers

Verifier que le rapport liste un bundle global d'entrainement et pas seulement
un bundle de curation. Un bundle de training doit contenir au minimum:

- dataset cure;
- rapport de curation;
- summary training;
- checkpoint modele;
- predictions test;
- splits;
- metadata;
- AD;
- activity cliffs si disponibles.

## 15. Commandes Git usuelles

Etat:

```bash
git status --short --branch
```

Commit:

```bash
git add <files>
git commit -m "Message"
```

Push branche courante:

```bash
git push
```

Si SSH a besoin de la cle explicite:

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_github' git push -u origin validation-strategies
```
