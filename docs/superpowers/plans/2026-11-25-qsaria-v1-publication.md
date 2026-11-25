# Plan d’exécution pour la publication de Qsaria V1

**Objectif** — Documenter et évaluer la V1 issue du stage, puis livrer les éléments permettant de reproduire les résultats de l’article.

**Architecture** — Exécuter le benchmark scientifique à partir des outils QSAR partagés. Évaluer l’orchestration séparément, à partir de demandes et de résultats attendus. Utiliser Python 3.12, `uv` et les contrats d’entraînement existants.

**Périmètre** — Préparer l’article sur Qsaria V1 dans `abrahambrisset/Qsaria`, avec son URL, ses étoiles et son historique. Citer ChemSpace Copilot comme projet d’origine. La feuille de route V2 appartient à l’autre stage. Le travail technique couvre la mesure, la reproductibilité et les correctifs V1 que les essais rendent nécessaires.

## Où intervenir

| Besoin | Emplacement |
| --- | --- |
| Corriger le comportement scientifique de la V1 | Cœur partagé dans `src/cs_copilot/tools/`, notamment `curation/`, `features/` et `prediction/`. Le même correctif bénéficie à l’équipe QSAR Chainlit et aux clients MCP. |
| Charger des données publiques, exécuter le protocole de l’article, entraîner les baselines et agréger les résultats | Scripts et configurations du benchmark dans le dépôt Qsaria. Les références comme Random Forest font partie du protocole de comparaison. |
| Corriger un écart de contrat, de transport ou d’orchestration propre aux clients MCP | `src/cs_copilot/mcp/qsaria/` et le bundle concerné dans `plugins/`, avec un test sur le parcours affecté. |
| Vérifier Chainlit | Tester l’équipe QSAR native via `chainlit_app.py` et `src/cs_copilot/agents/factories.py`. Un parcours représentatif permet de contrôler que les changements partagés sont utilisables sur cette interface. |

Le serveur MCP construit le toolkit partagé dans `src/cs_copilot/mcp/qsaria/toolkit_hub.py`. Le dépôt amont `Laboratoire-de-Chemoinformatique/chemspacecopilot` sert de référence pour la provenance et la citation. Qsaria est le dépôt de travail et de publication.

## 1. Établir la référence V1 et auditer la PR #5

- [ ] Prendre `61a0cc2` comme référence de départ et comparer la PR #5 à sa base. La tête disponible au cadrage est `14202ff`. Consigner les SHA complets effectivement testés.
- [ ] Installer les dépendances à partir de `uv.lock` dans des environnements isolés. Utiliser Python 3.12 et `uv sync --locked --extra mcp --extra prediction`.
- [ ] Exécuter les vérifications de `.github/workflows/ci.yml` sur les deux révisions. Lancer `uv run black --check src/ tests/`, `uv run ruff check src/ tests/`, puis `uv run pytest tests/unit/ -v --tb=short` avec `USE_S3=false` et `CS_COPILOT_STORAGE_ROOT=data`. Reprendre les commandes de validation des bundles dans les workflows correspondants selon les fichiers touchés.
- [ ] Faire un essai de curation, d’entraînement et de restitution sur une petite fixture. Comparer les appels effectifs et les sorties, en particulier pour les métriques, l’audit des fuites, les baselines et les opérations de benchmark. Consigner les écarts entre V1 et PR, les échecs reproductibles et les contraintes de l’environnement.

**Livrable** — Une revue de la PR, avec pour chaque changement une preuve d’intégration, un usage potentiel dans le benchmark ou un motif de report. Les cibles calculées par RDKit dans la PR servent aux essais techniques du pipeline.

## 2. Définir le protocole du papier et lancer un pilote V1

- [ ] Définir les questions que le papier testera. Mesurer la performance prédictive, la pertinence des diagnostics existants et la capacité de l’agent à produire un résultat traçable.
- [ ] Choisir un jeu de régression et un jeu de classification. Les premiers candidats sont [AqSolDB](https://tdcommons.ai/benchmark/admet_group/06aqsol/) et [BBB_Martins](https://tdcommons.ai/benchmark/admet_group/07bbb/). Pour chacun, consigner version, licence, empreinte des données, identifiants des partitions et métrique primaire.
- [ ] Préparer une commande rejouable à partir d’une configuration versionnée. Réutiliser les toolkits de curation, d’entraînement et d’évaluation de la V1. Pour les comparaisons directes, donner les mêmes partitions aux modèles Qsaria et aux baselines naïve et Random Forest. Consigner les budgets d’optimisation et les versions logicielles.
- [ ] Vérifier les doublons d’identité moléculaire et la présence des classes dans les partitions. Préserver les partitions officielles pour les comparaisons avec le benchmark d’origine. Si une comparaison après curation modifie la population ou les partitions, enregistrer cette expérience avec son propre protocole.
- [ ] Exécuter le pilote sur les données de développement. Mesurer le coût, récupérer les prédictions et les erreurs éventuelles, et vérifier que les métriques se recalculent à partir des prédictions sauvegardées. Réserver les labels du test final au calcul des résultats du protocole figé.

**Livrable** — Une configuration pilote, un journal de coûts et des résultats de développement, avec une liste motivée des défauts V1 qui affecteraient les conclusions.

## 3. Corriger les défauts V1 confirmés

- [ ] Reproduire le risque de doublons après standardisation dans le parcours curation → descripteurs → partition. Identifier la règle d’identité réellement appliquée et vérifier son effet par partition.
- [ ] Reproduire l’exclusion de LightGBM et TabICL du benchmark de classification. Vérifier les capacités des backends installés avant de retenir le correctif approprié.
- [ ] Vérifier les autres remarques des audits sur les chemins effectivement évalués. Examiner le traitement des sels, la présence des classes après découpage, les ensembles à empreintes de comptage et le statut `workflow_demo` lorsque la validation repose sur un seul découpage aléatoire.
- [ ] Pour chaque défaut retenu, écrire un test qui le reproduit et vérifier le correctif au point d’entrée partagé. Réutiliser notamment `test_curation_backend_framework.py`, `test_qsar_training_toolkit.py`, `test_qsar_validation_strategy.py`, `test_benchmark_toolkit.py` et `test_ensemble_backend.py` sous `tests/unit/`. Vérifier les tests MCP et les bundles lorsque leurs contrats sont touchés.
- [ ] Consigner les correctifs retenus et les reproduire sur le pilote. Conserver les résultats de V1 avant correction pour expliquer les écarts qui affectent l’article.

**Livrable** — Une révision V1 candidate, des tests de régression ciblés et un journal des changements qui influent sur les résultats.

## 4. Figer le protocole et exécuter les évaluations finales

- [ ] Figer les jeux, partitions, règles de curation, modèles, représentations, budgets, graines, métriques et comparaisons à partir du pilote. Enregistrer le commit, l’empreinte de `uv.lock` et celle de la configuration avant l’évaluation finale.
- [ ] Exécuter le benchmark scientifique avec les ressources mesurées au pilote. Enregistrer les prédictions, métriques, durées, échecs et artefacts par configuration. Rapporter les différences face aux baselines avec des intervalles d’incertitude dont l’unité et la méthode sont explicites.
- [ ] Analyser un cas représentatif des diagnostics de V1, selon les fonctionnalités et données disponibles. Par exemple, comparer les erreurs dans et hors du domaine d’applicabilité ou étudier les falaises d’activité sur une version corrigée et identifiée de [MoleculeACE](https://github.com/molML/MoleculeACE).
- [ ] Exécuter des demandes QSAR et leurs reformulations via le client MCP retenu pour le papier. Consigner client, modèle, paramètres et résultats. Vérifier le routage, l’exécution des étapes attendues, la présence des artefacts et l’accord entre les chiffres du rapport et les sorties des outils. Faire un essai de bout en bout de l’équipe QSAR de Chainlit et documenter la couverture réelle de chaque interface citée.

**Livrable** — Deux ensembles de résultats, l’un pour la qualité prédictive et l’autre pour la fiabilité des workflows, accompagnés d’un manifeste et de sorties vérifiables.

## 5. Préparer le paquet de soumission

- [ ] Régénérer les tableaux et figures à partir des sorties enregistrées. Vérifier la commande de reproduction dans un environnement propre et contrôler la traçabilité de chaque chiffre utilisé dans l’article.
- [ ] Rédiger les sections méthodes, résultats, limites et disponibilité du code et des données. Indiquer la révision évaluée et créditer ChemSpace Copilot. Décrire les dimensions effectivement évaluées de Qsaria V1.
- [ ] Préparer une version citable dans le dépôt existant, le guide de reproduction et le manuscrit pour relecture avec le tuteur.

**Livrable** — Le manuscrit à relire, la version citable et le guide de reproduction.

**Critère de fin** — Le manuscrit et ses résultats pointent vers la même révision V1. Les entrées, l’environnement, les scripts et les artefacts nécessaires permettent de reconstruire les chiffres, tableaux et figures utilisés pour la soumission.

## Sources de travail

- [PR #5 et plan Claude](https://github.com/abrahambrisset/Qsaria/pull/5), pour les modifications proposées et leur contexte.
- Audit Gemini Carbon transmis avec le dossier de travail, pour les anomalies de curation, de classification, d’ensemble et de découpage à reproduire.
- [ChemSpace Copilot](https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot), pour la provenance du projet.
