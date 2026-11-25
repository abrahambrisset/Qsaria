# Audit de la PR 5 pour la publication de Qsaria V1

La comparaison porte sur la V1 `61a0cc2fb49a5f9650c75a45f8dee5cfeedd936c` et la PR `14202ffba8948ae55e14a04bfbe11d3aa1494774`. Les deux révisions ont le même `uv.lock`. Les vérifications ont utilisé Python 3.12, l’environnement verrouillé et un stockage local. [PR 5](https://github.com/abrahambrisset/Qsaria/pull/5)

## Vérifications

| Vérification | V1 | PR 5 |
| --- | --- | --- |
| Black et Ruff | OK | OK |
| Suite unitaire sur export du code | 1 244 réussites, 6 échecs, 2 ignorés, 1 désélectionné | 1 524 réussites, 6 échecs, 2 ignorés, 2 désélectionnés |
| Rejeu ciblé des zones en échec, dans un clone local avec son historique et le chemin de stockage de CI | 7 réussites | 5 réussites, 2 échecs de signature Codex |

Le rejeu isole un défaut de provenance dans la PR. Son `compatibility.json` porte les signatures du contenu, mais le champ `source` décrit encore un arbre de travail sale rattaché à `38d70bf`. Les contrôles `test_bundle_validator_passes` et `test_compatibility_signatures_are_current` échouent sur le commit de la PR. Les journaux sont dans `.files/publication/logs/`.

Le même essai technique a ensuite curé 40 molécules dans chaque révision, puis entraîné LightGBM sur la même partition, avec 12 arbres, deux descripteurs pré-calculés et une optimisation désactivée. Les deux révisions produisent un résumé et des prédictions de test. Les scores sur cette cible calculée servent uniquement à vérifier l’exécution. Les artefacts sont sous `.files/publication/comparison/`.

## Évaluation des changements

| Changement | Intégration observée | Décision pour la V1 |
| --- | --- | --- |
| Métriques étendues, statistiques et incertitude | Les métriques sont intégrées à l’orchestration et au résumé d’entraînement. Les méthodes conformes et les statistiques sont fournies comme modules et exercices unitaires. | Calculer les métriques préspécifiées et leurs intervalles depuis les prédictions exportées. Une évaluation conforme demanderait un protocole de calibration dédié et une preuve d’intégration au workflow. |
| Audit des fuites | La PR fournit un module et des tests. Les risques d’identité sont à vérifier sur les partitions utilisées dans l’article. | Tester les collisions avec la même identité moléculaire que la V1. Conserver la distinction entre partition officielle et analyse de sensibilité. |
| Baselines scikit-learn | Backend auxiliaire avec tests. | Entraîner la référence naïve et Random Forest dans le benchmark avec les mêmes features et partitions que LightGBM. |
| Jeux et stockage de résultats | Le loader TDC passe par `single_pred.get_split()`. Le groupe ADMET publie pour les deux jeux pilotes un fichier de développement et un fichier de test. Le registre de résultats Parquet est une brique auxiliaire. | Charger les fichiers du groupe ADMET depuis l’archive officielle vérifiée. Fournir des CSV de prédictions et un manifeste par exécution. [API TDC BenchmarkGroup](https://github.com/mims-harvard/TDC/blob/main/tdc/benchmark_group/base_group.py) |
| Représentations et filtre de descripteurs | Les représentations sont raccordées au toolkit. Plusieurs tests de génération ont passé 45 secondes dans la fermeture d’un pool de processus sur le Mac de test. | Utiliser Morgan binaire, déjà présent dans la V1, pour mesurer les modèles sur des entrées comparables. |
| Données de référence techniques | La cible de régression est le cLogP calculé par RDKit. La cible de classification est un seuil de TPSA. | Utiliser ces données pour tester le flux et distinguer ces essais des résultats sur des mesures publiques. |

Les résultats et le manuscrit seront rattachés à la révision V1 effectivement évaluée. Les tests, usages auxiliaires et écarts de la PR sont consignés ici pour que chaque décision soit traçable.
