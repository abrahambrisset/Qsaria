# Suivi de la publication de Qsaria V1

Plan de référence : [Publication de Qsaria V1](2026-11-25-qsaria-v1-publication.md).

## État des étapes

- [ ] Établir la référence V1 et auditer la PR #5.
- [ ] Définir le protocole et exécuter le pilote V1.
- [ ] Corriger les défauts V1 confirmés.
- [ ] Figer le protocole et exécuter les évaluations finales.
- [ ] Préparer le paquet de soumission.

## 25 novembre 2026

- Référence locale V1 : `61a0cc2fb49a5f9650c75a45f8dee5cfeedd936c`. Référence de la PR #5 : `14202ffba8948ae55e14a04bfbe11d3aa1494774`. La copie d’audit de la PR se trouve dans `.files/publication/pr5/`.
- Les environnements Python, caches et journaux de vérification sont regroupés dans `.files/publication/`, un dossier ignoré par Git. Les commits locaux seront contrôlés fichier par fichier.
- L’installation verrouillée en sandbox a échoué lors de la résolution de `github.com` pour `deepchemography`. Le téléchargement a été relancé avec un accès réseau autorisé et les mêmes destinations locales.
- La lecture du code confirme trois sources d’optimisation implicite : le contrat d’entraînement, le défaut du moteur interne et la construction des candidats du benchmark. Le correctif V1 rendra l’optimisation explicite tout en conservant les configurations de recherche demandées.
- Le commit local `1c8b2fa` contient le plan et ce journal. L’environnement verrouillé Python 3.12 est installé. Sur ce Mac, LightGBM charge `libomp` via `DYLD_FALLBACK_LIBRARY_PATH` pointant vers la bibliothèque fournie avec PyTorch dans l’environnement du projet. Les tests bornent les threads BLAS et OpenMP à un.
- Les régressions reproduisent l’optimisation implicite avant correction. Après correction, les 87 tests ciblés des contrats QSAR, de l’optimisation et du benchmark passent. Les contrôles Black et Ruff et les signatures des trois bundles passent. L’opt-in d’entraînement est `tuning.enabled=true`, celui du benchmark est `tuning_enabled=true`.
- Le test d’installation isolée du plugin Codex passe. Une session MCP locale sur stdio initialise le profil `qsaria` et appelle `qsaria_bootstrap` avec succès, avec 53 outils exposés. Le chat Codex de vérification a été lancé avec GPT‑6.1 Terra et réflexion élevée. Il signale que le MCP n’est pas connecté à ce chat. La connexion au client Codex reste à vérifier.
- L’archive officielle TDC ADMET, identifiant Dataverse `4426004`, a été mise en cache localement. SHA-256 `bd0005246cb6f1672333c28ad202112a8bf071ab65d847848715030308faf1f1`. Elle fournit `train_val.csv` et `test.csv` pour AqSolDB et BBB_Martins. Le loader TDC de la PR #5 appelle `single_pred.get_split()`. L’audit doit distinguer ce découpage généré du test du groupe de benchmark publié.
