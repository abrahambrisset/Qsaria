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
