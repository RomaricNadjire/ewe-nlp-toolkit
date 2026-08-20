# NOTICE — Attributions & licences

Ce dépôt agrège du **code original** (licence MIT), des **modèles** dérivés de
modèles de base tiers, et s'appuie sur des **jeux de données** aux licences
variées. Ce document récapitule les attributions requises.

## Modèles de base (Meta AI)

| Modèle de base                                                                                | Utilisé pour | Licence          |
| --------------------------------------------------------------------------------------------- | ------------ | ---------------- |
| [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all)                           | ASR éwé      | **CC-BY-NC-4.0** |
| [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) | Traduction   | **CC-BY-NC-4.0** |
| [`facebook/mms-tts-{ewe,eng,fra}`](https://huggingface.co/facebook/mms-tts-ewe)               | TTS (démo)   | **CC-BY-NC-4.0** |

➡️ Les modèles fine-tunés publiés dans ce projet **héritent de CC-BY-NC-4.0**
(usage **non commercial**, attribution de Meta requise).

## Jeux de données (non redistribués dans ce dépôt)

| Source                             | Tâche      | Licence                                | Usage                                          |
| ---------------------------------- | ---------- | -------------------------------------- | ---------------------------------------------- |
| **BibleTTS** (éwé)                 | ASR        | **CC-BY-SA 4.0**                       | Entraînement — attribution requise             |
| GhanaNLP (navigation, bible-audio) | ASR / MT   | CC-BY 4.0                              | Entraînement                                   |
| OPUS _bible-uedin_ (ee-en, ee-fr)  | Traduction | Domaine public / conditions OPUS       | Entraînement                                   |
| Kaggle _tchaye59_ (éwé-anglais)    | Traduction | Sources mixtes (dont textes religieux) | Recherche                                      |
| Corpus pivot _français-anglais_    | Traduction | À vérifier                             | Entraînement (pivot)                           |
| DUDH / _glk360_                    | Traduction | Domaine public                         | —                                              |
| Wikipédia éwé                      | Monolingue | CC-BY-SA 4.0                           | Attribution requise                            |
| `masakhane/mafand`                 | Traduction | **CC-BY-NC**                           | **Évaluation seule** (exclu de l'entraînement) |

Catalogue complet et statuts : [`data/CATALOG.md`](data/CATALOG.md).

## Résumé des licences de ce dépôt

- **Code source** → MIT (`LICENSE`).
- **Modèles entraînés** → CC-BY-NC-4.0 (sur Hugging Face, cartes dans `model_cards/`).
- **Données brutes** → non versionnées ; se référer à la licence de chaque source.

> ⚠️ Récapitulatif fourni de bonne foi, sans valeur d'avis juridique. Vérifier les
> licences amont avant tout usage commercial ou toute rediffusion des données.
