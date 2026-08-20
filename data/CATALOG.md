# Catalogue des corpus — Traduction & ASR éwé

Catalogue central de **toutes** les sources de données du projet (éwé `ee` / `ewe_Latn`),
qu'elles soient déjà téléchargées ou à récupérer. Sert de référence unique pour la phase
de collecte (P1) puis de nettoyage/fusion (P2/P3).

## Légende des statuts

| Statut           | Signification                                                                      |
| ---------------- | ---------------------------------------------------------------------------------- |
| ✅ local         | Déjà présent dans `data/raw/`                                                      |
| ⬇️ à télécharger | Script disponible dans `src/download/` ou `src/scraping/`                          |
| 🗂️ catalogué     | Repéré mais **pas encore** téléchargé (priorité basse / pivot / monolingue)        |
| 🧪 éval. seule   | À n'utiliser **que** pour l'évaluation, **jamais** dans le train (fuite / licence) |

## Conventions

- **Format cible traduction** : JSONL, une ligne =
  `{"translation": {"ewe_Latn": "...", "eng_Latn"|"fra_Latn": "..."}}`
- **Format cible ASR** : `audiofolder` (dossier `split/` + `metadata.jsonl` `{file_name, sentence}`).
- Chaque source téléchargée reçoit un `manifest.json` (provenance, licence, format, nb lignes, date).
- Les gros corpus (minés, audio) sont **gitignored** (cf. `.gitignore`) ; seules les petites
  sources texte sont versionnées.
- Les comptes de lignes marqués `~` proviennent de la documentation amont et sont **à confirmer**
  au téléchargement (les scripts écrivent le compte réel dans le `manifest.json`).

---

## 1. Traduction — sources parallèles

### 1.1 Déjà présentes en local (`data/raw/translation/`)

| Source (dossier)     | Langues | Lignes      | Format                 | Type                              | Licence                 | Statut   |
| -------------------- | ------- | ----------- | ---------------------- | --------------------------------- | ----------------------- | -------- |
| `opus_bible_ee_en`   | ee↔en   | 48 003      | CSV + `.ee/.en/.xml`   | Bible (bible-uedin, OPUS)         | domaine public / CC     | ✅ local |
| `kaggle_ewe_english` | ee↔en   | 30 850      | CSV (`,EWE,ENGLISH`)   | parallèle Ghana-NLP               | voir source Kaggle      | ✅ local |
| `opus_qed_ee_en`     | ee↔en   | 852         | CSV + `.ee/.en/.xml`   | sous-titres éducatifs (QED, OPUS) | CC-BY (QED)             | ✅ local |
| `peterlin`           | ee↔en   | ~400 + dico | CSV (scraping)         | dictionnaire + phrases            | scraping (peterlin.com) | ✅ local |
| `francais_anglais`   | en↔fr   | 175 622     | CSV (`English,French`) | générique (pivot en-fr)           | voir source             | ✅ local |

> `francais_anglais` n'est **pas** de l'éwé : c'est un appui en↔fr (pivot) intégré au jeu
> multilingue `data/processed/translation/` (cf. `src/prepare/add_fr_en_to_nllb.py`).

**Jeu consolidé après P2** (`data/processed/translation/`, produit par
`src/prepare/build_translation_dataset.py`) :

| Fichier             | Lignes        | Contenu                                                                        |
| ------------------- | ------------- | ------------------------------------------------------------------------------ |
| `train.jsonl`       | **307 503**   | sources **propres** trilingues : en-fr 168 450 · ewe-fr 75 576 · ewe-en 63 477 |
| `train_mined.jsonl` | **1 791 752** | corpus **minés** filtrés (sim ≥ 1.06) : ewe-en 1 611 323 · ewe-fr 180 429      |
| `validation.jsonl`  | 11 763        | held-out (inchangé)                                                            |
| `test.jsonl`        | 8 754         | held-out (inchangé)                                                            |

> P2 : normalisation NFC, filtres (écriture étrangère, ratio de longueur, copies, longueur max,
> score de similarité pour les minés), **déduplication canonique** (clé indépendante de l'ordre des
> langues) et exclusion de tout ce qui figure dans `validation`/`test` (anti-fuite, 20 515 clés).
> L'ancien `train.jsonl` (222 567) est préservé dans `data/interim/translation_prev/`
> (`train_original_multilingual.jsonl`, immuable) et ré-injecté comme source propre.
> Exclus : `opus_qed_*` (turc désaligné), `glk360` (éwé monolingue), `kaggle_ghana_maternal_health`
> (batches bruts). Rapport détaillé par source : `data/processed/translation/build_report.json`.

### 1.2 Hugging Face — à télécharger (`src/download/download_hf_translation.py`)

| Dataset HF                                        | Langues            | Lignes (~)         | Type               | Licence               | Statut           |
| ------------------------------------------------- | ------------------ | ------------------ | ------------------ | --------------------- | ---------------- |
| `adaboubvincent/translation-eng-fra-ewe`          | ee+en+fr (3 voies) | ~311 282           | humain/mixte       | à vérifier            | ⬇️ à télécharger |
| `ghananlpcommunity/english-ewe-sentence-pairs-4m` | ee↔en              | ~4 408 322         | **miné** (bruité)  | à vérifier (GhanaNLP) | ⬇️ à télécharger |
| `michsethowusu/ewe-french_sentence-pairs`         | ee↔fr              | ~1–10 M            | **miné** (bruité)  | à vérifier            | ⬇️ à télécharger |
| `Ghana-NLP/EWE_ENGLISH_PARALLEL_TEXT`             | ee↔en              | à confirmer        | parallèle          | à vérifier            | ⬇️ à télécharger |
| `ghananlpcommunity/navigation-corpus-ewe`         | ee↔en              | à confirmer        | domaine navigation | à vérifier            | ⬇️ à télécharger |
| `glk360/fr-ewe-corpus`                            | ee↔fr              | ~60                | DUDH (UDHR)        | domaine public        | ⬇️ à télécharger |
| `masakhane/mafand` (config `en-ewe`)              | ee↔en              | 2026 / 1414 / 1563 | humain, presse     | **CC-BY-NC**          | 🧪 éval. seule   |

> ⚠️ Corpus **minés** (4M, 1–10M) = beaucoup de bruit/désalignement → à **filtrer par score de
> similarité** en P2 avant tout entraînement. Ne pas mélanger brut au train.
> ⚠️ `mafand` est **CC-BY-NC** : utilisable pour l'évaluation, à exclure d'un train destiné à
> une diffusion non restreinte.

### 1.3 OPUS — à télécharger (`src/download/download_opus.py`, via l'API OPUS)

| Corpus      | Paire | Lignes    | Type                   | Licence              | Statut           |
| ----------- | ----- | --------- | ---------------------- | -------------------- | ---------------- |
| NLLB        | ee↔fr | 1 039 385 | **miné**               | conditions OPUS/NLLB | ⬇️ à télécharger |
| bible-uedin | ee↔fr | 8 004     | Bible                  | domaine public / CC  | ⬇️ à télécharger |
| QED         | ee↔fr | 166       | sous-titres            | CC-BY                | ⬇️ à télécharger |
| Tatoeba     | ee↔en | (présent) | phrases communautaires | CC-BY                | ⬇️ à télécharger |

> Comptes confirmés via l'API OPUS. **NLLB ee↔en n'existe pas** comme bitext OPUS direct
> (utiliser plutôt `ghananlpcommunity/english-ewe-sentence-pairs-4m` côté HF pour le ee↔en miné).
> `bible-uedin`/`QED` **ee↔en** sont déjà en local (§1.1) ; ici on récupère le **français** (ee↔fr) manquant.

### 1.4 Kaggle — à télécharger (`src/download/download_kaggle.py`, nécessite `~/.kaggle/kaggle.json`)

| Dataset Kaggle                                            | Langues   | Type                 | Statut           |
| --------------------------------------------------------- | --------- | -------------------- | ---------------- |
| `tchaye59/eweenglish-bilingual-pairs`                     | ee↔en     | paires bilingues     | ⬇️ à télécharger |
| `yvicherita/ewe-language-corpus`                          | ee (+en?) | corpus éwé           | ⬇️ à télécharger |
| `ghanaairesnet/ghana-maternal-health-q-and-a-dataset-ewe` | ee        | Q/R santé maternelle | ⬇️ à télécharger |

---

## 2. ASR (parole → texte) éwé

### 2.1 Déjà présentes en local (`data/raw/asr/`)

| Source      | Lignes                           | Format                                          | Type                | Statut       |
| ----------- | -------------------------------- | ----------------------------------------------- | ------------------- | ------------ |
| `bible_ewe` | train 22 192 · val 186 · test 66 | audiofolder (`chunk_00..02` + `metadata.jsonl`) | Bible audio alignée | ✅ local     |
| `waxal_ewe` | — (placeholder vide)             | —                                               | Waxal NLP           | 🗂️ catalogué |

### 2.2 Hugging Face — à télécharger (`src/download/download_hf_asr.py`)

Identifiants HF **vérifiés** (existence confirmée via l'API HF).

| Dataset HF                                                   | Lignes (~)         | Type                | Statut           |
| ------------------------------------------------------------ | ------------------ | ------------------- | ---------------- |
| `Paywinful/ewe-asr`                                          | ~22 258            | parole lue          | ⬇️ à télécharger |
| `ghananlpcommunity/navigation-corpus-ewe-speech`             | ~49 348            | navigation          | ⬇️ à télécharger |
| `ghananlpcommunity/ewe-bible-audio-text-tts`                 | ~48 775            | Bible TTS/ASR       | ⬇️ à télécharger |
| `ghananlpcommunity/ghana-nlp-health-UNICEF-asr-ewe`          | à confirmer        | santé (UNICEF)      | ⬇️ à télécharger |
| `ghananlpcommunity/ghana-bible-combined-90k-twi-ewe-dagbani` | ~90 k (filtrer ee) | multi-langue Bible  | ⬇️ à télécharger |
| `1nnocent/waxal-ewe-tts-filtered-single-speaker`             | à confirmer        | Waxal mono-locuteur | ⬇️ à télécharger |

> Audio = **volumineux** → reste en local **gitignored**, push éventuel vers HF en P3.
> `--limit` permet de tester un petit échantillon avant le téléchargement complet.

---

## 3. Pivots & monolingue (catalogués, téléchargement différé)

Utiles pour le multilingue (pivots africains) ou le pré-entraînement / TTS (monolingue).
À ne récupérer qu'après la collecte ee↔en / ee↔fr prioritaire.

| Dataset                                                      | Langues               | Type                         | Statut       |
| ------------------------------------------------------------ | --------------------- | ---------------------------- | ------------ |
| `michsethowusu/ewe-{hausa,yoruba,fon,igbo,…}_sentence-pairs` | ee↔langues africaines | **miné** (pivots)            | 🗂️ catalogué |
| `VKAgbesi/Ewe_News_Dataset`                                  | ee                    | ~4 264 articles (monolingue) | 🗂️ catalogué |
| `ghananlpcommunity/ewe-sentiments-corpus-300k`               | ee                    | sentiments (monolingue)      | 🗂️ catalogué |
| `ghananlpcommunity/ewe-emotions-corpus-300k`                 | ee                    | émotions (monolingue)        | 🗂️ catalogué |

---

## 4. Scraping (sources libres, `src/scraping/`)

| Cible                              | Script                    | Langues         | Licence         | Statut                                        |
| ---------------------------------- | ------------------------- | --------------- | --------------- | --------------------------------------------- |
| Dictionnaire/phrases peterlin      | `scrape_peterlin.py`      | ee↔en           | usage recherche | ✅ scrapé (1052 mots + 14 phrases + 6 textes) |
| Wikipédia éwé (`ee.wikipedia.org`) | `scrape_wikipedia_ewe.py` | ee (monolingue) | CC-BY-SA        | ✅ opérationnel (~1352 articles)              |

> Wikipédia (CC-BY-SA) : conserver l'attribution. Le script respecte `maxlag`, applique un
> back-off sur les réponses 429/503 et limite le débit. Sortie : `data/raw/monolingual/ewe_wikipedia/`.

**Sources écartées (déjà couvertes ailleurs) :**

- **DUDH (Déclaration universelle des droits de l'homme)** : l'URL Unicode `udhr_ewe` renvoie 404 ;
  la version ee↔fr est déjà fournie par `glk360/fr-ewe-corpus` (§1.2, `download_hf_translation.py`).
- **eBible / Bible éwé** : déjà couverte par OPUS `bible-uedin` (ee-en / ee-fr, §1.3,
  `download_opus.py`) et par l'audio `bible_ewe` (§2.1, ASR).

---

## 5. Évaluation uniquement (🧪 — jamais dans le train)

| Jeu                                        | Langues | Rôle               | Licence  |
| ------------------------------------------ | ------- | ------------------ | -------- |
| `facebook/flores` (FLORES-200, `ewe_Latn`) | ee↔\*   | benchmark standard | CC-BY-SA |
| `masakhane/mafand` test (`en-ewe`)         | ee↔en   | benchmark presse   | CC-BY-NC |

> **FLEURS ne contient PAS l'éwé** — ne pas l'utiliser comme source ASR éwé.
> Garder FLORES/MAFAND **hors** des splits d'entraînement pour éviter toute fuite d'évaluation.

---

## 6. Ordre de priorité (P1)

1. **Traduction HF** (`adaboubvincent`, GhanaNLP, OPUS ee↔fr) — meilleur rapport valeur/effort.
2. **ASR HF** (`Paywinful/ewe-asr`, navigation-speech, bible-audio).
3. **Scraping** (Wikipédia, DUDH, eBible) — en dernier.

Nettoyage/déduplication/fusion = **P2** ; ré-entraînement + push audio HF = **P3**.
