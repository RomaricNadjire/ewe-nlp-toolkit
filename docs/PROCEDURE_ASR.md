# Procédure de constitution du dataset ASR Éwé (`ewe-asr-mixed-16k`)

> Document de synthèse — collecte, sélection, nettoyage et fusion des corpus de
> parole éwé pour l'entraînement de modèles ASR (MMS / Whisper).
> Projet M1 INF2229 — Informatique & Gestion de Données.

---

## 0. Vue d'ensemble (résumé exécutif)

| Étape                | Outil                                                                   | Sortie                                |
| -------------------- | ----------------------------------------------------------------------- | ------------------------------------- |
| Inventaire           | [src/prepare/inventory_asr.py](../src/prepare/inventory_asr.py)         | `data/interim/asr/inventory.json`     |
| Téléchargement       | [src/download/download_hf_asr.py](../src/download/download_hf_asr.py)   | `data/raw/asr/<source>/`              |
| Normalisation 16 kHz | [src/prepare/process_asr_16k.py](../src/prepare/process_asr_16k.py)     | `data/processed/asr/<source>/`        |
| Fusion + dédup       | [src/prepare/build_asr_dataset.py](../src/prepare/build_asr_dataset.py) | `data/processed/asr_mixed/`           |
| Publication          | `build_asr_dataset.py --push`                                           | HF `romaricnadjire/ewe-asr-mixed-16k` |

**Principe directeur :** maximiser la **diversité acoustique** (locuteurs, genre,
tons, accents, durées) tout en **bornant le biais de domaine** (lecture biblique
sur-représentée) et en respectant une **bande passante très limitée**.

---

## 1. Constat initial : volume vs bande passante

- **Matériel local :** ordinateur portable, **12 cœurs CPU, pas de GPU**.
- **Réseau :** débit très faible et instable → impossible de rapatrier
  naïvement des corpus de plusieurs Go (datasets HF en **parquet**, shards de
  ~390 Mo, jusqu'à ~14,5 Go par corpus).
- **Conséquences stratégiques :**
  1. Ne **jamais** télécharger un corpus entier « pour voir » → on **échantillonne**
     d'abord (200 clips) pour mesurer durées, fréquence d'échantillonnage, qualité.
  2. Téléchargement en **streaming** (lecture parquet à la volée, arrêt anticipé)
     plutôt qu'en téléchargement complet des shards.
  3. **Parallélisme borné** (`max_workers=4`) + **`sleep` anti-429** pour ne pas
     se faire bannir par le Hub.
  4. **Reprise après interruption** obligatoire (`progress.json`).

---

## 2. Inventaire & échantillonnage initial

Script : [src/prepare/inventory_asr.py](../src/prepare/inventory_asr.py)

- Lecture **uniquement des en-têtes** audio (`soundfile.info`, pas de décodage
  complet) → rapide même sur des dizaines de milliers de fichiers.
- Mesures agrégées par source/split : nb clips, heures totales, min/médiane/max,
  nb clips < 1 s et > 5 s, fréquences d'échantillonnage, canaux, formats.

### Résultats mesurés (en-têtes)

| Source                           | Clips  | Heures     | SR     | Médiane    | Observation                                             |
| -------------------------------- | ------ | ---------- | ------ | ---------- | ------------------------------------------------------- |
| `bible_ewe` (BibleTTS local)     | 22 444 | **77,6 h** | 48 kHz | 11,6 s     | Bloc volumineux **monotone** (lecture biblique)         |
| `ghananlp_health_unicef_ewe`     | 200\*  | 1,8 h      | 16 kHz | **32,3 s** | Voix M/F, **segments longs** (santé/UNICEF)             |
| `ghananlp_navigation_ewe_speech` | 49 348 | ~27 h      | 24 kHz | **2,0 s**  | Segments **courts**, parole de navigation               |
| `waxal_ewe_single_speaker`       | 489    | ~1,6 h     | 48 kHz | 6–8 s      | **Mono-locuteur**, outliers > 300 s                     |
| `ghananlp_ewe_bible_audio`       | 200\*  | 0,43 h     | 24 kHz | 7,3 s      | Bible, corpus **distinct** de BibleTTS                  |
| `ghana_bible_combined`           | 200\*  | 0,43 h     | 24 kHz | 7,3 s      | Bible multilingue (filtre éwé) — **doublon** (voir 3.5) |

\* échantillon de sondage (200 clips) — volume complet récupéré ensuite par streaming.

> **Contrôle d'identité acoustique** réalisé sur les deux corpus bibliques GhanaNLP
> (corrélation de Pearson sur le signal rééchantillonné à 16 kHz, 5 versets communs) :
> `ghana_bible_combined` (partie éwé) **= 1,000** vs `ghananlp_ewe_bible_audio`
> (mêmes durées exactes, même SR 24 kHz) → **enregistrement identique** : la partie
> éwé de `ghana-bible-combined` n'est qu'une **redite** de `ewe-bible-audio`.
> À l'inverse, `ghananlp_ewe_bible_audio` **diffère** de `bible_ewe` (BibleTTS) :
> contenu/segmentation distincts (versets regroupés vs 1 verset/clip) — confirmé
> _a posteriori_ par la **déduplication acoustique (0 doublon supprimé)**.

---

## 3. Sélection stratégique des sources

### 3.1 UNICEF Health — source **primaire** (genre & durée)

Retenue en priorité pour la **diversité de genre (M/F)** et des **segments longs**
(médiane 32 s) qui apportent du contexte prosodique absent des autres corpus.

### 3.2 Navigation — diversité (3ᵉ priorité)

~27 h de parole spontanée courte (médiane 2 s) → complète UNICEF par des énoncés
brefs et un vocabulaire fonctionnel. **Plafonnée** (`--cap-hours 20`) pour ne pas
écraser les autres sources.

### 3.3 Waxal — secondaire (mono-locuteur)

Conservée mais marginale (mono-locuteur) ; un **filtre de durée max** élimine les
outliers (> 300 s) qui déséquilibreraient l'entraînement.

### 3.4 BibleTTS local — vocabulaire, **plafonné anti-biais**

Le corpus `bible_ewe` (≈ 77,6 h) est la plus grosse ressource mais **monotone**
(style liturgique, peu de locuteurs). Le garder en entier **biaiserait** le modèle
vers la lecture biblique. → **sous-échantillonné à ~13 h** par **tirage aléatoire
déterministe** (anti-biais), pour conserver la richesse lexicale sans la
sur-représentation.

### 3.5 Réintégration d'**un** corpus biblique alternatif (diversité acoustique)

L'intention initiale était de réintégrer **deux** corpus bibliques GhanaNLP
(`ghananlp_ewe_bible_audio` et `ghana_bible_combined`) pour ajouter des voix/tons
alternatifs à BibleTTS. Un **contrôle d'identité acoustique** (voir § 2) a montré
que la partie éwé de `ghana_bible_combined` est l'**enregistrement identique**
(corr = 1,000) de `ghananlp_ewe_bible_audio` : ce **n'est pas une voix alternative**
mais un doublon.

→ **Décision corrigée :**

- `ghana_bible_combined` **exclu** — aucun gain de diversité, et la dédup l'aurait
  de toute façon supprimé ; on évite ainsi de télécharger ~3 000 clips redondants
  sur un réseau lent.
- `ghananlp_ewe_bible_audio` **conservé** (~5 h, `--cap-hours 5`) : corpus réellement
  **distinct** de BibleTTS (segmentation/contenu différents, **0 doublon** détecté
  à la fusion). Le plafonnement strict est ce qui distingue « diversité » de « biais ».

---

## 4. Pipeline technique

### 4.1 Téléchargement résumable (streaming, anti-429)

Script : [src/download/download_hf_asr.py](../src/download/download_hf_asr.py)

- `--streaming` : itère le dataset sans télécharger les 14,5 Go de parquet.
- `--max-workers 4` : décodage/écriture parallèles bornés.
- `--sleep 0.3` : throttle proactif contre l'erreur HTTP **429**.
- `--limit N` : arrêt après N clips écrits (estimé pour viser ~5–6 h brutes).
- **Reprise** via `progress.json` (`n_seen`, `n_written`, `done`) + écriture
  atomique (`.part` → `os.replace`) → une coupure réseau ne corrompt rien.
- Backoff exponentiel sur 429/5xx (lecture de l'en-tête `Retry-After`).

### 4.2 Normalisation 16 kHz (mono, PCM_16, soxr, NFC)

Script : [src/prepare/process_asr_16k.py](../src/prepare/process_asr_16k.py)

- **Resampling** soxr vers **16 000 Hz** (toutes sources : 48 k / 24 k / 16 k).
- **Mixdown mono** (moyenne des canaux), sortie **WAV PCM_16** (compact, standard ASR).
- **Texte :** normalisation **Unicode NFC**, suppression des caractères de contrôle
  et du BOM, compactage des espaces — **maintien strict des diacritiques éwé**
  (`ɖ ɔ ɛ ŋ ƒ ʋ Ɖ Ɔ Ɛ Ŋ …`, tons), vérifié par test de round-trip.
- **Filtres :** durée `[--min-sec 1.0 ; --max-sec 60.0]`, clips vides/illisibles écartés.
- **12 cœurs** via `ProcessPoolExecutor` (décodage + resampling CPU-bound).
- Schéma de sortie : `{file_name, sentence, duration, dataset}`.

### 4.3 Échantillonnage par plafond horaire (`--cap-hours`)

- Sonde les durées (en-têtes, multithread), **mélange déterministe** (graine),
  accumule les clips jusqu'au plafond **par split** (les petits splits passent
  entiers). Garantit des sous-ensembles **reproductibles** (Bible 13 h, alt. 5 h).

### 4.4 Fusion + déduplication + chunking

Script : [src/prepare/build_asr_dataset.py](../src/prepare/build_asr_dataset.py)

- **Union par split** de toutes les sources préparées.
- **Déduplication acoustique** : empreinte **MD5 des échantillons PCM 16 bits**
  → suppression des doublons exacts (indépendant de l'en-tête WAV).
- **Mélange déterministe** puis écriture en **chunks ≤ 9 000 fichiers** (limite HF
  10 000/dossier) via **hardlink** (zéro copie disque, repli copie si FS différent).
- **Métadonnées centralisées** `metadata.jsonl` + `manifest.json` récapitulatif.
- **Publication** (`--push`) : dépôt **privé** `ewe-asr-mixed-16k`, README
  audiofolder + citations, `upload_large_folder` (reprise automatique).

---

## 5. Récapitulatif des volumes (**fusion finale mesurée**)

| Source                           | Rôle                   | Plafond      | Clips      | Heures              |
| -------------------------------- | ---------------------- | ------------ | ---------- | ------------------- |
| `ghananlp_navigation_ewe_speech` | Diversité courte       | 20 h         | 30 505     | 20,00 h             |
| `bible_ewe` (BibleTTS)           | Vocabulaire            | 14 h         | 4 010      | 14,00 h             |
| `ghananlp_ewe_bible_audio`       | Diversité acoustique   | 5 h          | 2 493      | 5,00 h              |
| `ghananlp_health_unicef_ewe`     | Primaire (genre/durée) | —            | 199        | 1,79 h              |
| `waxal_ewe_single_speaker`       | Mono-locuteur          | filtre durée | 440        | 1,21 h              |
| **TOTAL**                        |                        |              | **37 647** | **42,00 h**         |
| ~~`ghana_bible_combined`~~       | ~~diversité~~          | —            | **0**      | **exclu (doublon)** |

> Répartition par split (`data/processed/asr_mixed/manifest.json`) :
> **train** 37 315 clips / 40,81 h · **test** 103 / 0,38 h · **validation** 229 / 0,81 h.
> Déduplication acoustique : **0 doublon** supprimé (sources réellement disjointes).

---

## 6. Reproductibilité (commandes)

```bash
# 0) Inventaire (lecture en-têtes, rapide)
.venv/bin/python src/prepare/inventory_asr.py --save data/interim/asr/inventory.json

# 1) Téléchargement (streaming, reprise auto, anti-429)
.venv/bin/python src/download/download_hf_asr.py --only ghananlp_health_unicef_ewe --max-workers 4
.venv/bin/python src/download/download_hf_asr.py --only ghananlp_navigation_ewe_speech --streaming --sleep 0.3
.venv/bin/python src/download/download_hf_asr.py --only waxal_ewe_single_speaker --streaming --sleep 0.3
.venv/bin/python src/download/download_hf_asr.py --only ghananlp_ewe_bible_audio --streaming --limit 3000 --max-workers 4 --sleep 0.3
# ghana_bible_combined NON téléchargé : doublon exact de ghananlp_ewe_bible_audio (corr=1.0)

# 2) Normalisation 16 kHz (+ plafonds anti-biais)
.venv/bin/python src/prepare/process_asr_16k.py --only ghananlp_health_unicef_ewe
.venv/bin/python src/prepare/process_asr_16k.py --only ghananlp_navigation_ewe_speech --cap-hours 20
.venv/bin/python src/prepare/process_asr_16k.py --only waxal_ewe_single_speaker
.venv/bin/python src/prepare/process_asr_16k.py --only bible_ewe --cap-hours 13
.venv/bin/python src/prepare/process_asr_16k.py --only ghananlp_ewe_bible_audio --cap-hours 5

# 3) Fusion (dédup acoustique + chunking) — 5 sources, ghana_bible_combined exclu
.venv/bin/python src/prepare/build_asr_dataset.py

# 4) Publication HF (token via .env)
set -a; source ./.env; set +a
.venv/bin/python src/prepare/build_asr_dataset.py --no-clean --push
```

---

## 7. Journal chronologique

1. **Constat** capacité réseau/disque → décision d'échantillonner avant tout.
2. **Inventaire** des durées (en-têtes) sur tous les corpus candidats.
3. **Sélection** : UNICEF (primaire, genre/durée) + Navigation (diversité) +
   Waxal (mono-locuteur) + BibleTTS (vocabulaire).
4. **Anti-biais** : plafonnement de BibleTTS (77,6 h → ~13 h) par tirage aléatoire.
5. **Outils** : téléchargeur résumable (streaming, anti-429), normaliseur 16 kHz
   12 cœurs (diacritiques préservés), fusion (dédup acoustique + chunking).
6. **Réintégration ciblée** d'un corpus biblique alternatif. Le contrôle d'identité
   acoustique révèle que `ghana_bible_combined` (éwé) est un **doublon exact**
   (corr = 1,000) de `ghananlp_ewe_bible_audio` → **exclu**. Seul
   `ghananlp_ewe_bible_audio` est conservé (~5 h) pour la **diversité acoustique**.
7. **Fusion finale** → `data/processed/asr_mixed/` (**37 647 clips / 42,0 h**,
   0 doublon) → publication HF privée.

---

_Dernière mise à jour : 2026-06-21._
