# Structure du Dataset ASR Éwé — `ewe_asr/`

## Contexte du projet

Ce dataset audio est utilisé pour le fine-tuning d'un modèle ASR (Automatic Speech Recognition)
en langue **éwé** dans le cadre d'un pipeline NLP complet :

| Tâche                        | Modèle                                           | Données                              |
| ---------------------------- | ------------------------------------------------ | ------------------------------------ |
| **ASR** (voix → texte éwé)   | `openai/whisper-medium` ou `facebook/mms-1b-all` | `ewe_asr/`                           |
| **Traduction** (éwé → fr/en) | `facebook/nllb-200-distilled-600M`               | `opus/`, `kaggle/`, `peterlin/`      |
| **TTS** (texte éwé → voix)   | `facebook/mms-tts-ewe`                           | _(pré-entraîné, pas de fine-tuning)_ |

---

## Structure originale (`ewe/`)

Le dossier source était organisé par livre biblique, un niveau trop profond
pour être reconnu directement par `datasets.load_dataset("audiofolder", ...)` :

```
ewe/
  train/
    GEN/           ← livre (Genèse)
      GEN_001_Verse_003.flac
      GEN_001_Verse_003.txt   ← transcription éwé (1 ligne)
      ...
    EXO/
      ...
  dev/             ← 186 fichiers audio dans EZR/
    EZR/
      EZR_001_Verse_000.flac
      EZR_001_Verse_000.txt
      ...
  test/
    COL/
      COL_001_Verse_000.flac
      ...
```

**Problèmes :**

- `dev/` n'est pas le nom de split reconnu par Hugging Face (`validation` est attendu)
- Les audio et transcriptions sont répartis dans N sous-dossiers → impossible de générer
  un `metadata.jsonl` plat automatiquement
- L'absence de `metadata.jsonl` force `audiofolder` à ignorer les transcriptions

---

## Structure cible (`ewe_asr/`)

Format **audiofolder** standard, lisible directement par `datasets.load_dataset` :

```
ewe_asr/
  train/
    metadata.jsonl          ← index de tous les fichiers audio + transcriptions
    GEN_001_Verse_003.flac
    GEN_001_Verse_004.flac
    ...                     ← tous les .flac aplatis dans un seul dossier
  validation/               ← "dev" renommé selon la convention HF
    metadata.jsonl
    EZR_001_Verse_000.flac
    ...
  test/
    metadata.jsonl
    COL_001_Verse_000.flac
    ...
```

### Format de `metadata.jsonl`

Chaque ligne est un objet JSON avec exactement deux clés :

```jsonl
{"file_name": "GEN_001_Verse_003.flac", "transcription": "Mawu ɖe gbe foo be: ‟Haɖe anye ɣleti""}
{"file_name": "GEN_001_Verse_004.flac", "transcription": "Mawu kpɔ ɣleti la, eye enye nyuie."}
```

> **Note Whisper vs MMS** : La colonne s'appelle `transcription` (convention Whisper).
> Pour MMS / wav2vec2, renommer la colonne en `sentence` dans le script de fine-tuning :
> `ds = ds.rename_column("transcription", "sentence")`

---

## Génération du dataset

### Prérequis

Aucune dépendance externe — utilise uniquement la bibliothèque standard Python 3.10+.

### Lancer le script

```bash
# Depuis la racine du projet
python restructure_ewe.py
```

Options disponibles :

```bash
python restructure_ewe.py --src ewe --dst ewe_asr   # chemins par défaut
python restructure_ewe.py --dry-run                  # simulation sans copie
python restructure_ewe.py --dst /chemin/custom       # destination personnalisée
```

Le script :

1. Copie tous les `.flac` depuis `ewe/{split}/{BOOK}/` vers `ewe_asr/{split}/`
2. Lit la transcription dans le `.txt` correspondant
3. Génère `ewe_asr/{split}/metadata.jsonl`
4. Affiche un rapport avec le nombre de fichiers par split et les éventuelles anomalies

---

## Chargement avec Hugging Face `datasets`

### Chargement de base

```python
from datasets import load_dataset

ds = load_dataset("audiofolder", data_dir="./ewe_asr")
print(ds)
# DatasetDict({
#     train: Dataset({features: ['audio', 'transcription'], num_rows: ...}),
#     validation: Dataset({features: ['audio', 'transcription'], num_rows: ...}),
#     test: Dataset({features: ['audio', 'transcription'], num_rows: ...})
# })
```

### Normalisation de la fréquence d'échantillonnage (obligatoire pour Whisper)

Whisper et MMS attendent un audio à **16 000 Hz**.
La conversion est faite automatiquement par `datasets` :

```python
from datasets import Audio

ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
print(ds["train"][0])
# {
#   'audio': {'array': array([...]), 'sampling_rate': 16000, 'path': '...'},
#   'transcription': 'Mawu ɖe gbe foo be: ‟Haɖe anye ɣleti"'
# }
```

### Inspecter un exemple

```python
sample = ds["train"][0]
print("Transcription :", sample["transcription"])
print("Sampling rate :", sample["audio"]["sampling_rate"])
print("Durée (s)     :", len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"])
```

---

## Utilisation pour le fine-tuning Whisper

### Extraction des features audio

```python
from transformers import WhisperFeatureExtractor, WhisperTokenizer

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="ewe", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"])
```

### Adaptation pour MMS (wav2vec2)

```python
# Renommer la colonne transcription → sentence (convention MMS)
ds = ds.rename_column("transcription", "sentence")

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
# adapter target_lang selon la langue cible du processeur
```

---

## Vérification de l'intégrité

```python
total = sum(len(ds[split]) for split in ds)
print(f"Total exemples : {total}")

# Vérifier qu'aucune transcription n'est vide
for split in ds:
    empty = [i for i, ex in enumerate(ds[split]) if not ex["transcription"].strip()]
    if empty:
        print(f"[WARN] {split} : {len(empty)} transcription(s) vide(s) aux indices {empty[:5]}")
    else:
        print(f"[OK] {split} : aucune transcription vide")
```

---

## Remarques

- Les fichiers `.flac` dans `ewe_asr/` sont des **copies** — l'original `ewe/` n'est pas modifié.
- Le nommage `{BOOK}_{chapter}_Verse_{verse}.flac` garantit l'unicité des fichiers
  même après aplatissement (pas de collision entre livres).
- Ajouter `ewe_asr/` dans `.gitignore` si le dépôt est sur GitHub/GitLab
  (les fichiers audio sont volumineux — préférer Git LFS ou un dépôt HF Hub).
