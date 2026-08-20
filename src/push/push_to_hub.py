"""
push_to_hub.py
==============
Dépose les datasets éwé sur Hugging Face Hub (dépôts privés).

Datasets disponibles :
  - translation  : nllb_translation/*.jsonl  (~quelques Mo)
  - asr          : ewe_asr/  (~19 Go, FLAC + metadata.jsonl)
  - all          : les deux

Usage :
    python3 push_to_hub.py --dataset translation
    python3 push_to_hub.py --dataset asr
    python3 push_to_hub.py --dataset all
    python3 push_to_hub.py --dataset all --username MonPseudo --private

Authentification (une des deux méthodes) :
  1. Variable d'environnement : export HF_TOKEN=hf_xxx
  2. CLI huggingface :          huggingface-cli login
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import CommitOperationDelete, HfApi, login
except ImportError:
    print("[ERREUR] huggingface_hub requis : pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration par défaut
# ---------------------------------------------------------------------------
DEFAULT_USERNAME     = "romaricnadjire"
REPO_TRANSLATION     = "ewe-nllb-translation"
REPO_ASR             = "ewe-asr-whisper"
TRANSLATION_DIR      = Path("data/processed/translation")
ASR_DIR              = Path("data/raw/asr/bible_ewe")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_token() -> str | None:
    """Récupère le token HF depuis l'env ou le cache huggingface-cli."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token or None   # None → huggingface_hub utilise le cache login


def ensure_repo(api: HfApi, repo_id: str, repo_type: str, private: bool) -> None:
    """Crée le dépôt s'il n'existe pas encore."""
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"  Dépôt existant : {repo_id}")
    except Exception:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
        visibility = "privé" if private else "public"
        print(f"  Dépôt créé ({visibility}) : {repo_id}")


# ---------------------------------------------------------------------------
# Restructuration : évite la limite HF de 10 000 fichiers par répertoire
# ---------------------------------------------------------------------------

def restructure_split_for_hf(split_dir: Path, max_per_chunk: int = 9000) -> None:
    """
    Si split_dir/ contient plus de max_per_chunk fichiers FLAC à la racine,
    les déplace dans des sous-dossiers chunk_00/, chunk_01/, …  et met à jour
    metadata.jsonl en conséquence.

    Idempotent : ne fait rien si les fichiers sont déjà dans des sous-dossiers.
    """
    flac_files = sorted(split_dir.glob("*.flac"))
    if len(flac_files) <= max_per_chunk:
        return  # rien à faire

    print(f"  [restructure] {split_dir.name}/ : {len(flac_files)} flac "
          f"> {max_per_chunk} → découpage en chunks…")

    # Lire metadata existante
    meta_path = split_dir / "metadata.jsonl"
    records: list[dict] = []
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    # Indexer les métadonnées par nom de fichier
    meta_by_name: dict[str, dict] = {r["file_name"]: r for r in records}

    # Découper et déplacer
    new_records: list[dict] = []
    for idx, flac in enumerate(flac_files):
        chunk_name = f"chunk_{idx // max_per_chunk:02d}"
        chunk_dir = split_dir / chunk_name
        chunk_dir.mkdir(exist_ok=True)
        dest = chunk_dir / flac.name
        shutil.move(str(flac), str(dest))

        # Mettre à jour le chemin dans la métadonnée
        old_key = flac.name
        new_key = f"{chunk_name}/{flac.name}"
        rec = meta_by_name.get(old_key, {}).copy()
        rec["file_name"] = new_key
        new_records.append(rec)

    # Réécrire metadata.jsonl
    with meta_path.open("w", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_chunks = (len(flac_files) - 1) // max_per_chunk + 1
    print(f"  [restructure] {split_dir.name}/ → {n_chunks} chunks "
          f"(≤ {max_per_chunk} fichiers chacun) — metadata.jsonl mis à jour")


# ---------------------------------------------------------------------------
# Nettoyage HF : supprime les anciens fichiers plats (avant restructuration)
# ---------------------------------------------------------------------------

def cleanup_flat_files_on_hf(
    api: HfApi,
    repo_id: str,
    split: str = "train",
    batch_size: int = 500,
) -> None:
    """
    Supprime sur HuggingFace les fichiers <split>/*.flac engagés à plat
    (avant la restructuration en chunks).  Ne retransmet aucun contenu
    binaire — XET/LFS déduplique par hash.

    Idempotent : ne fait rien si aucun fichier plat n'est présent.
    """
    print(f"  [cleanup] Listage des fichiers distants dans {repo_id}…")
    try:
        all_remote = list(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception as exc:
        print(f"  [cleanup] Impossible de lister les fichiers distants : {exc}")
        return

    # Fichiers plats : split/filename.flac  (exactement 1 slash, pas de sous-dossier)
    flat = [
        f for f in all_remote
        if f.startswith(f"{split}/")
        and f.endswith(".flac")
        and f.count("/") == 1
    ]

    if not flat:
        print(f"  [cleanup] Aucun fichier plat dans {split}/ — rien à supprimer")
        return

    total = len(flat)
    n_batches = (total - 1) // batch_size + 1
    print(f"  [cleanup] {total} fichiers plats à supprimer "
          f"({n_batches} commits de ≤ {batch_size})…")

    for i in range(0, total, batch_size):
        batch = flat[i : i + batch_size]
        ops = [CommitOperationDelete(path_in_repo=p) for p in batch]
        batch_num = i // batch_size + 1
        while True:
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"[cleanup] Suppression fichiers plats {split}/ "
                                   f"(batch {batch_num}/{n_batches})",
                )
                print(f"  [cleanup] Batch {batch_num}/{n_batches} supprimé "
                      f"({len(batch)} fichiers)")
                break
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "Too Many Requests" in msg:
                    match = re.search(r"Retry after (\d+) seconds", msg)
                    wait = int(match.group(1)) + 10 if match else 130
                    print(f"  [cleanup] Rate limit — attente {wait}s avant batch "
                          f"{batch_num}/{n_batches}…")
                    time.sleep(wait)
                else:
                    raise

    print(f"  [cleanup] Terminé — {total} fichiers plats supprimés de {split}/")


# ---------------------------------------------------------------------------
# Push translation (JSONL, ~quelques Mo)
# ---------------------------------------------------------------------------

def push_translation(
    api: HfApi, username: str, private: bool, repo_name: str = REPO_TRANSLATION
) -> None:
    repo_id = f"{username}/{repo_name}"
    print("\n" + "=" * 60)
    print(f"[TRANSLATION]  {TRANSLATION_DIR}  →  {repo_id}")
    print("=" * 60)

    if not TRANSLATION_DIR.exists():
        print(f"  [ERREUR] Dossier introuvable : {TRANSLATION_DIR.resolve()}")
        return

    files = list(TRANSLATION_DIR.glob("*.jsonl"))
    if not files:
        print(f"  [ERREUR] Aucun fichier .jsonl dans {TRANSLATION_DIR}")
        return

    print(f"  Fichiers à uploader : {[f.name for f in files]}")
    ensure_repo(api, repo_id, "dataset", private)

    # Statistiques calculées dynamiquement (le dataset a pu être enrichi en fr↔en)
    split_counts = {"train": 0, "validation": 0, "test": 0}
    pair_counts = {"ewe-en": 0, "ewe-fr": 0, "en-fr": 0}
    for f in sorted(files):
        n = 0
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                keys = set(json.loads(line)["translation"].keys())
                if {"ewe_Latn", "eng_Latn"} <= keys:
                    pair_counts["ewe-en"] += 1
                elif {"ewe_Latn", "fra_Latn"} <= keys:
                    pair_counts["ewe-fr"] += 1
                elif {"eng_Latn", "fra_Latn"} <= keys:
                    pair_counts["en-fr"] += 1
        split_counts[f.stem] = n
    total_lines = sum(split_counts.values())

    # README avec citations de toutes les sources
    readme_translation = f"""---
task_categories:
- translation
language:
- ee
- en
- fr
license: other
---

# Ewe Translation Dataset (NLLB format)

Dataset de traduction éwé↔anglais et éwé↔français pour le fine-tuning de
`facebook/nllb-200-distilled-600M`. Assemblé et structuré dans le cadre d'un
projet académique (M1 INF2229 — Informatique & Gestion de Données).

## Statistiques

| Split | Lignes |
|-------|--------|
| train | {split_counts['train']} |
| validation | {split_counts['validation']} |
| test | {split_counts['test']} |
| **Total** | **{total_lines}** |

- Paires éwé↔anglais : {pair_counts['ewe-en']}
- Paires éwé↔français : {pair_counts['ewe-fr']}
- Paires anglais↔français : {pair_counts['en-fr']}
- Split éwé 80/10/10 (seed=42) + ajout anglais↔français (test fixe, validation prélevée du train)

## Format

```json
{{"translation": {{"ewe_Latn": "...", "eng_Latn": "..."}}}}
{{"translation": {{"ewe_Latn": "...", "fra_Latn": "..."}}}}
```

## Chargement

```python
from datasets import load_dataset

ds = load_dataset(
    "{repo_id}",
    data_files={{
        "train":      "train.jsonl",
        "validation": "validation.jsonl",
        "test":       "test.jsonl",
    }},
    token=True,
)
```

## Sources et citations

### 1. OPUS bible-uedin — Licence CC0 1.0

Textes bibliques parallèles (éwé↔anglais, 16 001 paires).

```bibtex
@article{{christodoulopoulos2015massively,
  title={{A massively parallel corpus: the Bible in 100 languages}},
  author={{Christodoulopoulos, Christos and Steedman, Mark}},
  journal={{Language Resources and Evaluation}},
  volume={{49}},
  number={{2}},
  pages={{375--395}},
  year={{2015}},
  publisher={{Springer}}
}}
```

```bibtex
@inproceedings{{tiedemann2012parallel,
  title={{Parallel Data, Tools and Interfaces in OPUS}},
  author={{Tiedemann, J{{\"o}}rg}},
  booktitle={{LREC}},
  year={{2012}}
}}
```

### 2. OPUS QED — Recherche uniquement (© QCRI)

Sous-titres éducatifs (éwé↔anglais, 284 paires).

```bibtex
@inproceedings{{abdelali2014amara,
  title={{The AMARA Corpus: Building Parallel Language Resources for and from the
         Web}},
  author={{Abdelali, Ahmed and Guzman, Francisco and Sajjad, Hassan and Vogel,
         Stephan}},
  booktitle={{LREC}},
  year={{2014}}
}}
```

### 3. Kaggle — EWE-English Bilingual Pairs (licence « Other »)

Paires éwé↔anglais crawlées (28 606 paires brutes).

```bibtex
@misc{{gbedevi2020ewe,
  title={{EWE-English Bilingual Pairs}},
  url={{https://www.kaggle.com/dsv/1462736}},
  doi={{10.34740/KAGGLE/DSV/1462736}},
  publisher={{Kaggle}},
  author={{Akouyo Yvette GBEDEVI and Jude Tchaye-Kondi}},
  year={{2020}}
}}
```

### 4. Kaggle — ewe\_english\_train.csv (licence MIT)

Paires éwé↔anglais (21 714 paires brutes).
Source : https://www.kaggle.com/datasets/kuroio/ewe-english-train-csv

### 5. Peterlin — Ewe language project (© 2009-2011)

Phrasebook et dictionnaire éwé↔anglais (1 070 paires).

```
Djoubogbe Kossi Afoutou & Piotr Kozłowski
http://www.peterlin.pl/ewe/
Usage académique uniquement.
```

### 6. Zenodo — French to Ewe Dataset

Paires français↔éwé (23 807 paires brutes).
Source : zenodo.org (French\_to\_ewe\_dataset.xlsx)
"""

    api.upload_file(
        path_or_fileobj=readme_translation.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Ajout README avec citations de toutes les sources",
    )

    api.upload_folder(
        folder_path=str(TRANSLATION_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload nllb_translation JSONL (éwé↔anglais + éwé↔français)",
        ignore_patterns=["README.md"],
    )
    print(f"\n  ✔ Translation dataset disponible sur :")
    print(f"    https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Push ASR (audiofolder FLAC, ~19 Go — reprise automatique)
# ---------------------------------------------------------------------------

def push_asr(api: HfApi, username: str, private: bool) -> None:
    repo_id = f"{username}/{REPO_ASR}"
    print("\n" + "=" * 60)
    print(f"[ASR]  {ASR_DIR}  →  {repo_id}")
    print("=" * 60)

    if not ASR_DIR.exists():
        print(f"  [ERREUR] Dossier introuvable : {ASR_DIR.resolve()}")
        return

    splits = [d.name for d in sorted(ASR_DIR.iterdir()) if d.is_dir()]
    print(f"  Splits détectés : {splits}")

    # Compter les fichiers par split
    for split in splits:
        n_flac = len(list((ASR_DIR / split).glob("*.flac")))
        meta   = (ASR_DIR / split / "metadata.jsonl")
        n_meta = sum(1 for _ in meta.open()) if meta.exists() else 0
        print(f"  {split:12s} : {n_flac} flac  |  {n_meta} lignes metadata.jsonl")

    print(f"\n  Taille estimée : ~19 Go — upload en cours (reprend si interrompu)...")
    print(f"  Ne pas interrompre sauf nécessité (Ctrl+C relançable sans tout ré-uploader)")

    # Découper les splits trop volumineux (limite HF : 10 000 fichiers/répertoire)
    for split in splits:
        restructure_split_for_hf(ASR_DIR / split)

    ensure_repo(api, repo_id, "dataset", private)

    # Supprimer les fichiers plats éventuellement déjà commités sur HF
    # (opération git pure — XET déduplique le contenu, aucun retransfer binaire)
    cleanup_flat_files_on_hf(api, repo_id, split="train")

    # upload_large_folder gère la reprise (chunked, cache local .cache_lfs/)
    api.upload_large_folder(
        folder_path=str(ASR_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        # num_workers=4,            # décommenter pour paralléliser l'upload
    )

    # Ajoute un README complet avec citation BibleTTS (obligatoire CC BY-SA 4.0)
    readme_content = f"""---
task_categories:
- automatic-speech-recognition
language:
- ee
license: cc-by-sa-4.0
---

# Ewe ASR Dataset (BibleTTS)

Dataset audio éwé pour le fine-tuning de modèles ASR (Whisper, MMS).
Assemblé dans le cadre d'un projet académique (M1 INF2229 — Informatique &
Gestion de Données).

**Licence : CC BY-SA 4.0** — dérivé de BibleTTS (OpenSLR 129).
Toute redistribution doit conserver la même licence et citer les auteurs.

## Statistiques

| Split | Fichiers .flac | Transcriptions |
|-------|---------------|----------------|
| train | 22 192 | 22 192 |
| validation | 186 | 186 |
| test | 66 | 66 |
| **Total** | **22 444** | **22 444** |

Audio : mono 48 kHz FLAC (format original BibleTTS).

## Structure (audiofolder)

```
ewe_asr/
  train/        (22 192 fichiers .flac + metadata.jsonl)
  validation/   (   186 fichiers .flac + metadata.jsonl)
  test/         (    66 fichiers .flac + metadata.jsonl)
```

Chaque `metadata.jsonl` contient des lignes de la forme :
```json
{{"file_name": "GEN_001_Verse_001.flac", "transcription": "..."}}
```

## Chargement

```python
from datasets import load_dataset, Audio

ds = load_dataset(
    "{repo_id}",
    token=True,
)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

# Pour Whisper : colonne 'transcription'
# Pour MMS    : renommer en 'sentence' si besoin
```

## Source et citation

Ce dataset est dérivé du corpus **BibleTTS** (OpenSLR 129).
Si vous utilisez ce dataset, veuillez citer :

```bibtex
@inproceedings{{meyer2022bibletts,
  title     = {{BibleTTS: a large, high-fidelity, multilingual, and uniquely
               African speech corpus}},
  author    = {{Josh Meyer and David Adelani and Edresson Casanova and
               Alp {{\"O}}ktem and Daniel Whitenack and Julian Weber and
               Salomon Kabongo Kabenamualu and Elizabeth Salesky and
               Iroro Orife and Colin Leong and Perez Ogayo and
               Chris Chinenye Emezue and Jonathan Mukiibi and
               Salomey Osei and Apelete Agbolo and Victor Akinode and
               Bernard Opoku and Olanrewaju Samuel and Jesujoba Alabi and
               Shamsuddeen Hassan Muhammad}},
  booktitle = {{Interspeech}},
  publisher = {{ISCA}},
  year      = {{2022}},
  url       = {{https://arxiv.org/pdf/2207.03546.pdf}}
}}
```

Dataset original : https://www.openslr.org/129/
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Ajout README audiofolder",
    )

    print(f"\n  ✔ ASR dataset disponible sur :")
    print(f"    https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dépose les datasets éwé sur Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset",
        choices=["translation", "asr", "all"],
        default="all",
        help="Dataset à uploader (défaut : all)",
    )
    parser.add_argument(
        "--username",
        default=DEFAULT_USERNAME,
        help=f"Nom d'utilisateur HF (défaut : {DEFAULT_USERNAME})",
    )
    parser.add_argument(
        "--repo",
        default=REPO_TRANSLATION,
        help=f"Nom du dépôt translation (défaut : {REPO_TRANSLATION}). "
             "Indiquez un NOUVEAU nom pour ne pas écraser l'ancien dataset.",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dépôt privé (défaut : True). --no-private pour public.",
    )
    args = parser.parse_args()

    # --- Authentification ---
    token = get_token()
    if token:
        login(token=token, add_to_git_credential=False)
        print("  Authentifié via HF_TOKEN")
    else:
        print("  Aucun HF_TOKEN trouvé — utilisation du cache huggingface-cli login")
        print("  Si erreur 401, exécutez : huggingface-cli login")

    api = HfApi()

    visibility = "privé" if args.private else "public"
    print(f"\nMode : {args.dataset}  |  Dépôt : {visibility}  |  Compte : {args.username}")

    if args.dataset in ("translation", "all"):
        push_translation(api, args.username, args.private, repo_name=args.repo)

    if args.dataset in ("asr", "all"):
        push_asr(api, args.username, args.private)

    print("\n" + "=" * 60)
    print("TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()
