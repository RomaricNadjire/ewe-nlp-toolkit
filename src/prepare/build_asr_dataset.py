#!/usr/bin/env python3
"""Fusionne les corpus ASR éwé déjà traités (16 kHz) en un dataset audiofolder unique.

Entrée  : `data/processed/asr/<source>/<split>/` (sorties de process_asr_16k.py).
Sortie  : `data/processed/asr_mixed/<split>/chunk_XX/*.wav` + `metadata.jsonl`
          `{file_name, sentence, duration, dataset}` (schéma audiofolder).

Étapes :
  1. collecte les clips de chaque source/split (métadonnées + chemins absolus) ;
  2. **déduplication acoustique** : hash MD5 des échantillons PCM 16 bits, on
     supprime les doublons exacts (mêmes samples), en conservant la 1ʳᵉ source ;
  3. mélange déterministe (seed) pour un mix homogène ;
  4. écriture par **chunks** de ≤ 9000 fichiers (limite HF 10 000/dossier) via
     hardlink (fallback copie) — pas de recompression ;
  5. `manifest.json` récapitulatif (clips, heures, répartition par source).

Les plafonds horaires (Bible, Navigation) sont appliqués EN AMONT par
`process_asr_16k.py --cap-hours`. La fusion ne fait qu'assembler + dédupliquer.

Push (réseau, optionnel) :
    .venv/bin/python src/prepare/build_asr_dataset.py --push

Dépendances : soundfile, numpy   (+ huggingface_hub pour --push)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np  # noqa: F401  (utilisé indirectement via soundfile)
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"[ERREUR] Dépendance manquante ({exc.name}).\n"
        f"         Installez :  .venv/bin/pip install soundfile numpy"
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_ASR = REPO_ROOT / "data" / "processed" / "asr"
MERGED_ROOT = REPO_ROOT / "data" / "processed" / "asr_mixed"

# Sources fusionnées par défaut (ignorées si absentes de data/processed/asr/).
# NB : ghana_bible_combined (partie éwé) est un DOUBLON exact de ghananlp_ewe_bible_audio
#      (corrélation audio = 1.0) -> exclu (n'apporterait aucune diversité acoustique).
DEFAULT_SOURCES = [
    "ghananlp_health_unicef_ewe",
    "ghananlp_navigation_ewe_speech",
    "waxal_ewe_single_speaker",
    "bible_ewe",
    "ghananlp_ewe_bible_audio",     # corpus biblique distinct (segmentation/voix), ~5 h
]
MAX_PER_CHUNK = 9000
DEFAULT_REPO = "romaricnadjire/ewe-asr-mixed-16k"


# ---------------------------------------------------------------------------
# Collecte
# ---------------------------------------------------------------------------

def collect_records(sources: list[str]) -> dict[str, list[dict]]:
    """Retourne {split: [record, …]} où record = {path, sentence, duration, dataset}."""
    by_split: dict[str, list[dict]] = defaultdict(list)
    for src in sources:
        src_dir = PROC_ASR / src
        if not src_dir.exists():
            print(f"  [ignoré] source absente : {src_dir.relative_to(REPO_ROOT)}")
            continue
        splits = sorted(d.name for d in src_dir.iterdir()
                        if d.is_dir() and (d / "metadata.jsonl").exists())
        for split in splits:
            split_dir = src_dir / split
            meta = split_dir / "metadata.jsonl"
            n = 0
            with meta.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    by_split[split].append({
                        "path": str(split_dir / rec["file_name"]),
                        "sentence": rec.get("sentence", ""),
                        "duration": float(rec.get("duration", 0.0)),
                        "dataset": rec.get("dataset", src),
                    })
                    n += 1
            print(f"  {src:<34} {split:<12} {n:>7} clips")
    return by_split


# ---------------------------------------------------------------------------
# Déduplication acoustique
# ---------------------------------------------------------------------------

def _audio_hash(path: str) -> str | None:
    """MD5 des échantillons PCM 16 bits (indépendant du header WAV)."""
    try:
        data, _sr = sf.read(path, dtype="int16", always_2d=False)
    except Exception:  # noqa: BLE001
        return None
    return hashlib.md5(data.tobytes()).hexdigest()


def dedup(records: list[dict], workers: int) -> tuple[list[dict], int, int]:
    """Supprime les doublons acoustiques exacts. Conserve la 1ʳᵉ occurrence."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(_audio_hash, [r["path"] for r in records]))
    seen: set[str] = set()
    kept: list[dict] = []
    n_dup = n_bad = 0
    for rec, h in zip(records, hashes):
        if h is None:
            n_bad += 1
            continue
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        kept.append(rec)
    return kept, n_dup, n_bad


# ---------------------------------------------------------------------------
# Écriture chunkée
# ---------------------------------------------------------------------------

def write_split(split: str, records: list[dict], out_split: Path,
                *, max_per_chunk: int, seed: int) -> tuple[int, float, Counter]:
    random.Random(seed).shuffle(records)
    out_split.mkdir(parents=True, exist_ok=True)
    meta_path = out_split / "metadata.jsonl"
    per_dataset: Counter = Counter()
    total_dur = 0.0
    with meta_path.open("w", encoding="utf-8") as mf:
        for idx, rec in enumerate(records):
            chunk = f"chunk_{idx // max_per_chunk:02d}"
            chunk_dir = out_split / chunk
            chunk_dir.mkdir(exist_ok=True)
            out_name = f"{idx:06d}.wav"
            dest = chunk_dir / out_name
            src = Path(rec["path"])
            if dest.exists():
                dest.unlink()
            try:
                os.link(src, dest)            # hardlink : zéro copie disque
            except OSError:
                shutil.copyfile(src, dest)    # FS différent → copie
            mf.write(json.dumps({
                "file_name": f"{chunk}/{out_name}",
                "sentence": rec["sentence"],
                "duration": rec["duration"],
                "dataset": rec["dataset"],
            }, ensure_ascii=False) + "\n")
            per_dataset[rec["dataset"]] += 1
            total_dur += rec["duration"]
    return len(records), total_dur, per_dataset


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(sources: list[str], *, workers: int, seed: int, max_per_chunk: int,
          clean: bool) -> dict:
    if clean and MERGED_ROOT.exists():
        shutil.rmtree(MERGED_ROOT)
    print("\n[1/3] Collecte des sources traitées…")
    by_split = collect_records(sources)
    if not by_split:
        raise SystemExit("[ERREUR] Aucune source traitée trouvée dans "
                         f"{PROC_ASR.relative_to(REPO_ROOT)} — lancez d'abord "
                         "process_asr_16k.py.")

    summary: dict = {"splits": {}, "by_dataset": Counter(), "hours_total": 0.0,
                     "rows_total": 0}
    for split, records in by_split.items():
        print(f"\n[2/3] Split « {split} » — {len(records)} clips collectés")
        kept, n_dup, n_bad = dedup(records, workers)
        print(f"       dédup : -{n_dup} doublons, -{n_bad} illisibles → {len(kept)} gardés")
        print(f"[3/3] Écriture chunkée → {(MERGED_ROOT / split).relative_to(REPO_ROOT)}")
        n, dur, per_ds = write_split(split, kept, MERGED_ROOT / split,
                                     max_per_chunk=max_per_chunk, seed=seed)
        print(f"       {n} clips, {dur / 3600:.2f} h, par source : {dict(per_ds)}")
        summary["splits"][split] = {
            "rows": n, "hours": round(dur / 3600, 3),
            "dup_removed": n_dup, "unreadable": n_bad,
            "by_dataset": dict(per_ds),
        }
        summary["by_dataset"].update(per_ds)
        summary["hours_total"] += dur / 3600
        summary["rows_total"] += n

    summary["by_dataset"] = dict(summary["by_dataset"])
    summary["hours_total"] = round(summary["hours_total"], 3)
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    (MERGED_ROOT / "manifest.json").write_text(json.dumps({
        "sources": sources,
        "target_sr": 16000,
        "channels": 1,
        "subtype": "PCM_16",
        "max_per_chunk": max_per_chunk,
        "seed": seed,
        "schema": "audiofolder (.wav + metadata.jsonl {file_name, sentence, duration, dataset})",
        **summary,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n===== RÉSUMÉ FUSION =====")
    print(f"  Total : {summary['rows_total']} clips  ({summary['hours_total']} h)")
    for ds, c in summary["by_dataset"].items():
        print(f"    {ds:<34} {c:>7} clips")
    return summary


# ---------------------------------------------------------------------------
# Push HF (réseau, optionnel)
# ---------------------------------------------------------------------------

def _readme(summary: dict, repo_id: str) -> str:
    rows = "\n".join(
        f"| {s} | {d['rows']} | {d['hours']} |"
        for s, d in summary["splits"].items()
    )
    by_ds = "\n".join(f"- `{k}` : {v} clips" for k, v in summary["by_dataset"].items())
    return f"""---
task_categories:
- automatic-speech-recognition
language:
- ee
license: cc-by-sa-4.0
---

# Ewe ASR — Mixed 16 kHz

Corpus ASR éwé **multi-sources** ré-échantillonné en **16 kHz mono (WAV PCM_16)**,
assemblé pour le fine-tuning de modèles de reconnaissance vocale (MMS, Whisper).
Projet académique (M1 INF2229 — Informatique & Gestion de Données).

**Licence : CC BY-SA 4.0** (la plus restrictive des sources — partage à l'identique).

## Statistiques

| Split | Clips | Heures |
|-------|-------|--------|
{rows}

**Total : {summary['rows_total']} clips — {summary['hours_total']} h**

Répartition par source :
{by_ds}

## Structure (audiofolder)

```
<split>/
  chunk_00/  (≤ 9000 .wav)
  chunk_01/  …
  metadata.jsonl   {{"file_name": "chunk_00/000000.wav", "sentence": "...", "duration": 4.2, "dataset": "..."}}
```

## Chargement

```python
from datasets import load_dataset, Audio
ds = load_dataset("{repo_id}", token=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
```

## Sources & citations

- **BibleTTS** (OpenSLR 129, CC BY-SA 4.0) — Meyer et al., *BibleTTS*, Interspeech 2022.
- **GhanaNLP** — corpus éwé (navigation, santé/UNICEF).
- **Waxal** — corpus de parole éwé.

Le corpus conserve les diacritiques éwé (ɖ ɔ ɛ ŋ ƒ ʋ …). Traitement :
mixdown mono, resampling soxr 16 kHz, normalisation NFC du texte, filtrage des
durées et déduplication acoustique (hash PCM).
"""


def push(repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("[ERREUR] huggingface_hub requis : pip install huggingface_hub")

    if not MERGED_ROOT.exists():
        raise SystemExit(f"[ERREUR] {MERGED_ROOT} absent — lancez d'abord le build.")
    summary = json.loads((MERGED_ROOT / "manifest.json").read_text(encoding="utf-8"))

    # On force le jeton d'écriture EXPLICITEMENT sur l'instance HfApi : sinon une
    # variable d'environnement HF_TOKEN (souvent en lecture seule) prend le dessus
    # et provoque un « 403 Forbidden: you must use a write token ».
    token = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN") \
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("[ERREUR] Aucun jeton trouvé (HF_TOKEN_WRITE attendu dans .env).")
    api = HfApi(token=token)

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"  Dépôt existant : {repo_id}")
    except Exception:  # noqa: BLE001
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"  Dépôt créé ({'privé' if private else 'public'}) : {repo_id}")

    api.upload_file(
        path_or_fileobj=_readme(summary, repo_id).encode(),
        path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
        commit_message="README audiofolder (mixed 16k)",
    )
    print("  Upload des audios (reprise automatique si interrompu)…")
    api.upload_large_folder(
        folder_path=str(MERGED_ROOT), repo_id=repo_id, repo_type="dataset",
    )
    print(f"\n  ✔ https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="sources à fusionner (séparées par des virgules)")
    ap.add_argument("--workers", type=int, default=16, help="threads de dédup (défaut 16)")
    ap.add_argument("--seed", type=int, default=42, help="graine du mélange (défaut 42)")
    ap.add_argument("--max-per-chunk", type=int, default=MAX_PER_CHUNK,
                    help=f"fichiers max par chunk (défaut {MAX_PER_CHUNK})")
    ap.add_argument("--no-clean", action="store_true",
                    help="ne pas vider data/processed/asr_mixed/ avant build")
    ap.add_argument("--push", action="store_true", help="pousser sur HF après le build")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"dépôt HF (défaut {DEFAULT_REPO})")
    ap.add_argument("--public", action="store_true", help="dépôt public (défaut privé)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.only.split(",")] if args.only else DEFAULT_SOURCES
    summary = build(sources, workers=args.workers, seed=args.seed,
                    max_per_chunk=args.max_per_chunk, clean=not args.no_clean)

    if args.push:
        if summary["rows_total"] == 0:
            sys.exit("[ERREUR] Rien à pousser (0 clip).")
        print("\n[push] Envoi vers Hugging Face…")
        push(args.repo, private=not args.public)


if __name__ == "__main__":
    main()
