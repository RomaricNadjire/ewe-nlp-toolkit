"""
restructure_ewe.py
==================
Restructure le dossier `ewe/` au format audiofolder Hugging Face (ASR).

Structure source :
    ewe/
      {split}/          (train | dev | test)
        {BOOK}/         (GEN, EZR, COL, ...)
          {ID}.flac
          {ID}.txt      <- transcription éwé sur une seule ligne

Structure cible :
    ewe_asr/
      train/
        metadata.jsonl
        *.flac
      validation/       <- dev renommé (convention HF)
        metadata.jsonl
        *.flac
      test/
        metadata.jsonl
        *.flac

Chaque ligne de metadata.jsonl :
    {"file_name": "GEN_001_Verse_003.flac", "transcription": "..."}

Usage :
    python restructure_ewe.py
    python restructure_ewe.py --src ewe --dst ewe_asr --dry-run

Compatible : Whisper (colonne `transcription`) et MMS (renommer en `sentence`).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPLIT_MAP = {
    "train": "train",
    "dev": "validation",
    "test": "test",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_transcription(txt_path: Path) -> str | None:
    """Lit et nettoie la transcription depuis un fichier .txt."""
    try:
        text = txt_path.read_text(encoding="utf-8").strip()
        return text if text else None
    except OSError as exc:
        return None


def process_split(
    src_split_dir: Path,
    dst_split_dir: Path,
    split_name: str,
    dry_run: bool,
) -> dict:
    """
    Parcourt src_split_dir/{BOOK}/*.flac, copie les .flac et génère metadata.jsonl.

    Retourne un dict avec les statistiques du split.
    """
    stats = {
        "split": split_name,
        "copied": 0,
        "skipped_no_txt": 0,
        "skipped_empty_txt": 0,
        "already_exists": 0,
        "anomalies": [],
    }

    if not src_split_dir.exists():
        print(f"  [WARN] Split source introuvable : {src_split_dir} — ignoré.")
        return stats

    if not dry_run:
        dst_split_dir.mkdir(parents=True, exist_ok=True)

    # Collecte tous les .flac du split (tous les sous-dossiers)
    flac_files = sorted(src_split_dir.rglob("*.flac"))

    if not flac_files:
        print(f"  [WARN] Aucun fichier .flac trouvé dans {src_split_dir}")
        return stats

    metadata_lines = []

    for flac_path in flac_files:
        stem = flac_path.stem          # ex. GEN_001_Verse_003
        txt_path = flac_path.with_suffix(".txt")

        # Vérification du .txt correspondant
        if not txt_path.exists():
            stats["skipped_no_txt"] += 1
            stats["anomalies"].append(f"TXT manquant : {flac_path.relative_to(src_split_dir.parent.parent)}")
            continue

        transcription = read_transcription(txt_path)

        if transcription is None:
            stats["skipped_empty_txt"] += 1
            stats["anomalies"].append(f"TXT vide : {txt_path.relative_to(src_split_dir.parent.parent)}")
            continue

        dst_flac = dst_split_dir / flac_path.name

        if not dry_run:
            if dst_flac.exists():
                stats["already_exists"] += 1
            shutil.copy2(flac_path, dst_flac)

        metadata_lines.append({
            "file_name": flac_path.name,
            "transcription": transcription,
        })
        stats["copied"] += 1

    # Écriture du metadata.jsonl
    if not dry_run and metadata_lines:
        jsonl_path = dst_split_dir / "metadata.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for entry in metadata_lines:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Restructure ewe/ en dataset audiofolder Hugging Face (ASR)."
    )
    parser.add_argument(
        "--src",
        default="ewe",
        help="Dossier source (défaut : ewe)",
    )
    parser.add_argument(
        "--dst",
        default="data/raw/asr/bible_ewe",
        help="Dossier de destination (défaut : data/raw/asr/bible_ewe)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulation : affiche les stats sans copier ni écrire.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"[ERREUR] Dossier source introuvable : {src_root.resolve()}", file=sys.stderr)
        sys.exit(1)

    mode_label = "[DRY-RUN] " if args.dry_run else ""
    print(f"{mode_label}Restructuration : {src_root.resolve()} → {dst_root.resolve()}")
    print("-" * 60)

    all_stats = []
    total_copied = 0
    total_anomalies = 0

    for src_split_name, dst_split_name in SPLIT_MAP.items():
        src_split_dir = src_root / src_split_name
        dst_split_dir = dst_root / dst_split_name

        print(f"\n[{src_split_name} → {dst_split_name}]")
        stats = process_split(src_split_dir, dst_split_dir, dst_split_name, args.dry_run)
        all_stats.append(stats)

        print(f"  ✔ Copiés    : {stats['copied']}")
        if stats["already_exists"]:
            print(f"  ↺ Existants : {stats['already_exists']} (écrasés)")
        if stats["skipped_no_txt"]:
            print(f"  ✘ Sans TXT  : {stats['skipped_no_txt']}")
        if stats["skipped_empty_txt"]:
            print(f"  ✘ TXT vide  : {stats['skipped_empty_txt']}")

        total_copied += stats["copied"]
        total_anomalies += stats["skipped_no_txt"] + stats["skipped_empty_txt"]

    # Résumé global
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    for s in all_stats:
        print(f"  {s['split']:<12} : {s['copied']} fichiers")
    print(f"  {'TOTAL':<12} : {total_copied} fichiers")
    if total_anomalies:
        print(f"\n  ⚠  {total_anomalies} anomalie(s) détectée(s) :")
        for s in all_stats:
            for msg in s["anomalies"]:
                print(f"     - {msg}")
    else:
        print("\n  Aucune anomalie détectée.")

    if args.dry_run:
        print("\n[DRY-RUN] Aucun fichier n'a été copié ni créé.")
    else:
        print(f"\nDataset prêt dans : {dst_root.resolve()}")
        print("\nPour charger le dataset avec Hugging Face :")
        print("  from datasets import load_dataset")
        print(f'  ds = load_dataset("audiofolder", data_dir="{args.dst}")')


if __name__ == "__main__":
    main()
