#!/usr/bin/env python3
"""Nettoie et ré-échantillonne les corpus ASR éwé en 16 kHz mono (multiprocessing).

Pour chaque source `data/raw/asr/<source>/<split>/` :
  - lit `metadata.jsonl` (clé texte `sentence` OU `transcription`) ;
  - résout le chemin audio relatif au split (gère les sous-dossiers `chunk_*`) ;
  - convertit en **16 kHz mono WAV PCM_16** (resampling soxr, mixdown des canaux) ;
  - normalise le texte (NFC + nettoyage), **en préservant les diacritiques éwé** ;
  - calcule la durée et filtre les clips hors bornes / vides / illisibles ;
  - écrit `data/processed/asr/<source>/<split>/` + `metadata.jsonl`
    `{file_name, sentence, duration, dataset}` (compatible audiofolder).

Le travail CPU (décodage + resampling) est réparti sur plusieurs cœurs via
`ProcessPoolExecutor`. Les fichiers bruts ne sont jamais modifiés.

Dépendances : soundfile, soxr, numpy

Exemples :
    .venv/bin/python src/prepare/process_asr_16k.py --only ghananlp_health_unicef_ewe
    .venv/bin/python src/prepare/process_asr_16k.py --only bible_ewe --max-sec 40 --workers 12
    .venv/bin/python src/prepare/process_asr_16k.py --only waxal_ewe_single_speaker --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    import soxr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"[ERREUR] Dépendance manquante ({exc.name}).\n"
        f"         Installez :  .venv/bin/pip install soundfile soxr numpy"
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ASR = REPO_ROOT / "data" / "raw" / "asr"
PROC_ASR = REPO_ROOT / "data" / "processed" / "asr"
TARGET_SR = 16000

# Caractères de contrôle (hors tabulation/retour qui deviennent des espaces).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """NFC + suppression contrôle/BOM + espaces compactés. Diacritiques éwé préservés."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\ufeff", "")
    s = _CONTROL_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _process_one(task: tuple) -> tuple:
    """Worker : décode -> mono 16 kHz -> filtre -> écrit le WAV. Picklable (niveau module)."""
    idx, audio_path, text, out_dir, target_sr, min_sec, max_sec = task
    sentence = norm_text(text)
    if not sentence:
        return (idx, None, "", 0.0, "texte_vide")
    try:
        arr, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001  (clip illisible/corrompu)
        return (idx, None, sentence, 0.0, "illisible")
    if arr.ndim > 1:                                   # mixdown stéréo -> mono
        arr = arr.mean(axis=1)
    if sr != target_sr:
        try:
            arr = soxr.resample(arr, sr, target_sr)
        except Exception:  # noqa: BLE001
            return (idx, None, sentence, 0.0, "resample_echec")
    arr = np.asarray(arr, dtype="float32")
    dur = len(arr) / target_sr
    if dur < min_sec or (max_sec and dur > max_sec):
        return (idx, None, sentence, round(dur, 3), "duree_hors_bornes")
    out_name = f"{idx:06d}.wav"
    out_path = Path(out_dir) / out_name
    tmp = out_path.with_name(out_name + ".part")
    try:
        sf.write(tmp, arr, target_sr, format="WAV", subtype="PCM_16")
        os.replace(tmp, out_path)                      # WAV complet ou absent
    except Exception:  # noqa: BLE001
        return (idx, None, sentence, round(dur, 3), "ecriture_echec")
    return (idx, out_name, sentence, round(dur, 3), "ok")


def _read_metadata(split_dir: Path) -> list[tuple[str, str]]:
    """Lit metadata.jsonl -> [(file_name, texte)] ; clé texte = sentence|transcription."""
    entries: list[tuple[str, str]] = []
    meta = split_dir / "metadata.jsonl"
    with meta.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            fn = rec.get("file_name")
            txt = rec.get("sentence") or rec.get("transcription") or ""
            if fn:
                entries.append((fn, txt))
    return entries


def _select_capped(raw_split_dir: Path, entries: list[tuple[str, str]],
                   cap_hours: float, seed: int, workers: int) -> tuple[list[tuple[str, str]], float]:
    """Sélection aléatoire de clips jusqu'à `cap_hours` (anti-biais, tirage seedé).

    Sonde les durées brutes (sf.info, en parallèle), mélange, puis accumule
    jusqu'au plafond. Retourne (entrées retenues triées, secondes retenues).
    """
    import random

    def probe(path: str) -> float:
        try:
            info = sf.info(path)
            return info.frames / info.samplerate if info.samplerate else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    paths = [str(raw_split_dir / fn) for fn, _ in entries]
    with ThreadPoolExecutor(max_workers=max(8, workers)) as ex:
        durations = list(ex.map(probe, paths))

    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    cap_sec = cap_hours * 3600.0
    acc = 0.0
    chosen: list[int] = []
    for i in order:
        if acc >= cap_sec:
            break
        chosen.append(i)
        acc += durations[i]
    chosen.sort()
    return [entries[i] for i in chosen], acc


def process_split(src_name: str, split: str, raw_split_dir: Path, out_split_dir: Path,
                  *, min_sec: float, max_sec: float, workers: int, limit: int | None,
                  cap_hours: float, seed: int) -> tuple[int, Counter, float]:
    entries = _read_metadata(raw_split_dir)
    if limit is not None:
        entries = entries[:limit]
    if cap_hours and cap_hours > 0:
        before = len(entries)
        entries, sel_sec = _select_capped(raw_split_dir, entries, cap_hours, seed, workers)
        if len(entries) < before:
            print(f"  {split:<12} plafond {cap_hours}h -> {len(entries)}/{before} clips "
                  f"(~{sel_sec / 3600:.2f} h brut, seed={seed})")
    out_split_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (i, str(raw_split_dir / fn), txt, str(out_split_dir), TARGET_SR, min_sec, max_sec)
        for i, (fn, txt) in enumerate(entries)
    ]
    stats: Counter = Counter()
    kept = 0
    total_dur = 0.0
    meta_out = out_split_dir / "metadata.jsonl"
    with meta_out.open("w", encoding="utf-8") as mf, \
            ProcessPoolExecutor(max_workers=workers) as ex:
        results = ex.map(_process_one, tasks, chunksize=16)
        for idx, out_name, sentence, dur, status in sorted(results):
            stats[status] += 1
            if status == "ok":
                mf.write(json.dumps(
                    {"file_name": out_name, "sentence": sentence,
                     "duration": dur, "dataset": src_name},
                    ensure_ascii=False) + "\n")
                kept += 1
                total_dur += dur
    print(f"  {split:<12} {kept:>6}/{len(tasks):<6} clips gardés  "
          f"{total_dur/3600:>6.2f} h   " +
          "  ".join(f"{k}={v}" for k, v in stats.items() if k != "ok"))
    return kept, stats, total_dur


def find_splits(src_dir: Path) -> list[str]:
    return sorted(d.name for d in src_dir.iterdir()
                  if d.is_dir() and (d / "metadata.jsonl").exists())


def process_source(src_name: str, *, min_sec: float, max_sec: float,
                   workers: int, limit: int | None, cap_hours: float, seed: int) -> dict:
    raw_dir = RAW_ASR / src_name
    if not raw_dir.exists():
        print(f"\n=== {src_name} ===  [IGNORÉ] absent : {raw_dir}")
        return {"name": src_name, "status": "absent"}
    splits = find_splits(raw_dir)
    if not splits:
        print(f"\n=== {src_name} ===  [IGNORÉ] aucun split avec metadata.jsonl")
        return {"name": src_name, "status": "vide"}
    print(f"\n=== {src_name} ===  splits={splits}")
    out_dir = PROC_ASR / src_name
    per_split: dict[str, int] = {}
    total_kept = 0
    total_hours = 0.0
    for split in splits:
        kept, _stats, dur = process_split(
            src_name, split, raw_dir / split, out_dir / split,
            min_sec=min_sec, max_sec=max_sec, workers=workers, limit=limit,
            cap_hours=cap_hours, seed=seed)
        per_split[split] = kept
        total_kept += kept
        total_hours += dur / 3600

    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "source": src_name,
        "derived_from": str(raw_dir.relative_to(REPO_ROOT)),
        "target_sr": TARGET_SR,
        "channels": 1,
        "subtype": "PCM_16",
        "min_sec": min_sec,
        "max_sec": max_sec or None,
        "schema": "audiofolder (.wav + metadata.jsonl {file_name, sentence, duration, dataset})",
        "rows": per_split,
        "rows_total": total_kept,
        "hours_total": round(total_hours, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"name": src_name, "status": "ok", "rows": total_kept, "hours": round(total_hours, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="nom(s) de source séparés par des virgules")
    ap.add_argument("--min-sec", type=float, default=1.0, help="durée minimale gardée (défaut 1.0)")
    ap.add_argument("--max-sec", type=float, default=60.0,
                    help="durée maximale gardée, 0 = pas de plafond (défaut 60)")
    ap.add_argument("--workers", type=int, default=min(12, cpu_count()),
                    help=f"processus parallèles (défaut min(12, {cpu_count()}))")
    ap.add_argument("--limit", type=int, default=None, help="n max de clips par split (test)")
    ap.add_argument("--cap-hours", type=float, default=0.0,
                    help="plafond d'heures par split (tirage aléatoire anti-biais), 0 = aucun")
    ap.add_argument("--shuffle-seed", type=int, default=42,
                    help="graine du tirage --cap-hours (défaut 42)")
    args = ap.parse_args()

    if not RAW_ASR.exists():
        raise SystemExit(f"[ERREUR] Dossier introuvable : {RAW_ASR}")

    if args.only:
        sources = [x.strip() for x in args.only.split(",")]
    else:
        sources = sorted(d.name for d in RAW_ASR.iterdir() if d.is_dir())

    results = [
        process_source(s, min_sec=args.min_sec, max_sec=args.max_sec,
                       workers=args.workers, limit=args.limit,
                       cap_hours=args.cap_hours, seed=args.shuffle_seed)
        for s in sources
    ]

    print("\n===== RÉSUMÉ =====")
    for r in results:
        if r["status"] == "ok":
            print(f"  ✅ {r['name']:<32} {r['rows']} clips  ({r['hours']} h)")
        else:
            print(f"  ⚠️  {r['name']:<32} {r['status']}")


if __name__ == "__main__":
    main()
