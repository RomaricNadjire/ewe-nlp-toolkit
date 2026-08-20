#!/usr/bin/env python3
"""Inventaire (lecture seule) des durées des corpus ASR éwé.

Parcourt `data/raw/asr/<dataset>/<split>/` (récursivement, capte les sous-dossiers
`chunk_*`), sonde chaque fichier audio via `soundfile.info` (lecture d'en-tête,
rapide, ne décode pas le signal) et agrège par dataset/split :

    nombre de clips · durée totale (h) · min / médiane / max · part < 1 s et > 5 s ·
    samplerates · nombre de canaux · formats · clips illisibles.

But : dimensionner les plafonds d'équilibrage de la fusion (Phase 4) et vérifier
les hypothèses (Bible ~86 h voix unique, UNICEF segments > 5 s, etc.).
Aucune écriture audio.

Dépendances : soundfile

Exemples :
    .venv/bin/python src/prepare/inventory_asr.py
    .venv/bin/python src/prepare/inventory_asr.py --only ghananlp_health_unicef_ewe,bible_ewe
    .venv/bin/python src/prepare/inventory_asr.py --save data/interim/asr/inventory.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ASR = REPO_ROOT / "data" / "raw" / "asr"
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac"}


def _require_soundfile():
    try:
        import soundfile as sf  # noqa: PLC0415
        return sf
    except ImportError:
        raise SystemExit(
            "[ERREUR] Module 'soundfile' introuvable.\n"
            "         Installez-le :  .venv/bin/pip install soundfile"
        )


def split_of(dataset_dir: Path, path: Path) -> str:
    """Nom du split = 1er composant du chemin relatif (sinon « (racine) »)."""
    rel = path.relative_to(dataset_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "(racine)"


def collect_tasks(dataset_dir: Path) -> list[tuple[str, Path]]:
    """Liste (split, chemin_audio) pour tous les fichiers audio du dataset."""
    tasks: list[tuple[str, Path]] = []
    for p in dataset_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            tasks.append((split_of(dataset_dir, p), p))
    return tasks


def probe(sf, path: Path) -> dict:
    """Métadonnées d'en-tête d'un fichier audio (durée en secondes, sr, canaux, format)."""
    try:
        info = sf.info(str(path))
        dur = float(info.frames) / info.samplerate if info.samplerate else 0.0
        return {"duration": dur, "samplerate": int(info.samplerate),
                "channels": int(info.channels), "format": str(info.format), "ok": True}
    except Exception as exc:  # noqa: BLE001  (clip illisible/corrompu)
        return {"duration": 0.0, "samplerate": 0, "channels": 0,
                "format": "", "ok": False, "error": str(exc)}


def count_declared(dataset_dir: Path) -> int:
    """Somme des lignes de tous les `metadata.jsonl` (clips déclarés)."""
    total = 0
    for meta in dataset_dir.rglob("metadata.jsonl"):
        try:
            with meta.open("r", encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
        except OSError:
            pass
    return total


def summarise(durations: list[float], samplerates, channels, formats,
              n_err: int) -> dict:
    if durations:
        durations_sorted = sorted(durations)
        total = sum(durations_sorted)
        stats = {
            "clips": len(durations_sorted),
            "total_sec": total,
            "total_hours": total / 3600.0,
            "min_sec": durations_sorted[0],
            "median_sec": statistics.median(durations_sorted),
            "max_sec": durations_sorted[-1],
            "n_lt_1s": sum(1 for d in durations_sorted if d < 1.0),
            "n_gt_5s": sum(1 for d in durations_sorted if d > 5.0),
        }
    else:
        stats = {"clips": 0, "total_sec": 0.0, "total_hours": 0.0, "min_sec": 0.0,
                 "median_sec": 0.0, "max_sec": 0.0, "n_lt_1s": 0, "n_gt_5s": 0}
    stats["samplerates"] = sorted(samplerates)
    stats["channels"] = sorted(channels)
    stats["formats"] = sorted(formats)
    stats["n_unreadable"] = n_err
    return stats


def inventory(datasets: list[Path], workers: int) -> dict:
    sf = _require_soundfile()
    result: dict = {"datasets": {}, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    grand_durations: list[float] = []

    for dataset_dir in datasets:
        name = dataset_dir.name
        tasks = collect_tasks(dataset_dir)
        declared = count_declared(dataset_dir)
        print(f"\n=== {name} ===  ({len(tasks)} fichiers audio, {declared} déclarés dans metadata)")
        if not tasks:
            result["datasets"][name] = {"splits": {}, "declared": declared, "audio_files": 0}
            continue

        with ThreadPoolExecutor(max_workers=workers) as ex:
            probes = list(ex.map(lambda t: probe(sf, t[1]), tasks))

        per_split: dict[str, dict] = {}
        buckets: dict[str, dict] = {}
        for (split, _path), info in zip(tasks, probes):
            b = buckets.setdefault(split, {"dur": [], "sr": set(), "ch": set(), "fmt": set(), "err": 0})
            if info["ok"]:
                b["dur"].append(info["duration"])
                b["sr"].add(info["samplerate"])
                b["ch"].add(info["channels"])
                b["fmt"].add(info["format"])
                grand_durations.append(info["duration"])
            else:
                b["err"] += 1

        for split, b in sorted(buckets.items()):
            s = summarise(b["dur"], b["sr"], b["ch"], b["fmt"], b["err"])
            per_split[split] = s
            sr_str = "/".join(str(x) for x in s["samplerates"]) or "?"
            ch_str = "/".join(str(x) for x in s["channels"]) or "?"
            print(f"  {split:<12} {s['clips']:>7} clips  {s['total_hours']:>7.2f} h   "
                  f"min {s['min_sec']:>5.1f}s  méd {s['median_sec']:>5.1f}s  max {s['max_sec']:>6.1f}s   "
                  f">5s={s['n_gt_5s']:<6} <1s={s['n_lt_1s']:<5} sr={sr_str} ch={ch_str} fmt={','.join(s['formats'])}"
                  + (f"  [illisibles={s['n_unreadable']}]" if s['n_unreadable'] else ""))

        ds_hours = sum(v["total_hours"] for v in per_split.values())
        ds_clips = sum(v["clips"] for v in per_split.values())
        print(f"  {'TOTAL':<12} {ds_clips:>7} clips  {ds_hours:>7.2f} h")
        result["datasets"][name] = {"splits": per_split, "declared": declared,
                                    "audio_files": len(tasks),
                                    "total_hours": ds_hours, "total_clips": ds_clips}

    total_hours = sum(grand_durations) / 3600.0
    result["grand_total"] = {"clips": len(grand_durations), "hours": total_hours}
    print("\n===== TOTAL GÉNÉRAL =====")
    print(f"  {len(grand_durations)} clips lisibles  ·  {total_hours:.2f} h")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="nom(s) de dataset séparés par des virgules")
    ap.add_argument("--workers", type=int, default=8, help="threads de sondage (défaut 8)")
    ap.add_argument("--save", metavar="PATH", help="écrit le rapport JSON à ce chemin")
    args = ap.parse_args()

    if not RAW_ASR.exists():
        raise SystemExit(f"[ERREUR] Dossier introuvable : {RAW_ASR}")

    datasets = sorted(d for d in RAW_ASR.iterdir() if d.is_dir())
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        datasets = [d for d in datasets if d.name in wanted]
        if not datasets:
            raise SystemExit(f"Aucun dataset ne correspond à --only {args.only!r}")

    report = inventory(datasets, args.workers)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nRapport JSON écrit : {out}")


if __name__ == "__main__":
    main()
