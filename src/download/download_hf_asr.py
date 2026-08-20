#!/usr/bin/env python3
"""Télécharge des corpus ASR éwé depuis le Hugging Face Hub (format audiofolder).

Chaque dataset est écrit dans `data/raw/asr/<nom>/<split>/` avec les `.wav` et un
`metadata.jsonl` (`{"file_name": ..., "sentence": ...}`), format identique à
`data/raw/asr/bible_ewe/` (lisible par `load_dataset("audiofolder", ...)`).

Les colonnes audio et transcription sont détectées automatiquement. Un éventuel
filtre de langue (corpus multi-langues) ne garde que l'éwé.

Dépendances :  datasets, soundfile  (pip install datasets soundfile)

⚠️ Volumineux (plusieurs Go par corpus). Utilisez --limit pour tester d'abord.

Reprise automatique (un `progress.json` par split) : le téléchargement reprend là
où il s'est arrêté, ne re-télécharge jamais un clip déjà présent, parallélise le
décodage/écriture (ThreadPool borné), et applique pauses + backoff exponentiel sur
429 / erreurs réseau.

Exemples :
    python src/download/download_hf_asr.py --list
    python src/download/download_hf_asr.py --only ghananlp_health_unicef_ewe --max-workers 4
    python src/download/download_hf_asr.py --only ghananlp_navigation_ewe_speech --sleep 0.5
    python src/download/download_hf_asr.py --only ghana_bible_combined --limit 50
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from _common import RAW_ASR, require, write_manifest

# Colonnes de transcription candidates (minuscules, par priorité de spécificité).
TEXT_PATTERNS = ("sentence", "transcription", "transcript", "normalized_text",
                 "text", "ewe", "translation", "label")
# Colonnes audio candidates.
AUDIO_PATTERNS = ("audio", "wav", "speech", "sound", "file")
# Valeurs acceptées pour un filtre « langue == éwé ».
EWE_LANG_VALUES = {"ewe", "ee", "ewe_latn", "éwé"}

SOURCES = [
    {"name": "paywinful_ewe_asr", "hf_id": "Paywinful/ewe-asr",
     "notes": "⚠️ DÉJÀ PRÉTRAITÉ (colonnes input_features/labels Whisper) — pas d'audio/texte brut, inutilisable tel quel."},
    {"name": "ghananlp_navigation_ewe_speech", "hf_id": "ghananlpcommunity/navigation-corpus-ewe-speech",
     "notes": "Navigation parlée éwé (~49k)."},
    {"name": "ghananlp_ewe_bible_audio", "hf_id": "ghananlpcommunity/ewe-bible-audio-text-tts",
     "notes": "Bible audio + texte éwé (~49k)."},
    {"name": "ghananlp_health_unicef_ewe", "hf_id": "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-ewe",
     "notes": "Santé (UNICEF) ASR éwé."},
    {"name": "ghana_bible_combined", "hf_id": "ghananlpcommunity/ghana-bible-combined-90k-twi-ewe-dagbani",
     "filter_lang": True, "notes": "Bible multi-langues (~90k) — filtrer l'éwé."},
    {"name": "waxal_ewe_single_speaker", "hf_id": "1nnocent/waxal-ewe-tts-filtered-single-speaker",
     "notes": "Waxal éwé TTS mono-locuteur (filtré)."},
]


def _pick(columns, patterns):
    cols = list(columns)
    low = {c: str(c).lower() for c in cols}
    for pat in patterns:                       # correspondance exacte d'abord
        for c in cols:
            if low[c] == pat:
                return c
    for pat in patterns:                       # puis sous-chaîne
        for c in cols:
            if pat in low[c]:
                return c
    return None


def _find_lang_col(columns):
    low = {c: str(c).lower() for c in columns}
    for c in columns:
        if low[c] in ("language", "lang", "lang_id", "locale"):
            return c
    return None


# --------------------------------------------------------------------------- #
#  Reprise (progress.json), throttling réseau, parallélisme décodage/écriture
# --------------------------------------------------------------------------- #
RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc) -> bool:
    """Vrai si l'exception ressemble à un 429 / une erreur réseau transitoire."""
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) in RETRYABLE_CODES:
        return True
    msg = str(exc).lower()
    return any(k in msg for k in (
        "429", "too many requests", "rate limit", "502", "503", "504",
        "timed out", "timeout", "connection", "temporarily",
    ))


def _retry_after(exc) -> float | None:
    """Valeur de l'en-tête HTTP `Retry-After` (secondes) si présente."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    val = headers.get("Retry-After") if headers else None
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _load_progress(split_dir):
    pf = split_dir / "progress.json"
    if pf.exists():
        try:
            return json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def _write_progress(split_dir, n_seen, n_written, done):
    """Écrit progress.json de façon atomique (tmp + os.replace)."""
    pf = split_dir / "progress.json"
    tmp = pf.with_name("progress.json.tmp")
    tmp.write_text(json.dumps({"n_seen": n_seen, "n_written": n_written, "done": done}),
                   encoding="utf-8")
    os.replace(tmp, pf)


def _truncate_metadata(meta_path, keep_lines):
    """Coupe metadata.jsonl à `keep_lines` lignes (cohérence avec progress.json)."""
    if not meta_path.exists():
        return
    with meta_path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    if len(lines) > keep_lines:
        with meta_path.open("w", encoding="utf-8") as f:
            f.writelines(lines[:keep_lines])


def _decode_write(sf, split_dir, fname, audio) -> str:
    """Décode les octets audio et écrit le .wav. Retourne 'ok' | 'skip' | 'fail'.

    Écriture atomique (.part + os.replace) : un crash ne laisse jamais de .wav tronqué.
    """
    path = split_dir / fname
    if path.exists():
        return "skip"
    try:
        if audio.get("bytes") is not None:
            arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        elif audio.get("path"):
            arr, sr = sf.read(audio["path"], dtype="float32")
        elif "array" in audio:                         # déjà décodé (cast échoué)
            arr, sr = audio["array"], audio.get("sampling_rate", 16000)
        else:
            return "fail"
        tmp = path.with_name(fname + ".part")
        sf.write(tmp, arr, sr, format="WAV")           # format explicite (nom .part)
        os.replace(tmp, path)
    except Exception:                                  # noqa: BLE001  (clip illisible)
        return "fail"
    return "ok"


def _run_split(dset, sf, split_dir, meta, *, lang_col, text_col, audio_col,
               seen, written, limit, sleep, max_workers, retries, batch_size) -> int:
    """Parcourt un split par lots, avec reprise, parallélisme et backoff 429.

    `seen` = exemples déjà consommés (offset d'islice) ; `written` = clips déjà écrits.
    Les clips sont nommés par position absolue (`{seen:06d}.wav`) : reprise idempotente.
    Chaque lot est validé atomiquement (metadata + progress) -> jamais de doublon.
    """
    committed_seen = seen
    attempt = 0
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        while True:
            batch: list[tuple[int, str, str, dict]] = []

            def flush() -> None:
                nonlocal written, committed_seen
                if batch:
                    futs = {pool.submit(_decode_write, sf, split_dir, fn, au): (idx, fn, snt)
                            for (idx, fn, snt, au) in batch}
                    done_map: dict[int, tuple[str, str, str]] = {}
                    for fut in as_completed(futs):
                        idx, fn, snt = futs[fut]
                        done_map[idx] = (fn, snt, fut.result())
                    for idx in sorted(done_map):
                        fn, snt, status = done_map[idx]
                        if status in ("ok", "skip"):
                            meta.write(json.dumps({"file_name": fn, "sentence": snt},
                                                  ensure_ascii=False) + "\n")
                            written += 1
                    meta.flush()
                committed_seen = seen
                _write_progress(split_dir, seen, written, done=False)

            try:
                reached_end = True
                iterator = iter(dset)
                if seen:
                    iterator = itertools.islice(iterator, seen, None)
                for ex in iterator:
                    if limit is not None and written >= limit:
                        reached_end = False
                        break
                    seen += 1
                    if lang_col is not None:
                        lv = str(ex.get(lang_col, "")).strip().lower()
                        if lv not in EWE_LANG_VALUES:
                            continue
                    sentence = str(ex.get(text_col) or "").strip()
                    audio = ex.get(audio_col)
                    if not sentence or not isinstance(audio, dict):
                        continue
                    batch.append((seen - 1, f"{seen - 1:06d}.wav", sentence, audio))
                    if len(batch) >= batch_size:
                        flush()
                        batch = []
                        if sleep:
                            time.sleep(sleep)
                flush()
                _write_progress(split_dir, seen, written, done=reached_end)
                return written
            except Exception as exc:                   # noqa: BLE001
                if _is_retryable(exc) and attempt < retries:
                    attempt += 1
                    wait = (_retry_after(exc) or min(60.0, 2.0 ** attempt)) + random.uniform(0, 1.0)
                    print(f"    [429/réseau] tentative {attempt}/{retries}, pause {wait:.1f}s : {exc}")
                    seen = committed_seen              # rembobine au dernier lot validé
                    time.sleep(wait)
                    continue
                raise
    finally:
        pool.shutdown(wait=True)


def process_source(src: dict, limit: int | None, streaming: bool, *,
                   sleep: float = 0.0, max_workers: int = 4, retries: int = 5,
                   resume: bool = True, batch_size: int = 256) -> dict:
    datasets = require("datasets")
    sf = require("soundfile")
    name, hf_id = src["name"], src["hf_id"]
    dest = RAW_ASR / name
    print(f"\n=== {name}  ({hf_id}) ===")
    try:
        ds = datasets.load_dataset(hf_id, streaming=streaming)
    except Exception as e:  # noqa: BLE001
        print(f"  [ÉCHEC] chargement : {e}")
        return {"name": name, "status": "échec", "error": str(e)}

    if not hasattr(ds, "items"):
        ds = {"train": ds}

    split_counts: dict[str, int] = {}
    interrupted = False
    audio_col = text_col = None
    for split, dset in ds.items():
        cols = list(dset.column_names) if dset.column_names else []
        if not cols:                                   # streaming : lire 1 exemple
            cols = list(next(iter(dset)).keys())
        audio_col = _pick(cols, AUDIO_PATTERNS)
        text_col = _pick([c for c in cols if c != audio_col], TEXT_PATTERNS)
        lang_col = _find_lang_col(cols) if src.get("filter_lang") else None
        if not audio_col or not text_col:
            print(f"  [IGNORÉ] split '{split}' : audio/texte introuvable (colonnes={cols})")
            continue

        # datasets>=5 décode l'audio via torchcodec (lourd, exige ffmpeg). On désactive le
        # décodage auto et on lit les octets bruts avec soundfile (WAV/MP3/FLAC/OGG).
        try:
            dset = dset.cast_column(audio_col, datasets.Audio(decode=False))
        except Exception as e:  # noqa: BLE001
            print(f"  [AVERT] cast Audio(decode=False) impossible ({e}) — lecture directe tentée")

        split_dir = dest / split
        split_dir.mkdir(parents=True, exist_ok=True)
        meta_path = split_dir / "metadata.jsonl"

        prog = _load_progress(split_dir) if resume else None
        if prog and prog.get("done"):
            written = int(prog.get("n_written", 0))
            print(f"  {split:<12} déjà complet ({written} clips) — ignoré (--no-resume pour refaire)")
            split_counts[split] = written
            continue
        if prog:                                       # reprise
            seen, written = int(prog.get("n_seen", 0)), int(prog.get("n_written", 0))
            _truncate_metadata(meta_path, written)
            meta_mode = "a"
            print(f"  {split:<12} reprise (seen={seen}, written={written})")
        else:                                          # neuf ou --no-resume
            seen = written = 0
            meta_mode = "w"
            (split_dir / "progress.json").unlink(missing_ok=True)

        try:
            with meta_path.open(meta_mode, encoding="utf-8") as meta:
                written = _run_split(
                    dset, sf, split_dir, meta,
                    lang_col=lang_col, text_col=text_col, audio_col=audio_col,
                    seen=seen, written=written, limit=limit, sleep=sleep,
                    max_workers=max_workers, retries=retries, batch_size=batch_size)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            done_n = int((_load_progress(split_dir) or {}).get("n_written", written))
            print(f"  [INTERROMPU] {split} ({done_n} clips écrits) : {exc}\n"
                  f"               relancez la même commande pour reprendre.")
            split_counts[split] = done_n
            interrupted = True
            break
        split_counts[split] = written
        print(f"  {split:<12} -> {split_dir.relative_to(RAW_ASR.parent.parent)}  ({written} clips)")

    if not split_counts:
        return {"name": name, "status": "échec", "error": "audio/texte introuvable"}

    write_manifest(
        dest,
        source=name,
        provenance="huggingface",
        hf_id=hf_id,
        license="voir carte du dataset",
        format="audiofolder (.wav + metadata.jsonl {file_name, sentence})",
        audio_column=audio_col,
        text_column=text_col,
        lang_filter=bool(src.get("filter_lang")),
        rows=split_counts,
        rows_total=sum(split_counts.values()),
        limited_sample=limit,
        resume=resume,
        notes=src.get("notes", ""),
    )
    status = "interrompu" if interrupted else "ok"
    return {"name": name, "status": status, "rows": sum(split_counts.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", help="nom(s) séparés par des virgules")
    ap.add_argument("--limit", type=int, default=None, help="n max de clips par split (test)")
    ap.add_argument("--streaming", action="store_true", help="itère sans tout télécharger")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="pause (s) entre lots, throttle proactif anti-429")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="threads de décodage/écriture (4-6 conseillé)")
    ap.add_argument("--retries", type=int, default=5,
                    help="tentatives de reprise in-process sur 429/réseau")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="clips par lot (point de validation reprise)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore progress.json et repart de zéro")
    args = ap.parse_args()

    if args.list:
        for s in SOURCES:
            tag = "  [filtre éwé]" if s.get("filter_lang") else ""
            print(f"  {s['name']:<32} {s['hf_id']}{tag}")
        return

    selected = SOURCES
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        selected = [s for s in SOURCES if s["name"] in wanted]
        if not selected:
            raise SystemExit(f"Aucune source ne correspond à --only {args.only!r}")

    results = [
        process_source(s, args.limit, args.streaming,
                       sleep=args.sleep, max_workers=args.max_workers,
                       retries=args.retries, resume=not args.no_resume,
                       batch_size=args.batch_size)
        for s in selected
    ]

    print("\n===== RÉSUMÉ =====")
    marks = {"ok": "✅", "interrompu": "⏸️"}
    for r in results:
        mark = marks.get(r["status"], "❌")
        info = f"{r.get('rows', 0)} clips" if r["status"] in ("ok", "interrompu") else r.get("error", "")
        print(f"  {mark} {r['name']:<32} {info}")


if __name__ == "__main__":
    main()
