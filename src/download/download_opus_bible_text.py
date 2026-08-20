#!/usr/bin/env python3
"""Telecharge + nettoie la Bible ewe-francais (OPUS bible-uedin, ee-fr) et la
pousse sur le Hub comme reserve de donnees ultra-propres.

CPU uniquement. TEXTE uniquement : aucun fichier audio n'est telecharge ni traite
(garde-fou ASR -> on ne recupere jamais d'audio francais ; le focus acoustique
reste exclusivement l'ewe).

Dependances : requests, huggingface_hub (deja dans requirements.txt).
Auth : exporter HF_TOKEN_WRITE (ou HF_TOKEN). Ex. :  set -a; source ./.env; set +a
"""
from __future__ import annotations
import io, json, os, re, sys, unicodedata, zipfile
from datetime import datetime, timezone
from pathlib import Path

OPUS_API = "https://opus.nlpl.eu/opusapi/"
CORPUS   = "bible-uedin"
SRC, TGT = "ee", "fr"                       # OPUS ordonne ee < fr
CODE     = {"ee": "ewe_Latn", "fr": "fra_Latn"}

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "data" / "raw" / "translation" / "opus_bible_ee_fr"
OUT_JSONL = OUT_DIR / "bible_ee_fr.clean.jsonl"

HF_REPO    = "romaricnadjire/opus-bible-ee-fr"   # depot dedie (reserve)
HF_SUBDIR  = "opus_bible"                          # sous-dossier demande
HF_PRIVATE = True

AUDIO_EXT   = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac")
BIBLE_REF   = re.compile(r"^\s*\d{1,3}[:.]\d{1,3}(?:-\d{1,3})?\s*$")
EWE_CHARS   = set("ŋɖɔɛʋƒ")
_WS         = re.compile(r"\s+")
_BAD_RANGES = ((0x0400,0x04FF),(0x0590,0x05FF),(0x0600,0x06FF),
               (0x3040,0x30FF),(0x4E00,0x9FFF),(0xAC00,0xD7A3))

def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s or "")).replace("\ufeff", "")
    return _WS.sub(" ", s).strip()

def bad_script(s: str) -> bool:
    for ch in s[:200]:
        o = ord(ch)
        for lo, hi in _BAD_RANGES:
            if lo <= o <= hi:
                return True
    return False

def keep(ewe: str, fra: str) -> bool:
    if not ewe or not fra:                            return False   # vide
    if ewe == fra:                                    return False   # copie
    if BIBLE_REF.match(ewe) or BIBLE_REF.match(fra):  return False   # ref seule
    if bad_script(ewe) or bad_script(fra):            return False   # ecriture etrangere
    if sum(c in EWE_CHARS for c in fra) >= 2:         return False   # fuite ewe dans fr
    ratio = len(ewe) / max(len(fra), 1)
    if ratio < 0.2 or ratio > 5.0:                    return False   # ratio aberrant
    return True

def resolve_url() -> str:
    import requests
    p = {"corpus": CORPUS, "source": SRC, "target": TGT,
         "preprocessing": "moses", "version": "latest"}
    data = requests.get(OPUS_API, params=p, timeout=60).json()
    for e in data.get("corpora", []):
        if e.get("url", "").endswith(".zip"):
            return e["url"]
    sys.exit("[ERREUR] Archive moses introuvable via l'API OPUS.")

def download_clean() -> list[dict]:
    import requests
    url = resolve_url()
    print(f"Archive OPUS : {url}")
    zf = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=600).content))

    # GARDE-FOU ASR : on ignore explicitement tout fichier audio eventuel.
    audio = [n for n in zf.namelist() if n.lower().endswith(AUDIO_EXT)]
    if audio:
        print(f"[ASR] {len(audio)} fichier(s) audio IGNORE(s) : {audio}")
    else:
        print("[ASR] aucun audio dans l'archive (texte uniquement) - OK.")

    f_ee = next((n for n in zf.namelist() if n.endswith(f".{SRC}")), None)
    f_fr = next((n for n in zf.namelist() if n.endswith(f".{TGT}")), None)
    if not (f_ee and f_fr):
        sys.exit(f"[ERREUR] Fichiers texte .{SRC}/.{TGT} absents : {zf.namelist()}")
    print(f"Texte ewe : {f_ee}\nTexte fra : {f_fr}")

    with zf.open(f_ee) as a, zf.open(f_fr) as b:
        ee_lines = io.TextIOWrapper(a, encoding="utf-8")
        fr_lines = io.TextIOWrapper(b, encoding="utf-8")
        seen, recs, n_raw = set(), [], 0
        for ee, fr in zip(ee_lines, fr_lines):
            n_raw += 1
            ee, fr = norm(ee), norm(fr)
            if not keep(ee, fr):
                continue
            key = "\u241F".join(sorted((ee.lower(), fr.lower())))   # dedup canonique
            if key in seen:
                continue
            seen.add(key)
            recs.append({"translation": {CODE[SRC]: ee, CODE[TGT]: fr}})
    print(f"Brut : {n_raw}  ->  propre & dedupique : {len(recs)}")
    return recs

def write_local(recs: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "source": "opus_bible_ee_fr", "provenance": "opus", "corpus": CORPUS,
        "pair": "ee-fr", "license": "domaine public / conditions OPUS",
        "format": "jsonl {'translation': {'ewe_Latn','fra_Latn'}}",
        "rows": len(recs), "audio": "aucun (texte uniquement)",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Ecrit : {OUT_JSONL.relative_to(REPO_ROOT)}  (+ manifest.json)")

def push(recs: list[dict]) -> None:
    token = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN")
    if not token:
        print("[Hub] HF_TOKEN_WRITE absent -> push ignore. "
              "Faites : set -a; source ./.env; set +a")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(HF_REPO, repo_type="dataset", private=HF_PRIVATE, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(OUT_JSONL),
        path_in_repo=f"{HF_SUBDIR}/bible_ee_fr.jsonl",
        repo_id=HF_REPO, repo_type="dataset",
        commit_message=f"Bible ewe-fr (OPUS bible-uedin) nettoyee : {len(recs)} paires",
    )
    print(f"[Hub] Pousse : datasets/{HF_REPO} -> {HF_SUBDIR}/bible_ee_fr.jsonl")

if __name__ == "__main__":
    records = download_clean()
    write_local(records)
    push(records)
    print("Termine.")
