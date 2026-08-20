#!/bin/bash
set -euo pipefail

# Charge le token depuis .env (non versionné) au lieu de le coder en dur.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

# push_to_hub.py lit HF_TOKEN ; on utilise le token d'ÉCRITURE (upload).
export HF_TOKEN="${HF_TOKEN_WRITE:?HF_TOKEN_WRITE manquant dans .env}"

# Uploader uniquement l'ASR (la traduction est déjà sur HF)
# cleanup_flat_files_on_hf() supprime les anciens train/*.flac engagés à plat
# puis upload_large_folder push les nouvelles références chunk (XET déduplique)
cd "$ROOT"
/home/romaric/.venv/bin/python3 src/push/push_to_hub.py --dataset asr --private