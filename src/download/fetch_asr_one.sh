#!/usr/bin/env bash
# Téléchargement ASR pas-à-pas : UN dataset à la fois, en streaming (cache minimal).
# Pensé pour un disque limité — workflow « un dataset par jour ».
#
# Le mode streaming évite de copier le parquet complet dans ~/.cache/huggingface/hub
# (sinon il faut ~2x la taille : cache + .wav). Seuls les .wav de sortie occupent
# l'espace, écrits au fur et à mesure dans data/raw/asr/<nom>/.
#
# Usage :
#   src/download/fetch_asr_one.sh                 # liste les datasets disponibles
#   src/download/fetch_asr_one.sh <nom> [limit]   # télécharge ce dataset (complet, ou 'limit' clips)
#
# Exemples :
#   src/download/fetch_asr_one.sh ghananlp_navigation_ewe_speech
#   src/download/fetch_asr_one.sh ghana_bible_combined 5000
set -euo pipefail

cd "$(dirname "$0")/../.."                 # racine du dépôt
PY=.venv/bin/python
SEUIL_GO=5                                 # avertir si moins de X Go libres

if [[ $# -eq 0 ]]; then
  echo "Datasets ASR disponibles :"
  "$PY" src/download/download_hf_asr.py --list
  echo
  echo "Usage : $0 <nom> [limit]"
  exit 0
fi

NAME="$1"; LIMIT="${2:-}"
LIMIT_ARG=()
[[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")

libre() { df --output=avail -BG . | tail -1 | tr -dc '0-9'; }

AVANT=$(libre)
echo "Espace libre avant : ${AVANT} Go"
if [[ "$AVANT" -lt "$SEUIL_GO" ]]; then
  echo "⚠️  Moins de ${SEUIL_GO} Go libres — libère de l'espace avant de continuer." >&2
  exit 1
fi

echo "→ Téléchargement (streaming) de '$NAME'..."
"$PY" -u src/download/download_hf_asr.py --only "$NAME" --streaming "${LIMIT_ARG[@]}"

APRES=$(libre)
echo
echo "Espace libre après : ${APRES} Go  (Δ $((AVANT - APRES)) Go utilisés)"
echo "Sortie : data/raw/asr/$NAME/"
du -sh "data/raw/asr/$NAME" 2>/dev/null || true
if [[ "$APRES" -lt "$SEUIL_GO" ]]; then
  echo "⚠️  Disque presque plein — déplace ce dataset vers un stockage externe avant le suivant." >&2
fi
