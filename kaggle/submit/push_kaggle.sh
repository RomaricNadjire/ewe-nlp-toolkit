#!/bin/bash
# push_kaggle.sh
# ─────────────────────────────────────────────────────────────────────────────
# Pousse un (ou plusieurs) notebook(s) vers Kaggle, chacun comme un kernel
# distinct exécuté en arrière-plan (GPU + internet activés).
#
# Usage :
#   ./push_kaggle.sh traduction_ewe_fra.ipynb traduction_multilingue.ipynb
#
# Chaque notebook obtient :
#   - un slug dérivé de son nom de fichier (underscores -> tirets)
#   - un dossier kaggle_kernels/<slug>/ avec son kernel-metadata.json
#
# Prérequis :
#   - CLI Kaggle installé + ~/.kaggle/kaggle.json
#   - HF_TOKEN ajouté dans Kaggle (Settings -> Secrets) si besoin
#   - Accélérateur GPU choisi dans l'UF web (T4 x2 recommandé)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./push_kaggle.sh <notebook1.ipynb> [notebook2.ipynb ...]"
    exit 1
fi

# Active le venv (contient le CLI kaggle) si disponible
if [[ -f "$HOME/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.venv/bin/activate"
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Username Kaggle (évite les conflits 409 lors du push)
KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")

# Détection du CLI Kaggle (PATH, module python, ~/.local, venv)
KAGGLE_CMD=""
if command -v kaggle &>/dev/null; then
    KAGGLE_CMD="kaggle"
elif python3 -m kaggle --version &>/dev/null 2>&1; then
    KAGGLE_CMD="python3 -m kaggle"
elif [[ -x "$HOME/.local/bin/kaggle" ]]; then
    KAGGLE_CMD="$HOME/.local/bin/kaggle"
elif [[ -x "$HOME/.venv/bin/kaggle" ]]; then
    KAGGLE_CMD="$HOME/.venv/bin/kaggle"
else
    echo "[ERREUR] kaggle CLI introuvable. Installez-le : pip install kaggle"
    exit 1
fi
echo "CLI Kaggle : $KAGGLE_CMD   |   utilisateur : $KAGGLE_USER"
echo ""

push_one() {
    local notebook="$1"
    local nb_path
    nb_path="$(find "$ROOT/notebooks" -name "$notebook" -print -quit 2>/dev/null)"

    if [[ -z "$nb_path" || ! -f "$nb_path" ]]; then
        echo "[IGNORÉ] introuvable : $notebook"
        return 1
    fi

    local base slug title kernel_dir kernel_id
    base="$(basename "$notebook" .ipynb)"

    # Cherche d'abord un dossier kaggle_kernels/ dont le kernel-metadata.json
    # référence déjà ce notebook (code_file). Cela évite d'écraser un kernel
    # existant dont le slug diffère du nom de fichier (ex. nllb-fr-en vs fine-tuning-nllb-fr-en).
    kernel_dir=""
    while IFS= read -r meta; do
        if python3 -c "
import json, sys
m = json.load(open('$ROOT/' + sys.argv[1]))
sys.exit(0 if m.get('code_file') == '$notebook' else 1)
" "$meta" 2>/dev/null; then
            kernel_dir="$ROOT/$(dirname "$meta")"
            break
        fi
    done < <(find "$ROOT/kaggle/kernels" -name "kernel-metadata.json" -printf "%P\n" 2>/dev/null)

    if [[ -z "$kernel_dir" ]]; then
        # Nouveau notebook : créer un dossier avec un slug dérivé du nom de fichier
        slug="$(echo "$base" | tr '_' '-')"
        title="$(echo "$base" | tr '_' ' ')"
        kernel_id="$KAGGLE_USER/$slug"
        kernel_dir="$ROOT/kaggle/kernels/$slug"
        mkdir -p "$kernel_dir"
        cat > "$kernel_dir/kernel-metadata.json" <<EOF
{
  "id": "$kernel_id",
  "title": "$title",
  "code_file": "$notebook",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
    fi

    cp "$nb_path" "$kernel_dir/$notebook"

    kernel_id=$(python3 -c "import json; print(json.load(open('$kernel_dir/kernel-metadata.json'))['id'])")
    echo "==> Push : $kernel_id"
    $KAGGLE_CMD kernels push -p "$kernel_dir"
    echo "    Suivi : https://www.kaggle.com/code/$kernel_id"
    echo "    Statut: $KAGGLE_CMD kernels status $kernel_id"
    echo ""
}

for nb in "$@"; do
    push_one "$nb" || true
done

echo "Terminé. Les kernels tournent en arrière-plan sur Kaggle."
echo "Pensez à régler l'accélérateur sur T4 x2 dans l'UI web si ce n'est pas déjà fait."
