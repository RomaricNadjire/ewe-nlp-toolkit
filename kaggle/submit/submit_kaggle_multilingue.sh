#!/bin/bash
# submit_kaggle_multilingue.sh
# ─────────────────────────────────────────────────────────────────────────────
# Pousse le notebook multilingue (éwé/en/fr) vers Kaggle.
# Équivalent de submit_kaggle.sh, mais pour fine_tuning_nllb_multilingue.ipynb.
#
# Prérequis :
#   1. CLI Kaggle :  pip install kaggle
#   2. Identifiants dans ~/.kaggle/kaggle.json
#      (https://www.kaggle.com/settings → API → Create New Token)
#   3. Secrets Kaggle (https://www.kaggle.com/settings → Secrets) :
#        - HF_TOKEN_READ   : token READ  (accès aux datasets privés)
#        - HF_TOKEN_WRITE  : token WRITE (push de l'adaptateur)
#      Cocher "Attach to notebook" pour chacun.
#
# Usage :
#   chmod +x submit_kaggle_multilingue.sh
#   ./submit_kaggle_multilingue.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Active le venv si disponible (contient kaggle CLI)
if [[ -f "$HOME/.venv/bin/activate" ]]; then
    source "$HOME/.venv/bin/activate"
fi

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
KERNEL_DIR="$PROJECT_DIR/kaggle/kernels/fine-tuning-nllb-multilingue"
NOTEBOOK_SRC="$PROJECT_DIR/notebooks/translation/fine_tuning_nllb_multilingue.ipynb"
KERNEL_SLUG="fine-tuning-nllb-multilingue"

# Username depuis kaggle.json (évite les conflits 409)
KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])" 2>/dev/null || echo "votre-username")
KERNEL_ID="$KAGGLE_USER/$KERNEL_SLUG"

echo "==> Vérification du CLI Kaggle…"
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
echo "  CLI trouvé : $KAGGLE_CMD"

# Synchronise la dernière version du notebook dans le dossier kernel
# (Kaggle exige que code_file soit dans le même dossier que kernel-metadata.json)
cp "$NOTEBOOK_SRC" "$KERNEL_DIR/fine_tuning_nllb_multilingue.ipynb"
echo "  Notebook copié dans $KERNEL_DIR/"

# Met à jour l'id avec le bon username
python3 - <<EOF
import json, pathlib
meta_path = pathlib.Path("$KERNEL_DIR/kernel-metadata.json")
meta = json.loads(meta_path.read_text())
meta["id"] = "$KERNEL_ID"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"  kernel-metadata.json mis à jour : id = {meta['id']}")
EOF

echo "==> Push du kernel vers Kaggle…"
$KAGGLE_CMD kernels push -p "$KERNEL_DIR"

echo ""
echo "Kernel soumis sur Kaggle."
echo ""
echo "Commandes utiles :"
echo "  Statut : $KAGGLE_CMD kernels status $KERNEL_ID"
echo "  Logs   : $KAGGLE_CMD kernels output $KERNEL_ID"
echo ""
echo "URL de suivi : https://www.kaggle.com/code/$KERNEL_ID"
