#!/bin/bash
# submit_kaggle.sh
# ─────────────────────────────────────────────────────────────────────────────
# Pousse le script d'entraînement vers Kaggle et lance le kernel en arrière-plan.
#
# Prérequis :
#   1. Installer le CLI Kaggle :  pip install kaggle
#   2. Placer vos identifiants dans ~/.kaggle/kaggle.json
#      (téléchargeable sur https://www.kaggle.com/settings → API → Create New Token)
#   3. Ajouter votre HF_TOKEN dans Kaggle :
#      https://www.kaggle.com/settings → Secrets → Add secret "HF_TOKEN"
#
# Usage :
#   chmod +x submit_kaggle.sh
#   ./submit_kaggle.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Activer le venv si disponible (contient kaggle, pip, etc.)
if [[ -f "$HOME/.venv/bin/activate" ]]; then
    source "$HOME/.venv/bin/activate"
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KERNEL_DIR="$ROOT/kaggle/kernels/nllb-ewe-fine-tuning"
# Récupère le username depuis kaggle.json pour éviter les conflits 409
KAGGLE_USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])" 2>/dev/null || echo "votre-username")
KERNEL_SLUG="nllb-ewe-fine-tuning"
KERNEL_ID="$KAGGLE_USER/$KERNEL_SLUG"

echo "==> Vérification du CLI Kaggle…"
# Cherche kaggle dans le PATH, puis dans les emplacements pip courants
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
    echo "         Puis relancez en ajoutant ~/.local/bin au PATH :"
    echo "         export PATH=\$PATH:\$HOME/.local/bin"
    exit 1
fi
echo "  CLI trouvé : $KAGGLE_CMD"

# Copie le notebook dans le dossier kernel (Kaggle exige code_file dans le même dossier)
NOTEBOOK_SRC="$ROOT/notebooks/translation/traduction.ipynb"
cp "$NOTEBOOK_SRC" "$KERNEL_DIR/traduction.ipynb"
echo "  Notebook copié dans $KERNEL_DIR/"

# Met à jour l'id dans kernel-metadata.json avec le bon username
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
echo "Kernel soumis et en cours d'exécution en arrière-plan sur Kaggle."
echo ""
echo "Commandes utiles :"
echo "  Statut    : $KAGGLE_CMD kernels status $KERNEL_ID"
  echo "  Logs      : $KAGGLE_CMD kernels output $KERNEL_ID"
echo "  Annuler   : (via l'interface web Kaggle)"
echo ""
echo "URL de suivi : https://www.kaggle.com/code/$KERNEL_ID"
