#!/usr/bin/env bash
# One-shot install of the `graphretro` conda env for self-contained inference.
#
# Why Python 3.8 (not the upstream environment.yml's 3.7.3)? syntheseus
# requires Python >= 3.8. torch 1.7.0 has Py 3.8 wheels on conda-forge with
# cudatoolkit 10.1, so the model code itself doesn't need porting.
#
# Pass --cpu to install the CPU-only torch build. Default uses cudatoolkit 10.1.

set -euo pipefail
ENV_NAME="${GRAPHRETRO_ENV_NAME:-graphretro}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WANT_CPU="0"
if [[ "${1:-}" == "--cpu" ]]; then
    WANT_CPU="1"
fi

echo "==> Creating conda env: $ENV_NAME (Python 3.8, torch 1.7.0)"

if [[ "$WANT_CPU" == "1" ]]; then
    conda create -n "$ENV_NAME" -y --override-channels \
        -c conda-forge -c pytorch \
        python=3.8 rdkit pytorch=1.7.0 cpuonly "mkl<2024" \
        "numpy<1.24" \
        networkx joblib tqdm pandas pyyaml
else
    conda create -n "$ENV_NAME" -y --override-channels \
        -c conda-forge -c pytorch \
        python=3.8 rdkit pytorch=1.7.0 cudatoolkit=10.1 "mkl<2024" \
        "numpy<1.24" \
        networkx joblib tqdm pandas pyyaml
fi
# Pinning notes:
#   mkl<2024  : conda-forge torch 1.7 builds reference an older MKL symbol
#               (`iJIT_IsProfilingActive`) that newer MKLs no longer expose.
#   numpy<1.24: torch 1.7 ships its own bundled tensorboard which still uses
#               `np.object` (removed in numpy 1.24).

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> Installing syntheseus 0.6.0 (no-deps; layered runtime deps next)"
pip install --no-cache-dir syntheseus==0.6.0 --no-deps
pip install --no-cache-dir \
    omegaconf "hydra-core==1.3.2" \
    more-itertools \
    wandb \
    "tensorboard<2.5" \
    "protobuf<3.20"
# protobuf<3.20 is required by tensorboard 2.4 (the version that ships with torch 1.7).

echo "==> Editable install of seq_graph_retro"
cd "$REPO_DIR"
python setup.py develop

echo "==> Setting SEQ_GRAPH_RETRO + WANDB_MODE=offline in env activate hook"
ACT_DIR="$(conda info --base)/envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACT_DIR"
cat > "$ACT_DIR/seqgr.sh" <<EOF
export SEQ_GRAPH_RETRO="$REPO_DIR"
export WANDB_MODE=offline
EOF

echo
echo "Done. To activate:"
echo "    conda activate $ENV_NAME"
echo
echo "Next: download the un-typed checkpoints from the upstream Drive"
echo "    https://drive.google.com/drive/folders/1u4N6jIsjfA0XxqtKRGAd5N3T-tL8wdKO"
echo "and stage them under:"
echo "    $REPO_DIR/models/SingleEdit_10-02-2021--08-44-37/checkpoints/epoch_156.pt"
echo "    $REPO_DIR/models/LGIndEmbed_18-02-2021--12-23-26/checkpoints/step_101951.pt"
