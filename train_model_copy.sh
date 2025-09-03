#!/bin/bash
set -euo pipefail

IMG="/nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek/gatr_v0.sif"

# Variables que quieres dentro del contenedor
# export SINGULARITYENV_PYTHONUSERBASE="/nfs/cms/arqolmo/GPU_train/mlpf/extlib"
# export SINGULARITYENV_PATH="$SINGULARITYENV_PYTHONUSERBASE/bin:$PATH"
# export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH"
# Evita login interactivo de W&B
# export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"

CMD='
cd /nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek
python modelTrain/PID_GNN/src/torch_model_inference.py
'

# Lanza el contenedor con GPU y bind de rutas necesarias
# -B asegura que /pnfs y tu working dir sean visibles dentro
apptainer exec --nv \
 -B /nfs:/nfs -PWD "$PWD" \
  "$IMG" bash -lc "$CMD"
