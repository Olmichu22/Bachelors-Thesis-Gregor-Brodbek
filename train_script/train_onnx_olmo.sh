#!/usr/bin/env bash

CLUSTER_ID=$1
PROCESS_ID=$2

BASE_DIR="/nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek"
SCRIPT_DIR="$BASE_DIR/train_script"
TRAIN_DIR="$BASE_DIR/TrainTrees"
OUTPUT_DIR="$BASE_DIR/TestTrain_${CLUSTER_ID}"
CONFIG_DIR="$BASE_DIR/modelTrain/PID_GNN/config_files"
CODE_DIR="$BASE_DIR/modelTrain/PID_GNN"
SIF_PATH="$BASE_DIR/gatr_v0.sif"


mkdir -p ${OUTPUT_DIR}


singularity exec --nv \
-B ${TRAIN_DIR}:/mnt/TrainTrees \
-B ${OUTPUT_DIR}:/mnt/TestTrain \
-B ${CONFIG_DIR}:/mnt/config_files \
-B ${CODE_DIR}:/mnt/PID_GNN  \
${SIF_PATH} \
bash -c "cd /mnt/PID_GNN && \
export WANDB_API_KEY=\$WANDB_API_KEY && \
wandb login --relogin && \
python3 -m src.train_lightning1 \
  --data-train 'train:/mnt/TrainTrees/tree_*.root' \
  --data-config '/mnt/config_files/config_hit_tracks_tau.yaml' \
  -clust -clust_dim 3 \
  --network-config 'src/models/wrapper/example_mode_gatr_e.py' \
  --model-prefix '/mnt/TestTrain/' \
  --num-workers 0 \
  --gpus 1 \
  --batch-size 15 \
  --start-lr 1e-3 \
  --num-epochs 40 \
  --fetch-step 0.1 \
  --log-wandb \
  --wandb-displayname Olmo_TrainRun_\${CLUSTER_ID} \
  --wandb-projectname topas_logs \
  --wandb-entity 'olmo-arquero'"



