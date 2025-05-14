#!/bin/bash
OUTPUTDIR=${1}
INPUTDIR=${2}
FILEID=${3}

INPUT_RECO="${INPUTDIR}out_reco_edm4hep_edm4hep_${FILEID}.root"
OUTPUT_TREE="${OUTPUTDIR}tree_${FILEID}.root"
echo "Procesando ${INPUT_RECO} → ${OUTPUT_TREE}"
mkdir -p ${OUTPUTDIR}

source /nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek/setupKey4Hep.sh
# Create Tree
python /nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek/modelTrain/PID_GNN/data_generation/condor/make_pftree_clic_bindings_tautau.py ${INPUT_RECO} ${OUTPUT_TREE} False False 0

