#!/bin/bash

# Uso: ./generar_trabajos.sh <ID_INICIAL> <ID_FINAL>
if [[ $# -ne 2 ]]; then
    echo "Uso: $0 <ID_INICIAL> <ID_FINAL>"
    exit 1
fi

ID_INICIAL=$1
ID_FINAL=$2

# Configuración
OUTPUTDIR="/nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek/TrainTrees/"
INPUTDIR="/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/ZTauTau_SMPol_25Sept_MuonFix/"
INPUTLIST="input_ids.txt"

# Limpiar lista previa
rm -f "$INPUTLIST"

echo "Generando lista de trabajos entre ID $ID_INICIAL y $ID_FINAL..."
for (( ID=ID_INICIAL; ID<=ID_FINAL; ID++ )); do
    OUTFILE="${OUTPUTDIR}/tree_${ID}.root"
    if [[ ! -f "$OUTFILE" ]]; then
        echo "$OUTPUTDIR $INPUTDIR $ID" >> "$INPUTLIST"
    else
        echo "Ya existe $OUTFILE, no se incluye."
    fi
done

if [[ -s "$INPUTLIST" ]]; then
    echo "Lista generada en $INPUTLIST"
else
    echo "Todos los archivos ya existen. Nada que enviar."
fi
