import pandas as pd
import numpy as np
import logging
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------

PI0 = "π⁰"
PI = "π"
MU = "μ"
E = "e"
N = "n"
NEUTRINO = "ν"
TAU = "τ"
GAMMA = "γ"
BHABHA = "Bhabha"
QQ = "qq"

# --- helpers de filtrado (añádelos cerca de los imports) ---
def _keys_as_present(df: pd.DataFrame, keys):
    """Devuelve la lista de keys presentes en el índice/columnas del df,
    conservando el orden dado por 'keys'."""
    if not keys:
        return list(df.index), list(df.columns)
    keys = [int(k) for k in keys]
    idx = [k for k in keys if k in df.index]
    cols = [k for k in keys if k in df.columns]
    return idx, cols

def filter_cm_counts(cm_df: pd.DataFrame, keys):
    """Filtra cuentas (no normalizadas)."""
    if not keys:
        return cm_df
    idx, cols = _keys_as_present(cm_df, keys)
    return cm_df.loc[idx, cols]

def filter_cm_after_norm(cm_norm_df: pd.DataFrame, keys):
    """
    Filtra la CM ya normalizada por filas (hecha ANTES del filtrado),
    para evitar porcentajes ficticios. No re-normaliza.
    """
    if not keys:
        return cm_norm_df
    idx, cols = _keys_as_present(cm_norm_df, keys)
    return cm_norm_df.loc[idx, cols]


def id_to_key(event_id, photons=False):
    if photons:
        if event_id < 0:
            if event_id == -13:
                key = f"{MU}"
            elif event_id == -11:
                key = f"{E}"
            elif event_id <= -20:
                key = f"h{N}"
            elif event_id == -2:
                key = "Unmatched"
            elif event_id == -1:
                key = "Unknown"
            else:
                key = "Unknown ID"  
        elif event_id == 0:
            key = f"h"
        elif event_id == 1:
            key = f"h{GAMMA}"
        elif event_id < 10:
            key = f"h{event_id}{GAMMA}"
        elif event_id == 10:
            key = f"3h"
        else:
            key = f"3h{event_id-10}{GAMMA}"

    else:
        if event_id < 0:
            if event_id == -13:
                key = f"{TAU} → {MU}2{NEUTRINO}"
            elif event_id == -11:
                key = f"{TAU} → {E}2{NEUTRINO}"
            elif event_id == -20:
                key = f"{PI}{N}"
            elif event_id == -333:
                key = BHABHA
            elif event_id == -222:
                key = QQ
            elif event_id == -2:
                key = "Unmatched"
            elif event_id == -1:
                key = "Unknown"
            else:
                key = "Unknown ID" 
        elif event_id == 0:
            key = f"{TAU} → {PI}{NEUTRINO}"
        elif event_id == 1:
            key = f"{TAU} → {PI}{PI0}{NEUTRINO}"
        elif event_id < 10:
            key = f"{TAU} → {PI}{event_id}{PI0}{NEUTRINO}"
        elif event_id == 10:
            key = f"{TAU} → 3{PI}{NEUTRINO}"
        elif event_id == 11:
            key = f"{TAU} → 3{PI}{PI0}{NEUTRINO}"
        else:
            key = f"{TAU} → {3}{PI}{event_id-10}{PI0}{NEUTRINO}"
    return key

def incorrectCM(result_labels, failed_pred_ids):
    """Create a confusion matrix with the predictions of the other model when the current model fails"""
    gentauids = list(set(result_labels["GenTauID"].tolist()))
    recotausids = list(set(result_labels["RecoTauID"].tolist()))
    # Remove -999 from the list of taus
    gentauids = [tau for tau in gentauids if tau != -999]
    recotausids = [tau for tau in recotausids if tau != -999]
    gentauids.sort()
    recotausids.sort()
    # Assing a number to each tau
    gentauids_dict = {tau: i for i, tau in enumerate(gentauids)}
    recotausids_dict = {tau: i for i, tau in enumerate(recotausids)}

    incorrect_cm = np.zeros((len(gentauids), len(recotausids)))

    for ide, gentauid in failed_pred_ids.items():
        # Locate result_labels row with the same id
        row = result_labels.loc[ide]
        if row["RecoTauID"] == -999:
            continue
        if row["GenTauID"] != gentauid:
            print(f"Error: {gentauid} != {row['GenTauID']}")
            continue
        incorrect_cm[gentauids_dict[gentauid], recotausids_dict[row["RecoTauID"]]] += 1

    return pd.DataFrame(incorrect_cm, index=gentauids, columns=recotausids)


def event_to_tau(result_labels, model="GATr", outputpath=None):
    """Convert the result labels to a list of events"""
    tau_preds = {"id": [], "GenTauID": [], "RecoTauID": []}
    for i, row in result_labels.iterrows():
        tau_preds["id"].append(str(i) + "0")
        tau_preds["GenTauID"].append(row["tau1"])
        tau_preds["RecoTauID"].append(row["id-tau1"])
        tau_preds["id"].append(str(i) + "1")
        tau_preds["GenTauID"].append(row["tau2"])
        tau_preds["RecoTauID"].append(row["id-tau2"])
    tau_preds_df = pd.DataFrame(tau_preds)
    if outputpath is not None:
        tau_preds_df.to_csv(
            os.path.join(outputpath, f"result_labels_per_tau_{model}.csv"), index=False
        )
    return tau_preds_df


def evaluate_result_labels(
    result_labels,
    summary_evaluation_results,
    logger,
    output_path,
    algorithm="GNN",
):
    logger.info(f"Evaluating {algorithm} predictions...")
    gentauids = list(set(result_labels["GenTauID"].tolist()))
    recotausids = list(set(result_labels["RecoTauID"].tolist()))
    # Remove -999 from the list of taus
    gentauids = [tau for tau in gentauids if tau != -999]
    recotausids = [tau for tau in recotausids if tau != -999]
    gentauids.sort()
    recotausids.sort()
    # Assing a number to each tau
    gentauids_dict = {tau: i for i, tau in enumerate(gentauids)}
    recotausids_dict = {tau: i for i, tau in enumerate(recotausids)}
    # Generate CM Matrix
    cm_matrix = np.zeros((len(gentauids), len(recotausids)))
    failed_pred_ids = {}
    for id, row in result_labels.iterrows():
        if row["GenTauID"] == -999 or row["RecoTauID"] == -999:
            continue
        cm_matrix[
            gentauids_dict[row["GenTauID"]], recotausids_dict[row["RecoTauID"]]
        ] += 1
        if row["GenTauID"] != row["RecoTauID"]:
            failed_pred_ids[id] = row["GenTauID"]

    # generate evaluation results
    # Check purity and completeness for each gen tau
    total_correct_pred = 0
    for gentauid in gentauids:
        row = cm_matrix[gentauids_dict[gentauid], :]
        if gentauid not in recotausids_dict:
            logger.warning(f"GenTauID {gentauid} not found in RecoTauID")
            continue
        column = cm_matrix[:, recotausids_dict[gentauid]]
        tp = cm_matrix[gentauids_dict[gentauid], recotausids_dict[gentauid]]
        total_correct_pred += tp

        purity = tp / np.sum(column) if np.sum(column) > 0 else 0
        completeness = tp / np.sum(row) if np.sum(row) > 0 else 0

        summary_evaluation_results["Purity"][gentauid] = round(float(purity), 4)
        summary_evaluation_results["Completeness"][gentauid] = round(
            float(completeness), 4
        )

    summary_evaluation_results["total_correct_preds"] = float(total_correct_pred)
    accuracy = (
        summary_evaluation_results["total_correct_preds"] / np.sum(cm_matrix)
        if np.sum(cm_matrix) > 0
        else 0
    )
    summary_evaluation_results["total_correct_preds_fraction"] = round(
        float(accuracy), 4
    )

    logger.info(
        f"Total correct predictions: {summary_evaluation_results['total_correct_preds']}"
    )
    logger.info(
        f"Total correct predictions fraction: {summary_evaluation_results['total_correct_preds_fraction']}"
    )
    logger.info(f"Saving evaluation results to {output_path}")
    # Save the evaluation results to a file
    with open(
        os.path.join(output_path, f"evaluation_results_{algorithm}.yaml"), "w"
    ) as f:
        yaml.dump(summary_evaluation_results, f)
    # save cm matrix
    cm_matrix_df = pd.DataFrame(cm_matrix, index=gentauids, columns=recotausids)
    cm_matrix_df.to_csv(os.path.join(output_path, f"cm_matrix_{algorithm}.csv"))

    return cm_matrix_df, failed_pred_ids


def crossPseudoCMs(
    result_labels,
    evaluation_results,
    draw_path=None,
    algorithms=["GATr", "PFO"],
    keys=None,
):
    """Create a pseudo confusion matrix for the predictions"""
    real_taus = list(
        set(result_labels[0]["tau1"].tolist() + result_labels[0]["tau2"].tolist())
    )
    pred_taus = [
        list(
            set(
                result_labels[0]["id-tau1"].tolist()
                + result_labels[0]["id-tau2"].tolist()
            )
        ),
        list(
            set(
                result_labels[1]["id-tau1"].tolist()
                + result_labels[1]["id-tau2"].tolist()
            )
        ),
    ]

    real_taus.sort()
    pred_taus = [sorted(tau) for tau in pred_taus]
    # Remove -999 from the list of taus
    real_taus = [tau for tau in real_taus if tau != -999]
    pred_taus[0] = [tau for tau in pred_taus[0] if tau != -999]
    pred_taus[1] = [tau for tau in pred_taus[1] if tau != -999]

    # Assing a number to each tau
    real_taus_dict = {tau: i for i, tau in enumerate(real_taus)}
    pred_taus_dict = [
        {tau: i for i, tau in enumerate(pred_taus[0])},
        {tau: i for i, tau in enumerate(pred_taus[1])},
    ]

    # Confusion matrix with only incorrect predictions
    cm_incorrect = [
        np.zeros((len(real_taus), len(pred_taus[1]))),
        np.zeros((len(real_taus), len(pred_taus[0]))),
    ]

    for i in range(2):
        for j, row in evaluation_results[i].iterrows():
            if row["num_correct_preds"] < 2:
                for fail_tau in row["incorrect_preds_real_ids"]:
                    if fail_tau == -999:
                        continue
                    for pred_tau in [
                        result_labels[1 - i]["id-tau1"][j],
                        result_labels[1 - i]["id-tau2"][j],
                    ]:
                        if pred_tau == -999:
                            continue
                        cm_incorrect[i][
                            real_taus_dict[fail_tau], pred_taus_dict[1 - i][pred_tau]
                        ] += 0.5

    # if keys is not None:
    #     real_pos_keys = [real_taus_dict[tau] for tau in keys]
    #     pred_pos_keys_0 = [pred_taus_dict[0][tau] for tau in keys]
    #     pred_pos_keys_1 = [pred_taus_dict[1][tau] for tau in keys]
    #     # Filter the confusion matrix by keys
    #     cm_incorrect[0] = cm_incorrect[0][real_pos_keys, pred_pos_keys_1]
    #     cm_incorrect[1] = cm_incorrect[1][real_pos_keys, pred_pos_keys_1]
    cm_incorrect_df = [
        pd.DataFrame(cm_incorrect[i], index=real_taus, columns=pred_taus[1 - i])
        for i in range(2)
    ]

    cm_incorrect_norm = []
    for i in range(2):
        # Normalize the confusion matrix by row (i.e. by the number of samples)
        cm_incorrect_norm.append(
            cm_incorrect[i].astype("float") / cm_incorrect[i].sum(axis=1)[:, np.newaxis]
        )
        cm_incorrect_norm[-1][np.isnan(cm_incorrect_norm[-1])] = 0

    cm_incorrect_norm_df = [
        pd.DataFrame(cm_incorrect_norm[i], index=real_taus, columns=pred_taus[1 - i])
        for i in range(2)
    ]

    if draw_path is not None:
        real_tick_marks = np.arange(len(real_taus))
        pred_tick_marks = [np.arange(len(pred_taus[1])), np.arange(len(pred_taus[0]))]
        for alg in range(2):
            plt.figure(figsize=(12, 12))
            plt.imshow(cm_incorrect[alg], interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(
                f"Predictions of {algorithms[1-alg]} when {algorithms[alg]} fails",
                fontsize=20,
            )

            plt.xticks(pred_tick_marks[alg], pred_taus[1 - alg], rotation=45)
            plt.yticks(real_tick_marks, real_taus)

            # Añadir los valores dentro de cada celda
            # Añadir los valores dentro de cada celda
            for i in range(len(real_taus)):
                for j in range(len(pred_taus[1 - alg])):
                    plt.text(
                        j,
                        i,
                        f"{cm_incorrect[alg][i, j]:.1f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if cm_incorrect[alg][i, j] > cm_incorrect[alg].max() / 2
                            else "black"
                        ),
                    )

            plt.ylabel("Desintegración simulada", fontsize=15)
            plt.xlabel("Desintegración identificada", fontsize=15)
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    draw_path,
                    f"cm_incorrect_{algorithms[alg]}_vs_{algorithms[1-alg]}.png",
                )
            )
            plt.close()

            plt.figure(figsize=(12, 12))
            plt.imshow(
                cm_incorrect_norm[alg], interpolation="nearest", cmap=plt.cm.Blues
            )
            plt.title(
                f"Predictions of {algorithms[1-alg]} when {algorithms[alg]} fails",
                fontsize=20,
            )

            plt.xticks(pred_tick_marks[alg], pred_taus[1 - alg], rotation=45)
            plt.yticks(real_tick_marks, real_taus)

            # Añadir los valores dentro de cada celda
            # Añadir los valores dentro de cada celda
            for i in range(len(real_taus)):
                for j in range(len(pred_taus[1 - alg])):
                    plt.text(
                        j,
                        i,
                        f"{cm_incorrect_norm[alg][i, j]:.2f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if cm_incorrect_norm[alg][i, j]
                            > cm_incorrect_norm[alg].max() / 2
                            else "black"
                        ),
                    )

            plt.ylabel("Desintegración simulada", fontsize=15)
            plt.xlabel("Desintegración identificada", fontsize=15)
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    draw_path,
                    f"cm_incorrect_{algorithms[alg]}_vs_{algorithms[1-alg]}_norm.png",
                )
            )
            plt.close()
    return cm_incorrect_df, cm_incorrect_norm_df


def PlotCMs(cm, draw_path=None, name="CM Matrix GATr", normalize=False):
    """Plot the confusion matrix wich is a datagrame with values, index and columns"""
    # print(cm)
    # print(cm.max().max())
    real_tick_marks = np.arange(len(cm.index))
    pred_tick_marks = np.arange(len(cm.columns))
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(name, fontsize=20)

    x_labels = cm.columns
    x_labels = [id_to_key(int(x), False) for x in x_labels]
    y_labels = cm.index
    y_labels = [id_to_key(int(x), False) for x in y_labels]
    plt.xticks(pred_tick_marks, x_labels, rotation=45, fontsize=14)
    plt.yticks(real_tick_marks, y_labels, fontsize=14)
    # Añadir los valores dentro de cada celda
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            if normalize:
                plt.text(
                    j,
                    i,
                    f"{cm.iloc[i, j] * 100:.2f}%",
                    ha="center",
                    va="center",
                    color="white" if cm.iloc[i, j] > cm.max().max() / 2 else "black",
                    # fontsize = 15
                )
            else:
                plt.text(
                    j,
                    i,
                    f"{cm.iloc[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if cm.iloc[i, j] > cm.max().max() / 2 else "black",
                    # fontsize = 15
                )
    plt.ylabel("Desintegración simulada", fontsize=15)
    plt.xlabel("Desintegración identificada", fontsize=15)
    plt.tight_layout()
    if draw_path is not None:
        save_name = name.replace(" ", "_")
        plt.savefig(os.path.join(draw_path, f"{save_name}.png"), bbox_inches="tight")
        # print(f"Guardando imagen en {os.path.join(draw_path, f"{save_name}.png")}")
        plt.close()
    else:
        plt.show()
    return


# ---------------------------------------------------------------
# Argparse
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-i",
#     "--input",
#     type=str,
#     default="inference_data",
#     help="Input file with predictions",
# )
parser.add_argument("--c-1", type=str, default="inference_data/result_labels_MLID.csv", help="Model 1 to compare against")
parser.add_argument("--c-2", type=str, default="inference_data/result_labels_MLPF.csv", help="Model 2 to compare against")
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="results/",
    help="Output path of predictions evaluation",
)
parser.add_argument(
    "-d", "--draw", type=str, default=False, help="Output path of figures"
)
parser.add_argument("-k", "--decay-keys", nargs="+", type=int, default=[])

args = parser.parse_args()

# input_file = args.input
output_path = args.output
draw_path = args.draw
decay_keys = args.decay_keys
compare_1 = args.c_1
compare_2 = args.c_2

input_file_compare_1 = compare_1
input_file_compare_2 = compare_2
# input_file_pfo
# ---------------------------------------------------------------
if not os.path.exists(output_path):
    os.makedirs(output_path)

logging.basicConfig(
    level=logging.INFO,  # Nivel de registro
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato del mensaje
    handlers=[
        logging.StreamHandler(),  # Salida en consola
        logging.FileHandler(
            os.path.join(output_path, "predictions_evaluation.log")
        ),  # Salida en archivo
    ],
)
logger = logging.getLogger("InferenceEvalLogger")

logger.info("Starting predictions evaluation...")

if not draw_path:
    draw_path = os.path.join(output_path, "Figures")
    if not os.path.exists(draw_path):
        os.makedirs(draw_path)
else:
    draw_path = os.path.join(output_path, draw_path)
# else:
    # draw_path = None

if not os.path.exists(draw_path):
    os.makedirs(draw_path)
    
logger.info(f"Figures will be saved in {draw_path}")
# exit(0)
# ---------------------------------------------------------------
# Load the predictions
try:
    if "MLID" in input_file_compare_1:
        model_1 = "MLID"
    elif "pfo" in input_file_compare_1:
        model_1 = "PandoraPFO"
    elif "MLPF" in input_file_compare_1:
        model_1 = "MLPF"
    result_labels_compare_1 = pd.read_csv(input_file_compare_1)
    logger.info(f"Predictions loaded from {input_file_compare_1}")
    result_labels_compare_1 = event_to_tau(
        result_labels_compare_1, model=model_1, outputpath=output_path
    )
except FileNotFoundError as e:
    logger.error(f"File {input_file_compare_1} not found.")
    raise e

try:
    if "MLID" in input_file_compare_2:
        model_2 = "MLID"
    elif "pfo" in input_file_compare_2:
        model_2= "PandoraPFO"
    elif "MLPF" in input_file_compare_2:
        model_2 = "MLPF"
    result_labels_compare_2 = pd.read_csv(input_file_compare_2)
    logger.info(f"Predictions loaded from {input_file_compare_2}")
    result_labels_compare_2 = event_to_tau(
        result_labels_compare_2, model=model_2, outputpath=output_path
    )
except FileNotFoundError as e:
    logger.error(f"File {input_file_compare_2} not found.")
    result_labels_compare_2 = None

# ---------------------------------------------------------------
summary_evaluation_results = {
    "total_correct_preds": 0,
    "total_correct_preds_fraction": 0,
    "Purity": {},
    "Completeness": {},
}

logger.info("Starting evaluation of predictions...")
# Loop over the predictions
from copy import deepcopy

summary_evaluation_results_1 = deepcopy(summary_evaluation_results)
summary_evaluation_results_2 = deepcopy(summary_evaluation_results)
cm_matrix_model_1, fails_tau_model_1 = evaluate_result_labels(
    result_labels_compare_1, summary_evaluation_results_1, logger, output_path, model_1
)
cm_matrix_model_1_norm = cm_matrix_model_1.div(  # DataFrame original
    cm_matrix_model_1.sum(axis=1), axis=0
).fillna(  # divide cada fila por su suma
    0
)  # evita NaN cuando la fila suma 0
# Filtrar para mostrar (después de normalizar)
cm_matrix_model_1_show   = filter_cm_counts(cm_matrix_model_1, decay_keys)
cm_matrix_model_1_norm_show = filter_cm_after_norm(cm_matrix_model_1_norm, decay_keys)

PlotCMs(cm_matrix_model_1_show, draw_path=draw_path, name=f"Matriz de confusión ({model_1})")
PlotCMs(
    cm_matrix_model_1_norm_show,
    draw_path=draw_path,
    name=f"Matriz de confusión normalizada ({model_1})",
    normalize=True,
)


if result_labels_compare_2 is not None:
    cm_matrix_model_2, fails_tau_model_2 = evaluate_result_labels(
        result_labels_compare_2, summary_evaluation_results_2, logger, output_path, model_2
    )
    cm_incorrect_model_1 = incorrectCM(result_labels_compare_2, fails_tau_model_1)
    cm_incorrect_model_2 = incorrectCM(result_labels_compare_1, fails_tau_model_2)
    # Normalize the confusion matrix by row (i.e. by the number of samples)
    cm_incorrect_model_1_norm = cm_incorrect_model_1.div(  # DataFrame original
        cm_incorrect_model_1.sum(axis=1), axis=0
    ).fillna(  # divide cada fila por su suma
        0
    )  # evita NaN cuando la fila suma 0
    cm_incorrect_model_2_norm = cm_incorrect_model_2.div(  # DataFrame original
        cm_incorrect_model_2.sum(axis=1), axis=0
    ).fillna(  # divide cada fila por su suma
        0
    )  # evita NaN cuando la fila suma 0
    cm_matrix_model_2_norm = cm_matrix_model_2.div(  # DataFrame original
        cm_matrix_model_2.sum(axis=1), axis=0
    ).fillna(  # divide cada fila por su suma
        0
    )  # evita NaN cuando la fila suma 0
    cm_matrix_model_2_show   = filter_cm_counts(cm_matrix_model_2, decay_keys)
    cm_matrix_model_2_norm_show = filter_cm_after_norm(cm_matrix_model_2_norm, decay_keys)
    PlotCMs(cm_matrix_model_2_show, draw_path=draw_path, name=f"Matriz de confusión ({model_2})")
    PlotCMs(
        cm_matrix_model_2_norm_show,
        draw_path=draw_path,
        name=f"Matriz de confusión normalizada ({model_2})",
        normalize=True,
    )
    cm_incorrect_model_1_show = filter_cm_counts(cm_incorrect_model_1, decay_keys)
    cm_incorrect_model_2_show = filter_cm_counts(cm_incorrect_model_2, decay_keys)
    cm_incorrect_model_1_norm_show = filter_cm_after_norm(cm_incorrect_model_1_norm, decay_keys)
    cm_incorrect_model_2_norm_show = filter_cm_after_norm(cm_incorrect_model_2_norm, decay_keys)
    PlotCMs(
        cm_incorrect_model_1_show,
        draw_path=draw_path,
        name=f"{model_2} preds when {model_1} fails",
    )
    PlotCMs(
        cm_incorrect_model_2_show,
        draw_path=draw_path,
        name=f"{model_1} Method preds when {model_2} fails",
    )
    PlotCMs(
        cm_incorrect_model_1_norm_show,
        draw_path=draw_path,
        name=f"{model_2} preds when {model_1} fails normalized",
        normalize=True,
    )
    PlotCMs(
        cm_incorrect_model_2_norm_show,
        draw_path=draw_path,
        name=f"{model_1} Method preds when {model_2} fails normalized",
        normalize=True,
    )


logger.info("Evaluation results saved.")
logger.info("Evaluation completed.")
# ---------------------------------------------------------------
