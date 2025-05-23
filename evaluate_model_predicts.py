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


def id_to_key(event_id, photons=False):
    if photons:
        if event_id < 0:
            if event_id == -13:
                key = f"{MU}"
            elif event_id == -11:
                key = f"{E}"
            elif event_id <= -20:
                key = f"h{N}"
            else:
                key = "Unknown"
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
            else:
                key = "Unknown"
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

            plt.ylabel("True label", fontsize=15)
            plt.xlabel("Predicted label", fontsize=15)
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

            plt.ylabel("True label", fontsize=15)
            plt.xlabel("Predicted label", fontsize=15)
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
    plt.xticks(pred_tick_marks, x_labels, rotation=45)
    plt.yticks(real_tick_marks, y_labels)
    # Añadir los valores dentro de cada celda
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            if normalize:
                plt.text(
                    j,
                    i,
                    f"{cm.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm.iloc[i, j] > cm.max().max() / 2 else "black",
                )
            else:
                plt.text(
                    j,
                    i,
                    f"{cm.iloc[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if cm.iloc[i, j] > cm.max().max() / 2 else "black",
                )
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.tight_layout()
    if draw_path is not None:
        save_name = name.replace(" ", "_")
        plt.savefig(os.path.join(draw_path, f"{save_name}.png"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return


# ---------------------------------------------------------------
# Argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="inference_data",
    help="Input file with predictions",
)
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

input_file = args.input
output_path = args.output
draw_path = args.draw
decay_keys = args.decay_keys

input_file_gatr = os.path.join(input_file, "result_labels.csv")
input_file_pfo = os.path.join(input_file, "result_labels_pfo.csv")
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

if draw_path != "False":
    draw_path = os.path.join(output_path, "Figures")
    if not os.path.exists(draw_path):
        os.makedirs(draw_path)
    logger.info(f"Figures will be saved in {draw_path}")
else:
    draw_path = None

# ---------------------------------------------------------------
# Load the predictions
try:
    result_labels_gatr = pd.read_csv(input_file_gatr)
    logger.info(f"Predictions loaded from {input_file_gatr}")
    result_labels_gatr = event_to_tau(
        result_labels_gatr, model="GATr", outputpath=output_path
    )
except FileNotFoundError as e:
    logger.error(f"File {input_file_gatr} not found.")
    raise e

try:
    result_labels_pfo = pd.read_csv(input_file_pfo)
    logger.info(f"Predictions loaded from {input_file_pfo}")
    result_labels_pfo = event_to_tau(
        result_labels_pfo, model="PFO", outputpath=output_path
    )
except FileNotFoundError as e:
    logger.error(f"File {input_file_pfo} not found.")
    result_labels_pfo = None

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

summary_evaluation_results_gatr = deepcopy(summary_evaluation_results)
summary_evaluation_results_pfo = deepcopy(summary_evaluation_results)
cm_matrix_gatr, fails_tau_gatr = evaluate_result_labels(
    result_labels_gatr, summary_evaluation_results_gatr, logger, output_path, "GATr"
)
cm_matrix_gatr_norm = cm_matrix_gatr.div(  # DataFrame original
    cm_matrix_gatr.sum(axis=1), axis=0
).fillna(  # divide cada fila por su suma
    0
)  # evita NaN cuando la fila suma 0
PlotCMs(cm_matrix_gatr, draw_path=draw_path, name="GATr Confusion Matrix")
PlotCMs(
    cm_matrix_gatr_norm,
    draw_path=draw_path,
    name="GATr Confusion Matrix normalized",
    normalize=True,
)


if result_labels_pfo is not None:
    cm_matrix_pfo, fails_tau_pfo = evaluate_result_labels(
        result_labels_pfo, summary_evaluation_results_pfo, logger, output_path, "PFO"
    )
    cm_incorrect_gatr = incorrectCM(result_labels_pfo, fails_tau_gatr)
    cm_incorrect_pfo = incorrectCM(result_labels_gatr, fails_tau_pfo)
    PlotCMs(
        cm_incorrect_gatr,
        draw_path=draw_path,
        name="Classic Method preds when GATr fails",
    )
    PlotCMs(
        cm_incorrect_pfo,
        draw_path=draw_path,
        name="GATr Method preds when PlotCMs fails",
    )
    PlotCMs(cm_matrix_pfo, draw_path=draw_path, name="PFO Confusion Matrix")
    # Normalize the confusion matrix by row (i.e. by the number of samples)
    cm_incorrect_gatr_norm = cm_incorrect_gatr.div(  # DataFrame original
        cm_incorrect_gatr.sum(axis=1), axis=0
    ).fillna(  # divide cada fila por su suma
        0
    )  # evita NaN cuando la fila suma 0
    cm_incorrect_pfo_norm = cm_incorrect_pfo.div(  # DataFrame original
        cm_incorrect_pfo.sum(axis=1), axis=0
    ).fillna(  # divide cada fila por su suma
        0
    )  # evita NaN cuando la fila suma 0
    cm_matrix_pfo_norm = cm_matrix_pfo.div(  # DataFrame original
        cm_matrix_pfo.sum(axis=1), axis=0
    ).fillna(  # divide cada fila por su suma
        0
    )  # evita NaN cuando la fila suma 0

    PlotCMs(
        cm_incorrect_gatr_norm,
        draw_path=draw_path,
        name="Classic Method preds when GATr fails normalized",
        normalize=True,
    )
    PlotCMs(
        cm_incorrect_pfo_norm,
        draw_path=draw_path,
        name="GATr Method preds when PlotCMs fails normalized",
        normalize=True,
    )
    PlotCMs(
        cm_matrix_pfo_norm,
        draw_path=draw_path,
        name="PFO Confusion Matrix normalized",
        normalize=True,
    )


logger.info("Evaluation results saved.")
logger.info("Evaluation completed.")
# ---------------------------------------------------------------
