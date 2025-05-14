import pandas as pd
import numpy as np
import logging
import argparse
import os
import yaml
import matplotlib.pyplot as plt
# ---------------------------------------------------------------
def pseudoCMs(result_labels, evaluation_results_df, draw_path=None):
    """Create a pseudo confusion matrix for the predictions"""
    real_taus = list(set(result_labels["tau1"].tolist() + result_labels["tau2"].tolist()))
    pred_taus_set = list(set(result_labels["id-tau1"].tolist() + result_labels["id-tau2"].tolist()))
    
    real_taus.sort()
    pred_taus_set.sort()
    # Assing a number to each tau
    real_taus_dict = {tau: i for i, tau in enumerate(real_taus)}
    pred_taus_dict = {tau: i for i, tau in enumerate(pred_taus_set)}
    
    # Confusion matrix with all data
    cm_all = np.zeros((len(real_taus), len(pred_taus_set)))
    for i, row in result_labels.iterrows():
        cm_all[real_taus_dict[row["tau1"]], pred_taus_dict[row["id-tau1"]]] += 0.5
        cm_all[real_taus_dict[row["tau1"]], pred_taus_dict[row["id-tau2"]]] += 0.5
        cm_all[real_taus_dict[row["tau2"]], pred_taus_dict[row["id-tau2"]]] += 0.5
        cm_all[real_taus_dict[row["tau2"]], pred_taus_dict[row["id-tau1"]]] += 0.5
    
    # Confusion matrix with only incorrect predictions
    cm_incorrect = np.zeros((len(real_taus), len(pred_taus_set)))
    for i, row in evaluation_results_df.iterrows():
        if row["num_correct_preds"] < 2:
            for fail_tau in row["incorrect_preds_real_ids"]:
                for pred_tau in row["incorrect_preds_pred_ids"]:
                    cm_incorrect[real_taus_dict[fail_tau], pred_taus_dict[pred_tau]] += 1/len(row["incorrect_preds_real_ids"])
    
    cm_all_df = pd.DataFrame(cm_all, index=real_taus, columns=pred_taus_set)
    cm_incorrect_df = pd.DataFrame(cm_incorrect, index=real_taus, columns=pred_taus_set)
    if draw_path is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_all, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix (all data)')
        plt.colorbar()
        real_tick_marks = np.arange(len(real_taus))
        pred_tick_marks = np.arange(len(pred_taus_set))
        plt.xticks(pred_tick_marks, pred_taus_set, rotation=45)
        plt.yticks(real_tick_marks, real_taus)
        
        # Añadir los valores dentro de cada celda
        for i in range(len(real_taus)):
            for j in range(len(pred_taus_set)):
                plt.text(j, i, f"{cm_all[i, j]:.1f}", 
                         ha="center", va="center", 
                         color="white" if cm_all[i, j] > cm_all.max()/2 else "black")
                
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(draw_path, "cm_all.png"))
        plt.close()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_incorrect, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix (incorrect predictions)')
        plt.colorbar()
        plt.xticks(pred_tick_marks, pred_taus_set, rotation=45)
        plt.yticks(real_tick_marks, real_taus)
        
        # Añadir los valores dentro de cada celda
        for i in range(len(real_taus)):
            for j in range(len(pred_taus_set)):
                plt.text(j, i, f"{cm_incorrect[i, j]:.1f}", 
                         ha="center", va="center", 
                         color="white" if cm_incorrect[i, j] > cm_incorrect.max()/2 else "black")
                
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(draw_path, "cm_incorrect.png"))
        plt.close()
    return cm_all_df, cm_incorrect_df



# ---------------------------------------------------------------
# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input', type=str, default="inference_data/result_labels.csv",
                       help='Input file with predictions')
parser.add_argument("-o", '--output', type=str, default="results/",
                       help='Output path of predictions evaluation')
parser.add_argument("-d", '--draw', type=str, default=False,
                       help='Output path of figures')

args = parser.parse_args()

input_file = args.input
output_path = args.output
draw_path = args.draw

# ---------------------------------------------------------------
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
logging.basicConfig(
    level=logging.INFO,  # Nivel de registro
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato del mensaje
    handlers=[
        logging.StreamHandler(),  # Salida en consola
        logging.FileHandler(os.path.join(output_path, "predictions_evaluation.log"))  # Salida en archivo
    ]
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
    result_labels = pd.read_csv(input_file)
    logger.info(f"Predictions loaded from {input_file}")
except FileNotFoundError as e:
    logger.error(f"File {input_file} not found.")
    raise e
  
# ---------------------------------------------------------------
evaluation_results = {"num_correct_preds":[],
                      "num_incorrect_preds":[],
                      "correct_pred_ids":[],
                      "incorrect_preds_real_ids":[],
                      "incorrect_preds_pred_ids":[]}
summary_evaluation_results = {"total_correct_preds": 0,
                              "total_correct_preds_fraction": 0,
                              "total_incorrect_preds": 0,
                              "total_incorrect_preds_fraction": 0,
                              "incorrect_preds_summary": {},
                              "correct_preds_summary": {}}

logger.info("Starting evaluation of predictions...")
# Loop over the predictions
for i, row in result_labels.iterrows():
    real_taus_list = [row["tau1"], row["tau2"]]
    pred_taus_list = [row["id-tau1"], row["id-tau2"]]
    real_taus_set = set(real_taus_list)
    pred_taus_set = set(pred_taus_list)
    
    num_correct = 0
    correct_pred_ids = []
    incorrect_pred_real_ids = []
    pred_taus_temp = pred_taus_list.copy()
    real_taus_temp = real_taus_list.copy()
    for tau in real_taus_list:
        print(f"tau: {tau}")
        print(f"pred_taus_temp: {pred_taus_temp}")
        print(f"real_taus_temp: {real_taus_temp}")
        if tau in pred_taus_temp:
            correct_pred_ids.append(tau)
            num_correct += 1
            pred_taus_temp.remove(tau)
        else:
            incorrect_pred_real_ids.append(tau)
            
            # real_taus_temp.remove(tau)
    # print("\n")
    # print(f"correct_pred_ids: {correct_pred_ids}")
    # print(f"incorrect_pred_real_ids: {incorrect_pred_real_ids}")
    # print(f"num_correct: {num_correct}")
    # print(f"pred_taus_temp: {pred_taus_temp}")
    # if i == 2:
    #     break
        
    
    num_incorrect = 2 - num_correct
    evaluation_results["num_correct_preds"].append(num_correct)
    evaluation_results["num_incorrect_preds"].append(num_incorrect)        
    evaluation_results["correct_pred_ids"].append(correct_pred_ids)
    evaluation_results["incorrect_preds_real_ids"].append(incorrect_pred_real_ids)
    evaluation_results["incorrect_preds_pred_ids"].append(pred_taus_temp)
    logger.info(f"Event {i}: {num_correct} correct predictions, {num_incorrect} incorrect predictions")
    
summary_evaluation_results["total_correct_preds"] = sum(evaluation_results["num_correct_preds"])
summary_evaluation_results["total_incorrect_preds"] = sum(evaluation_results["num_incorrect_preds"])
summary_evaluation_results["total_correct_preds_fraction"] = summary_evaluation_results["total_correct_preds"] / (len(result_labels)*2)
summary_evaluation_results["total_incorrect_preds_fraction"] = summary_evaluation_results["total_incorrect_preds"] / (len(result_labels)*2)

miss_events = []
for incorrec_pred in evaluation_results["incorrect_preds_real_ids"]:
  if incorrec_pred != list() and incorrec_pred not in miss_events:
    miss_events.append(incorrec_pred)
# Count the number of occurrences of each event in the list
miss_events_count = {str(event): evaluation_results["incorrect_preds_real_ids"].count(event) for event in miss_events}
summary_evaluation_results["incorrect_preds_summary"] = miss_events_count

correct_events = []
for correct_pred in evaluation_results["correct_pred_ids"]:
  if correct_pred != list() and correct_pred not in correct_events:
    correct_events.append(correct_pred)

# Count the number of occurrences of each event in the list
correct_events_count = {str(event): evaluation_results["correct_pred_ids"].count(event) for event in correct_events}
summary_evaluation_results["correct_preds_summary"] = correct_events_count


logger.info(f"Saving evaluation results to {output_path}")
# Save the evaluation results
evaluation_results_df = pd.DataFrame(evaluation_results)
evaluation_results_df.to_csv(os.path.join(output_path, "evaluation_results.csv"), index=False)

# Save summary results as yaml
summary_evaluation_results_file = os.path.join(output_path, "summary_evaluation_results.yaml")
with open(summary_evaluation_results_file, 'w') as file:
    yaml.dump(summary_evaluation_results, file)



cm_all, cm_incorrect = pseudoCMs(result_labels, evaluation_results_df, draw_path)
cm_all.to_csv(os.path.join(output_path, "cm_all.csv"))
cm_incorrect.to_csv(os.path.join(output_path, "cm_incorrect.csv"))

logger.info("Evaluation results saved.")
logger.info("Evaluation completed.")
# ---------------------------------------------------------------    