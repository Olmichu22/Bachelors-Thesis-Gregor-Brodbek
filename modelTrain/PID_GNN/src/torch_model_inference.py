# save_model_for_inference.py
# ---------------------------------------------------------------
#  * Carga el checkpoint del modelo "no-ONNX".
#  * Envuelve solo las partes puramente-Torch (BatchNorm, GATr, MLP).
#  * Acepta un tensor plano (N,11) como en los ficheros NPY
#    pero descarta la última columna (hit_type4) para que queden 10
#    → de ahí tomamos los 3 contadores que la red usa.
# ---------------------------------------------------------------
import argparse, os, sys, torch, torch.nn as nn
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.models.Gatr_pf_e_tau_rho import ExampleWrapper as OldModel
from gatr.interface import embed_point
import numpy as np
import pandas as pd
import logging


# ---------------------------------------------------------------

def load_graph_tensor(array: np.ndarray) -> torch.Tensor:
  array = array.astype(np.float32)  # (N,11)
  return torch.from_numpy(array)

class InferenceWrapper(nn.Module):
    """
    Entrada  : tensor (N, 11)
               [0:3]   xyz
               [3]     hit_type
               [4:7]   (E, p, logE)
               [7:11]  one-hot hit_type1-4   ← descartamos la última
    Salida    : logits (1, 9)
    """
    def __init__(self, old: OldModel):
        super().__init__()
        # Re-usar directamente los sub-módulos del modelo entrenado
        self.bn   = old.ScaledGooeyBatchNorm2_1        # BatchNorm(3)
        self.gatr = old.gatr                           # GATr completo
        self.mlp  = old.MLP_layer                      # MLPReadout(20→9)

        # Congelar pesos (opcional)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):               # x: (N,11) en **CPU**
        # ----------- de-pack -----------
        pts        = x[:, 0:3]          # (N,3)
        hit_type   = x[:, 3:4]          # (N,1)
        energy     = x[:, 4:7]          # (N,3)
        one_hot_4  = x[:, 7:11]         # (N,4)
        one_hot    = one_hot_4[:, :3]   # (N,3)  ← tipos 1,2,3

        # ----------- paso GATr ---------
        pts_bn     = self.bn(pts)                       # (N,3)
        mv_in      = embed_point(pts_bn).unsqueeze(-2)  # (N,1,16)
        scalars    = torch.cat((energy[:, :2],          # usamos solo E,p
                                hit_type), dim=1)       # (N,3)

        mv_out, sc_out = self.gatr(mv_in, scalars=scalars)  # (N,1,16),(N,1)

        h_node   = torch.cat((mv_out.squeeze(1), sc_out), dim=1)  # (N,17)

        # ----------- reducción global (suma de nodos) ----------
        hg         = h_node.sum(dim=0, keepdim=True)    # (1,17)
        oh_sum     = one_hot.sum(dim=0, keepdim=True)   # (1,3)

        features   = torch.cat((oh_sum, hg), dim=1)     # (1,20)
        return self.mlp(features)                       # (1,9)


# ----------------------------------------------------------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default="/nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek/FinalModel/FinalModel.ckpt",
                   help="Checkpoint Lightning (.ckpt)")
    p.add_argument("--out", default="/nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek/FinalModel/wrapper.pt",
                   help="Path to save model (or load if exists)")
    p.add_argument("-i", "--input",
                   default="/nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek/inference_data/",
                   help="Path to folder with input files")
    p.add_argument("-t", "--test", default=False)
    return p.parse_args()


def main():
  args = cli()
  output_path = args.out
  inference_path = args.input
  ckpt_path = args.ckpt
  test = args.test
  test = True if test == "True" else False
  
  logging.basicConfig(
            level=logging.INFO,  # Nivel de registro
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato del mensaje
            handlers=[
                logging.StreamHandler(),  # Salida en consola
                logging.FileHandler(inference_path+"modelInference.log")  # Salida en archivo
            ]
        )

  logger = logging.getLogger("InferenceLogger")
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # Tengo errores con cuda parece...
  device = "cpu"
  logger.info(f"Device: {device}")
  
  if not os.path.exists(output_path):
    logger.info("Model not found, creating wrapper")
    # 1) cargar modelo original
    old = OldModel.load_from_checkpoint(ckpt_path, args=None, dev=0,
                                        strict=False).to(device).eval()
    # 2) construir wrapper
    model = InferenceWrapper(old).to(device).eval()
    logger.info("Model loaded!")
  else:
    model = torch.jit.load(output_path, map_location = device).eval()
    logger.info("Model loaded from file!")

  

  # Cargamos el diccionario de true_labels
  true_labels_path = inference_path+ "event_labels.csv"
  # Lo cargamos con yaml
  try:
    true_labels = pd.read_csv(true_labels_path)
    result_labels = {col: true_labels[col].tolist() for col in true_labels.columns}
    if test:
      # Cogemos solo las primeras 10 filas
      result_labels = {k: v[:10] for k, v in result_labels.items()}
    logger.info(f"True labels loaded from {true_labels_path}")
  except Exception as e:
    logger.info(f"Error when loading {true_labels_path}: {e}")
    result_labels = pd.DataFrame()
  
  
  result_labels["num-tau1"] = []
  result_labels["num-tau2"] = []
  result_labels["id-tau1"] = []
  result_labels["id-tau2"] = []
  
  
  # Listamos los ficheros de entrada

  input_file_path = inference_path + "all_data.npz"
  try:
    input_data = np.load(input_file_path)
  except Exception as e:
    logger.info(f"Error when loading {input_file_path}: {e}")
    return
  N_events = len(input_data)//2
  logger.info(f"Len of inference file: {len(input_data)}")
  logger.info(f"Number of events: {N_events}")
  # exit()
  pos_to_label = {0:"e",          
              1:"mu",
              2:"pi_pi0",
              3:"pi",
              4:"pi_2pi0",
              5:"3pi",
              6:"3pi_pi0",
              7:"qq background",
              8:"Bhabha"}
  pos_to_id = {0:-11,
                1:-13,
                2:1,
                3:0,
                4:2,
                5:10,
                6:11,
                7:-222,
                8:-333}
  
  logger.info(f"Starting inference with {N_events} events")
  # print(input_files)
  with torch.no_grad():
    for i in range(N_events):
      if test:
        if i > 9:
          break
      # Cargamos el fichero de entrada
      if np.isnan(input_data[f"tau_{i}_1"]).all():
        logger.info(f"Event {i} not found")
        result_labels["num-tau1"].append(-999)
        result_labels["num-tau2"].append(-999)
        result_labels["id-tau1"].append(-999)
        result_labels["id-tau2"].append(-999)
        continue
      x1 = load_graph_tensor(input_data[f"tau_{i}_1"]).to(device)  # (1,11)
      x2 = load_graph_tensor(input_data[f"tau_{i}_2"]).to(device)
      logits = model(x1)                               # (1,9)
      probs1  = torch.sigmoid(logits).cpu()               # multiclase independiente
      logits = model(x2)                               # (1,9)
      probs2  = torch.sigmoid(logits).cpu()                  # multiclase independiente
      # Guardamos los resultados
      label1 = np.argmax(probs1.numpy())
      label2 = np.argmax(probs2.numpy())
    
      logger.info(f"Event {i} out of {N_events}: \n tau1: {pos_to_label[label1]}, tau2: {pos_to_label[label2]}")
      
      # print(result_labels)
      # print(label1, label2)
      # print(type(label1), type(label2))
      result_labels["num-tau1"].append(label1)
      result_labels["num-tau2"].append(label2)
      result_labels["id-tau1"].append(pos_to_id[label1])
      result_labels["id-tau2"].append(pos_to_id[label2])
      
      # Guardamos los resultados en un fichero

  try:
    result_labels_df = pd.DataFrame(result_labels)
    result_labels_df.to_csv(inference_path + "result_labels.csv", index=False)
    logger.info(f"Results saved to {inference_path + 'result_labels.csv'}")
  except Exception as e:
    logger.info(f"Error when saving {inference_path + 'result_labels.csv'}: {e}")
    return
if __name__ == "__main__":
    main()


