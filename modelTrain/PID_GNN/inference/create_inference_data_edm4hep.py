import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep
import onnxruntime as ort
from tree_tools_tautau_inference import (
    initialize,
    get_tracks,
    store_calo_hits,
    create_inputs, 
    split_taus,
    associate_hemispheres_with_taus
)
import os
from tauReco import findAllGenTaus
import time
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
## global params
CALO_RADIUS_IN_MM = 1500


# Set up logging
# Configura el logger para que DEBUG vaya solo al archivo y INFO+ a la terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handler para archivo (DEBUG y superior)
file_handler = logging.FileHandler("inference_data/log.out")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Handler para terminal (INFO y superior)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_formatter)

# Añade los handlers al logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Create inference data for tau reconstruction")
parser.add_argument(
    "-i", 
    "--input_files",
    type=str,
    nargs="+",
    default=["out_reco_edm4hep_edm4hep_1.root"],
    help="List of input files to process",
)
parser.add_argument(
    "-d",
    "--draw",
    type=str,
    default="False",
    help="Draw the events clustering",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="inference_data/",
    help="Output directory for the inference data",
)
args = parser.parse_args()

input_files = args.input_files
draw = args.draw
draw = True if draw == "True" else False
output_dir = args.output
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info("Output directory %s created", output_dir)

event_labels = {"tau1":[], "tau2":[]}
tau_data = {}

logger.info("Input files: %s", input_files)

n_events = 0

for input_file in input_files:
    if not os.path.exists(input_file):
        logger.error("Input file %s does not exist", input_file)
        continue
    if not input_file.endswith(".root"):
        logger.error("Input file %s is not a ROOT file", input_file)
        continue
    if not os.path.isfile(input_file):
        logger.error("Input file %s is not a valid file", input_file)
        continue
    
    logger.info("Processing file: %s", input_file)
    
    reader = root_io.Reader(input_file)
    ### initiali
    for i, event in enumerate(reader.get("events")):
    # Get Gen Particles

        time_start = time.time()
        
        gen_particles = event.get("MCParticles")
        taus = findAllGenTaus(gen_particles)
        if len(taus)<=2:
            for j in range(len(taus)):
                    event_labels[f"tau{j+1}"].append(taus[j].getID())
                    logger.debug("Tau %d ID: %d", j+1, taus[j].getID())
        else:
            event_labels["tau1"].append(-999)
            event_labels["tau2"].append(-999)
            for j in range(len(taus)):
                logger.warning("More than 2 taus found in event %d: %s", i, [tau for tau in taus])

            
        dic = initialize()
        dic = get_tracks(
            event,
            dic
        )
        dic = store_calo_hits(
            event,
            dic
        )
        try:
            input_data, inputs = create_inputs(dic)
            input_data1, input_data2 = split_taus(input_data, inputs)
            input_data_dict = associate_hemispheres_with_taus(input_data1, input_data2, taus)

        except Exception as e:
            tau_data[f"tau_{n_events}_1"] = np.nan
            tau_data[f"tau_{n_events}_2"] = np.nan
            logger.debug("Shape of input_data: %s", str(input_data.shape))
            logger.debug("Error creating inputs for event %d: %s", n_events, str(e))
            n_events += 1
            continue
        if draw:
            print("Drawing event %d", n_events)
            # 3d plot of hit_x, hit_y, hit_z
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(input_data1[:, 0], input_data1[:, 1], input_data1[:, 2], c="r", marker="o", label="tau candidate 1")
            ax.scatter(input_data2[:, 0], input_data2[:, 1], input_data2[:, 2], c="b", marker="o", label="tau candidate 2")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            ax.set_title("Hemispheres Signal Clustering")
            ax.legend()
            plt.savefig(f"inference_data/event_{i}_clustering.png")
            plt.close(fig)
            exit()
        tau_data[f"tau_{n_events}_1"] = input_data_dict["tau1"]
        tau_data[f"tau_{n_events}_2"] = input_data_dict["tau2"]
        n_events += 1
        
        time_end = time.time()
        logger.info("Time taken for event %d: %f seconds", i, time_end - time_start)
        # print(event_labels)

np.savez(os.path.join(output_dir,"all_data.npz"), **tau_data)
results = pd.DataFrame(event_labels)
results.to_csv(os.path.join(output_dir,"event_labels.csv"), index=False)