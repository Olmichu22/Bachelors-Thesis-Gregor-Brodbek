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
    split_taus
)
from tauReco import findAllGenTaus
import time
import pandas as pd
## global params
CALO_RADIUS_IN_MM = 1500



input_file = sys.argv[1]

reader = root_io.Reader(input_file)

########################################

# ort.set_default_logger_severity(0)

# so = ort.SessionOptions()
# so.enable_profiling = True

# so.inter_op_num_threads = 1
# so.intra_op_num_threads = 1
# so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL


# ort_session = ort.InferenceSession(
#     "/nfs/cms/arqolmo/GPU_train/Bachelors-Thesis-Gregor-Brodbek/model_multivector_3_input.onnx",
#     sess_options=so,
# )

n_events = 20
event_labels = {"tau1":[], "tau2":[]}
### initiali
tau_data = {}
for i, event in enumerate(reader.get("events")):
# Get Gen Particles

    time_start = time.time()
    
    gen_particles = event.get("MCParticles")
    taus = findAllGenTaus(gen_particles)
    if len(taus)<=2:
        for j in range(len(taus)):
                event_labels[f"tau{j+1}"].append(taus[j].getID())
    else:
        event_labels["tau1"].append(-999)
        event_labels["tau2"].append(-999)
        print("Extra taus:")
        for j in range(len(taus)):
            print(taus[j])
        
    dic = initialize()
    dic = get_tracks(
        event,
        dic
    )
    dic = store_calo_hits(
        event,
        dic
    )

    input_data, inputs = create_inputs(dic)
    input_data1, input_data2 = split_taus(input_data, inputs)
    tau_data[f"tau_{i}_1"] = input_data1
    tau_data[f"tau_{i}_2"] = input_data2
    time_end = time.time()
    print("Time taken for event {}: ".format(i), time_end - time_start)
    # print(event_labels)

np.savez("inference_data/all_data.npz", **tau_data)
results = pd.DataFrame(event_labels)
results.to_csv("inference_data/event_labels.csv", index=False)