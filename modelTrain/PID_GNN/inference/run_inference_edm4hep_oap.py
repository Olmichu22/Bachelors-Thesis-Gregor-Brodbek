"""
Inferencia con modelo GATr PF para taus sin ONNX.
Procesa eventos de un ROOT de EDM4hep y clasifica dos taus por evento.
"""
import os
import sys
import torch
import numpy as np
from podio import root_io
from modelTrain.PID_GNN.inference.tree_tools_tautau_inference import (
    initialize,
    get_tracks,
    store_calo_hits,
    create_inputs,
    split_taus
)
# Asegúrate de que tu PYTHONPATH incluya la raíz del proyecto para src/
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.utils.parser_args import parser
# from src.utils.train_utils import get_samples_steps_per_epoch, model_setup
# from src.models.Gatr_pf_e_tau_rho import ExampleWrapper


def load_model(args, data_config):
    # Instancia modelo y carga checkpoint
    from src.models.Gatr_pf_e_tau_rho import ExampleWrapper as GravnetModel
    model = GravnetModel.load_from_checkpoint(
        args.load_model_weights, args=args, dev=0, strict=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)
    return model, device


def infer_event(model, device, inputs_dict):
    # Obtiene datos PF para dos taus
    data1, data2 = split_taus(*inputs_dict)
    outs = []
    for data in (data1, data2):
        # Convertir a tensores y agregar batch
        pf_points    = torch.from_numpy(data['pf_points']).unsqueeze(0).float().to(device)
        pf_features  = torch.from_numpy(data['pf_features']).unsqueeze(0).float().to(device)
        pf_vectors   = torch.from_numpy(data['pf_vectors']).unsqueeze(0).float().to(device)
        pf_vectoronly= torch.from_numpy(data['pf_vectoronly']).unsqueeze(0).float().to(device)
        pf_mask      = torch.from_numpy(data['pf_mask']).unsqueeze(0).float().to(device)
        # Inferencia
        with torch.no_grad():
            out = model(
                pf_points,
                pf_features,
                pf_vectors,
                pf_vectoronly,
                pf_mask
            )
        outs.append(out.cpu().numpy())
    return outs


def get_samples_steps_per_epoch(args):
    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!"
            )
    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!"
            )
    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(
            args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split
        )
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None
    return args
  
def main():
    args = parser.parse_args()
    # args = get_samples_steps_per_epoch(args)

    if not args.data_test:
        print("Debes pasar --data-test <ruta_root>")
        sys.exit(1)
    input_file = args.data_test[0]

    # Cargar configuración de datos (puede ser dummy si no se usa)
    # model_setup requiere data_config, pero parse_args() en modo inferencia podría usar test_load
    from src.utils.train_utils import test_load
    test_loaders, data_config = test_load(args)

    # Cargar modelo
    model, device = load_model(args, data_config)

    # Abrir reader
    reader = root_io.Reader(input_file)
    for i, event in enumerate(reader.get('events')):
        # Por ejemplo inferimos solo primer evento
        if i >= 1:
            break
        # Preprocesar evento
        dic = initialize()
        dic = get_tracks(event, dic)
        dic = store_calo_hits(event, dic)
        input_data, inputs = create_inputs(dic)

        # Empaquetar dict para split
        # split_taus espera (input_data, inputs)
        outs = infer_event(model, device, (input_data, inputs))
        print(f"Evento {i}: Tau1 pred = {outs[0]}, Tau2 pred = {outs[1]}")

if __name__ == '__main__':
    main()