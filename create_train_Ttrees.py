#!/usr/bin/env python3
import argparse
import subprocess
import logging
import time
from pathlib import Path

def run_job(input_path: Path, output_path: Path, idx: int, log_dir: Path) -> subprocess.Popen:
    """
    Lanza el proceso en background, redirige stdout/stderr a job_<idx>.log,
    y lo comienza en una nueva sesión (para que ignore SIGHUP al cerrar terminal).
    """
    log_file = log_dir / f"job_{idx}.log"
    cmd = [
        "python",
        "modelTrain/PID_GNN/data_generation/condor/make_pftree_clic_bindings_tautau.py",
        str(input_path),
        str(output_path),
        "False",
        "False",
        str(idx),
    ]
    logging.debug(f"Lanzando: {' '.join(cmd)} → {log_file}")
    p = subprocess.Popen(
        cmd,
        stdout=log_file.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    return p

def main():
    parser = argparse.ArgumentParser(description="Lanza jobs en paralelo tipo nohup")
    parser.add_argument("dir_path",      help="Directorio con los ROOT files de entrada")
    parser.add_argument("--file-base",   default="out_reco_edm4hep_edm4hep",
                        help="Prefijo de los archivos de entrada")
    parser.add_argument("--nfiles",      type=int, default=500,
                        help="Número total de archivos")
    parser.add_argument("--test",        action="store_true",
                        help="Modo test (solo procesa 2 archivos)")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Máximo de procesos concurrentes")
    parser.add_argument("--log-dir",     default="TestTrees/logs",
                        help="Directorio donde escribir los logs")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Modo verbose (debug logs)")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=level)

    nfiles = 2 if args.test else args.nfiles
    dir_path = Path(args.dir_path)
    log_dir  = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    out_dir = Path("TestTrees")
    out_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Leyendo archivos de: {dir_path}")
    logging.info(f"Prefijo: {args.file_base}  nfiles: {nfiles}  concurrentes: {args.max_workers}")

    active = []
    for i in range(1, nfiles + 1):
        inp = dir_path / f"{args.file_base}_{i}.root"
        out = out_dir / f"tree_{i}.root"

        if not inp.is_file():
            logging.warning(f"File no encontrado, salto: {inp}")
            continue
        if out.is_file():
            logging.info(f"Salida ya existe, salto: {out}")
            continue

        # Arrancar el job y añadirlo a la lista de activos
        p = run_job(inp, out, i, log_dir)
        active.append(p)
        logging.info(f"Job {i} lanzado con PID={p.pid}")

        # Mientras tengamos >= max_workers activos, esperamos a que alguno termine
        while len(active) >= args.max_workers:
            time.sleep(1)
            # filtramos para quedarnos solo con los que siguen vivos
            active = [proc for proc in active if proc.poll() is None]

    logging.info("Se han lanzado todos los jobs. El script puede terminar ahora.")
    # NOTA: no hacemos wait() final. 
    # Los procesos seguirán vivos en su propia sesión y podrás verlos con ps.
    # El script termina inmediatamente liberando la terminal.

if __name__ == "__main__":
    main()
