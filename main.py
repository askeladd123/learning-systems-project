import os
import sys
import toml
import select
import logging
import hashlib
import argparse
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

log = logging.getLogger(__name__)

@dataclass
class Params:
    epochs: int
    number_of_clauses: int
    t: int
    s: float
    depth: 3
    hypervector_size: int
    hypervector_bits: int
    message_size: int
    message_bits: int
    double_hashing: bool
    max_included_literals: int
    

if __name__ == '__main__':

    ## startup
    # parse options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True)

    args = parser.parse_args()

    # load parameteres
    if select.select([sys.stdin], [], [], 0.0)[0]:
        input = sys.stdin.read().strip()
    else:
        log.info('using default-params.toml')
        with open('default-params.toml', 'r') as file:
            input = file.read()
    input_toml = toml.loads(input)
    params = Params(**input_toml)

    # establish new run
    run_name = os.path.join('runs', datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
    os.makedirs(run_name)
    log_file = os.path.join(run_name, 'log.txt')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    logging.basicConfig(handlers=[file_handler, console_handler], level=logging.DEBUG, format='%(message)s')
    log.info(f'creating new run in {run_name}')

    # report which params used
    params_file = os.path.join(run_name, 'params.toml')
    with open(params_file, 'w') as file:
        data = asdict(params)
        toml.dump(data, file)

    # load dataset and report
    report_filename = os.path.join(run_name, 'dataset.toml')
    log.info(f'hashing dataset to {report_filename}')
    hasher = hashlib.blake2b()
    with open(args.data, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    hash = hasher.hexdigest()
    report = { 'filename': args.data, 'hash-blake2b': hash }
    with open(report_filename, 'w') as file:
        toml.dump(report, file)

    log.info(f'loading dataset from {args.data}')
    dataset = pd.read_csv(args.data)

    ## main
