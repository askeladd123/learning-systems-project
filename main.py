#!/usr/bin/env -S poetry run python
import os
import sys
import math
import toml
import select
import psutil
import logging
import hashlib
import argparse
import pandas as pd
import numpy as np
from time import time
from dataclasses import dataclass, asdict
from datetime import datetime
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

log = logging.getLogger(__name__)

@dataclass
class Params:
    epochs: int
    number_of_clauses: int
    t: int
    s: float
    depth: int
    hypervector_size: int
    hypervector_bits: int
    message_size: int
    message_bits: int
    double_hashing: bool
    max_included_literals: int
    

@dataclass
class Results:
    training_time_s: float
    inference_time_s: float
    memory_usage_mb: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class TrainOutput:
    results: Results
    confusion_matrix: list[list[int]]
    weights: list[int]


def train(data, params: Params, board_size: int) -> TrainOutput:
    subset_size = int(data.shape[0] * 0.9)
    test_size = data.shape[0] - subset_size
    X = data.iloc[:subset_size, 0].values
    X_test = data.iloc[subset_size:subset_size + test_size, 0].values
    y = data.iloc[:subset_size, 1].values
    y_test = data.iloc[subset_size:subset_size + test_size, 1].values

    log.info(f"X_train shape: {X.shape}")
    log.info(f"X_test shape: {X_test.shape}")

    symbol_names = ["O", "X", " "]

    # Create a list with node pairs for the edges
    edges = []
    for i in range(board_size):
        for j in range(1, board_size):
            # Connect rows
            edges.append(((i, j-1), (i, j)))
            # Connect columns
            edges.append(((j-1, i), (j, i)))
            # Connect "back-columns"
            if i < board_size - 1:
                edges.append(((i, j), (i+1, j-1)))


    # Make a list with the number of edges for each node
    # There is probably a better way to do this. I.e., with an adjacency matrix
    n_edges_list = []
    for i in range(board_size**2):
        # Top left and bottom right corner have 2 neighbors
        if i == 0 or i == board_size**2-1:
            n_edges_list.append(2)
        # Top right and bottom left corners have 3 neighbors
        elif i == board_size - 1 or i == board_size**2-board_size:
            n_edges_list.append(3)
        # Top and bottom edges excluding corners has 4 neighbors
        elif i // board_size == 0 or i // board_size == board_size - 1:
            n_edges_list.append(4)
        # The side nodes excluding corners also has 4 neighbors
        elif i % board_size == 0 or i % board_size == board_size-1:
            n_edges_list.append(4)
        # The interior nodes has 6 edges
        else:
            n_edges_list.append(6)


    # Helper function
    def position_to_edge_id(pos, board_size):
        return pos[0] * board_size + pos[1]

    log.info("Creating training data")
    graphs_train = Graphs(
        number_of_graphs=subset_size,
        symbols=symbol_names,
        hypervector_size=params.hypervector_size,
        hypervector_bits=params.hypervector_bits,
        double_hashing=params.double_hashing,
    )

    # Prepare nodes
    for graph_id in range(X.shape[0]):
        graphs_train.set_number_of_graph_nodes(
            graph_id=graph_id,
            number_of_graph_nodes=board_size**2,
        )
    graphs_train.prepare_node_configuration()

    # Prepare edges
    for graph_id in range(X.shape[0]):
        for k in range(board_size**2):
            graphs_train.add_graph_node(graph_id, k, n_edges_list[k])
    graphs_train.prepare_edge_configuration()

    # Create the graph
    for graph_id in range(X.shape[0]):
        for k in range(board_size**2):
            sym = X[graph_id][k]
            graphs_train.add_graph_node_property(graph_id, k, sym)
        # Loop through all edges
        for edge in edges:
            node_id = position_to_edge_id(edge[0], board_size)
            destination_node_id = position_to_edge_id(edge[1], board_size)
            graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type_name=0)
            graphs_train.add_graph_node_edge(graph_id, destination_node_id, node_id, edge_type_name=0)
    graphs_train.encode()

    # Test graph
    log.info("Creating test data")
    graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)

    # Prepare nodes
    for graph_id in range(X_test.shape[0]):
        graphs_test.set_number_of_graph_nodes(
            graph_id=graph_id,
            number_of_graph_nodes=board_size**2,
        )
    graphs_test.prepare_node_configuration()

    # Prepare edges
    for graph_id in range(X_test.shape[0]):
        for k in range(board_size**2):
            graphs_test.add_graph_node(graph_id, k, n_edges_list[k])
    graphs_test.prepare_edge_configuration()

    # Create the graph
    for graph_id in range(X_test.shape[0]):
        for k in range(board_size**2):
            sym = X_test[graph_id][k]
            graphs_test.add_graph_node_property(graph_id, k, sym)
             # Add edges
        for edge in edges:
            node_id = position_to_edge_id(edge[0], board_size)
            destination_node_id = position_to_edge_id(edge[1], board_size)
            graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type_name=0)
            graphs_test.add_graph_node_edge(graph_id, destination_node_id, node_id, edge_type_name=0)
        
    graphs_test.encode()

    # Train the Tsetlin Machine
    tm = MultiClassGraphTsetlinMachine(
        params.number_of_clauses,
        params.t,
        params.s,
        depth=params.depth,
        message_size=params.message_size,
        message_bits=params.message_bits,
        max_included_literals=params.max_included_literals,
        grid=(16*13,1,1),
        block=(128,1,1)
    )

    start_training = time()
    for i in range(params.epochs):
        tm.fit(graphs_train, y, epochs=1, incremental=True)
        log.info(f"Epoch#{i+1} -- Accuracy train: {np.mean(y == tm.predict(graphs_train))} -- Accuracy test: {np.mean(y_test == tm.predict(graphs_test))} ")
    stop_training = time()
    log.info(f"Time: {stop_training - start_training}")


    weights = tm.get_state()[1].reshape(2, -1)
    for i in range(tm.number_of_clauses):
        l = []
        for k in range(params.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < params.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - params.hypervector_size))
        log.info("Clause #%d W:(%d %d) " % (i, weights[0,i], weights[1,i]) + " AND ".join(l))
        log.info(f"Number of literals: {len(l)}")

    # After training is done and we have y_pred:
    training_time = stop_training - start_training

    # Measure memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

    # Measure inference time
    start_inference = time()
    y_pred = tm.predict(graphs_test)
    stop_inference = time()
    inference_time = stop_inference - start_inference

    accuracy = np.mean(y_test == y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results = Results(
        training_time_s=training_time,
        memory_usage_mb=memory_usage,
        inference_time_s=inference_time,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )

    cm = confusion_matrix(y_test, y_pred).tolist()
    weights = tm.get_state()[1].reshape(2, -1)
    
    log.info(f'results: {results}')
    log.info(f'confusion matrix: {cm}')
    return TrainOutput(results=results, confusion_matrix=cm, weights=weights)


if __name__ == '__main__':

    ## startup
    # parse options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True, help='dataset to use')
    parser.add_argument('--tag', '-t', help='append to name of run for clarity')

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
    tag = f'-{args.tag}' if args.tag else ''
    run_name = os.path.join('runs', datetime.now().strftime(f'%Y-%m-%dT%H:%M:%S{tag}'))
    os.makedirs(run_name)
    log_file = os.path.join(run_name, 'log.txt')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    logging.basicConfig(handlers=[file_handler, console_handler], level=logging.DEBUG, format='%(message)s')
    log.info(f'creating new run in {run_name}')

    # report which params used
    data = asdict(params)
    params_file = os.path.join(run_name, 'params.toml')
    log.info(f'reporting parameters in {params_file}, values:\n{data}')
    with open(params_file, 'w') as file:
        toml.dump(data, file)

    # load and report dataset
    log.info(f'loading dataset from {args.data}')
    dataset = pd.read_csv(args.data)
    first_row = dataset['board'].iloc[0]
    tiles = len(first_row)
    n = int(math.sqrt(tiles))

    report_filename = os.path.join(run_name, 'dataset.toml')
    log.info('hashing dataset')
    hasher = hashlib.blake2b()
    with open(args.data, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    hash = hasher.hexdigest()
    report = { 'filename': os.path.abspath(args.data), 'hash-blake2b': hash, 'dimensions': n, 'rows': len(dataset) }
    log.info(f'saving dataset report to {report_filename}, content:\n{report}')
    with open(report_filename, 'w') as file:
        toml.dump(report, file)

    ## main
    results = train(dataset, params, n)

    ## report
    results_file = os.path.join(run_name, 'results.toml')
    log.info(f'reporting {results_file}')
    with open(results_file, 'w') as file:
        data = asdict(results.results)
        toml.dump(data, file)

    cm_file = os.path.join(run_name, 'confusion_matrix.csv')
    log.info(f'reporting {cm_file}')
    np.savetxt(cm_file, results.confusion_matrix, delimiter=',', fmt='%d')

    weights_file = os.path.join(run_name, 'weights.csv')
    log.info(f'reporting {weights_file}')
    np.savetxt(weights_file, results.weights, delimiter=',')
