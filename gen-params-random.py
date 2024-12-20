#!/usr/bin/env -S poetry run python
import toml
import sys
import random

def sample_hyperparameters():
    """
    Sample hyperparameters from predefined distributions.
    Returns:
        dict: A dictionary of hyperparameters ready for TOML serialization.
    """
    # Random distributions for hyperparameters
    epochs = random.choice([50, 100, 150])
    number_of_clauses = random.choice([100, 200, 500])
    t = random.choice([100, 200, 400, 800])
    s = random.uniform(0.8, 3.0)  # float from uniform distribution
    depth = random.choice([3, 5, 7])
    hypervector_size = random.choice([256, 512, 1024])
    hypervector_bits = random.choice([1, 2, 4])
    message_size = random.choice([256, 512, 1024])
    message_bits = random.choice([1, 2, 4])
    double_hashing = random.choice([True, False])
    max_included_literals = random.choice([8, 16, 32])

    return {
        "epochs": epochs,
        "number_of_clauses": number_of_clauses,
        "t": t,
        "s": s,
        "depth": depth,
        "hypervector_size": hypervector_size,
        "hypervector_bits": hypervector_bits,
        "message_size": message_size,
        "message_bits": message_bits,
        "double_hashing": double_hashing,
        "max_included_literals": max_included_literals
    }


if __name__ == "__main__":
    params = sample_hyperparameters()
    sys.stdout.write(toml.dumps(params))
