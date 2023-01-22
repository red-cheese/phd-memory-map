

import numpy as np

from utils import *


# List of experiment specs: [ Exp 1: (mu0, sigma0, mu1, sigma1), ... ]
DISTRIB_PARAMS = [
    (
        # G1: (5.0, 0.1 * I), G2: (2.0, I) - very well separated Gaussians
        'exp0',
        np.full((INPUT_DIM,), 5., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64) * 0.1,
        np.full((INPUT_DIM,), 2., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64)
    ),

    (
        #
        'exp1',
        np.full((INPUT_DIM,), 2.5, dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64) * 0.2,
        np.full((INPUT_DIM,), 4., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64) * 0.2
    ),

    (
        #
        'exp2',
        np.full((INPUT_DIM,), 2.5, dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64) * 0.2,
        np.full((INPUT_DIM,), 4., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64) * 0.5
    ),

    (
        #
        'exp3',
        np.full((INPUT_DIM,), 3., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64),
        np.full((INPUT_DIM,), 3., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64) * 0.05
    ),

    (
        # Identical distributions - classifier should not learn anything.
        'exp4',
        np.full((INPUT_DIM,), 3., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64),
        np.full((INPUT_DIM,), 3., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64)
    ),
]
