# MCOpt

## Requirements

Beyond the dependencies listed in `setup.cfg`, this project also requires 
[topologytoolkit](https://topology-tool-kit.github.io/) be installed inorder to 
generate morse complexes.

## Usage
1. **Setup a virtual environment**
```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

2. **Install the package**
```bash
pip install -e .
```

3. **Computing the distance between two datasets**

A `MetricProbabilityNetwork` (MPN) is a triple *(V, p, W)* where *V* is the
set of nodes, *p* is a probability measure over *V* (weights summing to 1),
and *W* is a pairwise distance matrix over *V*.

```python
import numpy as np
from mcopt import MetricProbabilityNetwork
from mcopt.ot import GW, fGW

# --- Dataset 1: 3-node path graph (0 – 1 – 2) ---
space1   = np.array([0, 1, 2])
measure1 = np.ones(3) / 3          # uniform weights
metric1  = np.array([[0, 1, 2],
                     [1, 0, 1],
                     [2, 1, 0]], dtype=float)
features1 = np.array([0.0, 0.5, 1.0])  # optional per-node scalar features

# --- Dataset 2: 4-node path graph (0 – 1 – 2 – 3) ---
space2   = np.array([0, 1, 2, 3])
measure2 = np.ones(4) / 4
metric2  = np.array([[0, 1, 2, 3],
                     [1, 0, 1, 2],
                     [2, 1, 0, 1],
                     [3, 2, 1, 0]], dtype=float)
features2 = np.array([0.1, 0.4, 0.6, 0.9])

X = MetricProbabilityNetwork(space1, measure1, metric1)
Y = MetricProbabilityNetwork(space2, measure2, metric2)

# Gromov-Wasserstein distance (structure only)
coupling_gw, dist_gw = GW(X, Y)
print("GW distance:", dist_gw)

# Fused Gromov-Wasserstein distance (structure + node features, alpha=0.5)
M = np.abs(features1[:, None] - features2[None, :])  # feature cost matrix
coupling_fgw, dist_fgw = fGW(X, Y, M, alpha=0.5)
print("fGW distance:", dist_fgw)
```

See `demos/distance_example.py` for the full runnable version.

## Experiments
It is recommended that you use the included [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers)
to run the experiments. In order to run all experiments, run the following
```bash
cd experiments; python pipeline.py
```