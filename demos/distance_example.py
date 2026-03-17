"""
Example: Computing the distance between two datasets using MCOpt.

This script demonstrates how to use the MCOpt library to compute the
Gromov-Wasserstein (GW) distance and the Fused Gromov-Wasserstein (fGW)
distance between two metric measure networks built from simple graph data.

A MetricProbabilityNetwork (MPN) is a triple (V, p, W) where:
  - V  : the set of nodes (the "space")
  - p  : a probability measure over V  (weights that sum to 1)
  - W  : a pairwise distance matrix over V  (the "metric")

The GW distance compares two MPNs purely based on their structure (metric).
The fGW distance additionally takes per-node feature vectors into account.
"""

import numpy as np
from mcopt import MetricProbabilityNetwork
from mcopt.ot import GW, fGW

# ---------------------------------------------------------------------------
# Dataset 1 – a 3-node path graph  (0 – 1 – 2)
# ---------------------------------------------------------------------------
# Nodes
space1 = np.array([0, 1, 2])

# Uniform probability measure (each node has equal weight)
measure1 = np.ones(3) / 3          # [1/3, 1/3, 1/3]

# Shortest-path distance matrix
metric1 = np.array([
    [0, 1, 2],
    [1, 0, 1],
    [2, 1, 0],
], dtype=float)

# Optional per-node scalar features (e.g., a function value at each node)
features1 = np.array([0.0, 0.5, 1.0])

# ---------------------------------------------------------------------------
# Dataset 2 – a 4-node path graph  (0 – 1 – 2 – 3)
# ---------------------------------------------------------------------------
space2 = np.array([0, 1, 2, 3])

measure2 = np.ones(4) / 4          # [1/4, 1/4, 1/4, 1/4]

metric2 = np.array([
    [0, 1, 2, 3],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
    [3, 2, 1, 0],
], dtype=float)

features2 = np.array([0.1, 0.4, 0.6, 0.9])

# ---------------------------------------------------------------------------
# Build MetricProbabilityNetwork objects
# ---------------------------------------------------------------------------
X = MetricProbabilityNetwork(space1, measure1, metric1)
Y = MetricProbabilityNetwork(space2, measure2, metric2)

# ---------------------------------------------------------------------------
# Gromov-Wasserstein (GW) distance  –  structure only
# ---------------------------------------------------------------------------
# GW compares the internal geometry (metric) of two spaces by finding the
# optimal coupling between their probability measures.
coupling_gw, dist_gw = GW(X, Y)

print("=== Gromov-Wasserstein (GW) distance ===")
print(f"Distance: {dist_gw:.6f}")
print("Optimal coupling matrix (rows = X nodes, cols = Y nodes):")
print(np.array(coupling_gw))

# ---------------------------------------------------------------------------
# Fused Gromov-Wasserstein (fGW) distance  –  structure + features
# ---------------------------------------------------------------------------
# fGW combines the GW term (structural similarity) with a Wasserstein term
# (feature similarity).  The trade-off is controlled by alpha:
#   alpha = 1  →  pure GW  (structure only)
#   alpha = 0  →  pure Wasserstein  (features only)
#   alpha = 0.5  →  equal weight on structure and features

# Build the cross-domain feature cost matrix M[i, j] = |features1[i] - features2[j]|
M = np.abs(features1[:, None] - features2[None, :])

coupling_fgw, dist_fgw = fGW(X, Y, M, alpha=0.5)

print("\n=== Fused Gromov-Wasserstein (fGW) distance (alpha=0.5) ===")
print(f"Distance: {dist_fgw:.6f}")
print("Optimal coupling matrix (rows = X nodes, cols = Y nodes):")
print(np.array(coupling_fgw))
