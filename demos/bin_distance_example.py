"""
Example: Computing the distance between two scalar fields stored in .bin files.

A ``.bin`` file is expected to hold a flat, headerless binary array of scalar
values in row-major (C) order.  The caller must know the grid shape and the
numeric dtype of the stored values (typically ``float32`` or ``float64``).

Two approaches are shown:

1. **Full Morse-complex pipeline** (requires TopologyToolKit / TTK)
   The scalar fields are treated as height functions on a 2-D plane.  Their
   Morse–Smale complexes are computed, simplified, and converted to
   ``MorseGraph`` objects.  Each graph is then represented as a
   ``MetricProbabilityNetwork`` (MPN) and the Gromov–Wasserstein (GW) or
   Fused Gromov–Wasserstein (fGW) distance is computed between the two MPNs.

2. **Lightweight path** (no TTK required)
   The raw scalar grids are used directly.  Grid points become nodes, the
   scalar values become a probability measure, and L1 geodesic distances on
   the grid become the metric.  This is a coarser approximation but runs
   without TTK.

Set ``USE_TTK = True`` to run the full pipeline (TTK must be installed).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Configuration – edit these to match your files
# ---------------------------------------------------------------------------

FILE1 = 'field1.bin'           # path to the first .bin file
FILE2 = 'field2.bin'           # path to the second .bin file
SHAPE = (100, 100)             # (rows, cols) of the 2-D scalar field
DTYPE = 'float32'              # numpy dtype of the stored values

PERSISTENCE_THRESHOLD = 0.1   # Morse-complex simplification threshold (TTK)
SAMPLE_RATE = 5                # graph down-sampling step (TTK)

USE_TTK = False                # set True if TTK is installed

# ---------------------------------------------------------------------------
# Utility: load a .bin file as a 2-D numpy array
# ---------------------------------------------------------------------------

def load_bin(file_path: str, shape: tuple, dtype: str = 'float32') -> np.ndarray:
    """Load a raw binary scalar-field file into a 2-D NumPy array.

    Parameters
    ----------
    file_path : str
        Path to the ``.bin`` file.
    shape : tuple of int
        ``(rows, cols)`` dimensions of the stored field.
    dtype : str
        NumPy dtype string, e.g. ``'float32'`` or ``'float64'``.

    Returns
    -------
    np.ndarray, shape ``shape``
    """
    data = np.fromfile(file_path, dtype=dtype)
    if data.size != shape[0] * shape[1]:
        raise ValueError(
            f"File '{file_path}' contains {data.size} values but "
            f"shape {shape} requires {shape[0] * shape[1]}."
        )
    return data.reshape(shape)


# ===========================================================================
# Approach 1 – Full TTK pipeline
# ===========================================================================

def compute_distance_ttk(file1, file2, shape, dtype='float32',
                          persistence_threshold=0.1, sample_rate=5):
    """Compute the GW distance between two scalar fields via Morse complexes.

    Requires TopologyToolKit (TTK) to be installed.

    Parameters
    ----------
    file1, file2 : str
        Paths to the two ``.bin`` files.
    shape : tuple of int
        ``(rows, cols)`` grid dimensions.
    dtype : str
        NumPy dtype of the stored values.
    persistence_threshold : float
        Persistence threshold used to simplify the Morse complex.
        Larger values → fewer critical points.
    sample_rate : int
        Step size used when down-sampling the separatrix graph.

    Returns
    -------
    coupling : mcopt.mm_space.Coupling
        Optimal transport coupling between the two graphs.
    dist : float
        GW distance between the two scalar fields.
    """
    import mcpipeline.vtk as vtk_util
    from mcpipeline.ttk import MorseComplex
    from mcopt.ot import GW

    # --- Load scalar fields ---
    data1 = load_bin(file1, shape, dtype)
    data2 = load_bin(file2, shape, dtype)

    # --- Wrap in VTK sources (height-function plane) ---
    src1 = vtk_util.PlaneSource(data1.astype(float))
    src2 = vtk_util.PlaneSource(data2.astype(float))

    # --- Compute Morse–Smale complexes ---
    mc1 = MorseComplex.create(src1.GetOutputPort(),
                               persistence_threshold=persistence_threshold)
    mc2 = MorseComplex.create(src2.GetOutputPort(),
                               persistence_threshold=persistence_threshold)

    # --- Convert to MorseGraphs (optionally down-sample) ---
    g1 = mc1.to_graph()
    g2 = mc2.to_graph()
    if sample_rate is not None:
        g1 = g1.sample(sample_rate)
        g2 = g2.sample(sample_rate)

    # --- Build MetricProbabilityNetworks ---
    # hist='degree'  weights each node by its degree (topology-aware)
    # dist='geo'     uses geodesic (Euclidean arc-length) edge weights
    mpn1 = g1.to_mpn(hist='degree', dist='geo')
    mpn2 = g2.to_mpn(hist='degree', dist='geo')

    # --- Compute GW distance ---
    coupling, dist = GW(mpn1, mpn2)
    return coupling, dist


# ===========================================================================
# Approach 2 – Lightweight path (no TTK required)
# ===========================================================================

def _build_mpn_from_grid(data: np.ndarray):
    """Build a MetricProbabilityNetwork from a 2-D scalar-field grid.

    Nodes are the grid points (flattened), the probability measure is
    proportional to the absolute scalar value at each point, and the metric
    is the L1 (Manhattan) distance between grid coordinates.

    Parameters
    ----------
    data : np.ndarray, shape (rows, cols)

    Returns
    -------
    mcopt.MetricMeasureNetwork.MetricProbabilityNetwork
    """
    from mcopt import MetricProbabilityNetwork

    rows, cols = data.shape
    n = rows * cols

    # Nodes are linear indices 0 … n-1
    space = np.arange(n)

    # Measure: normalised absolute scalar values (shift so all values ≥ 0)
    values = data.ravel().astype(float)
    values = values - values.min()
    total = values.sum()
    if total == 0:
        measure = np.ones(n) / n
    else:
        measure = values / total

    # Metric: L1 grid distance between node (r1,c1) and (r2,c2)
    # = |r1-r2| + |c1-c2| (normalised to [0,1])
    r_idx = space // cols
    c_idx = space % cols
    max_dist = float((rows - 1) + (cols - 1))
    metric = (
        np.abs(r_idx[:, None] - r_idx[None, :])
        + np.abs(c_idx[:, None] - c_idx[None, :])
    ) / max_dist

    return MetricProbabilityNetwork(space, measure, metric)


def compute_distance_lightweight(file1, file2, shape, dtype='float32'):
    """Compute the GW distance between two scalar fields without TTK.

    The raw grid values are used directly as a weighted metric space.
    This is faster but less topologically precise than the TTK pipeline.

    Parameters
    ----------
    file1, file2 : str
        Paths to the two ``.bin`` files.
    shape : tuple of int
        ``(rows, cols)`` grid dimensions.
    dtype : str
        NumPy dtype of the stored values.

    Returns
    -------
    coupling : mcopt.mm_space.Coupling
    dist : float
        GW distance between the two scalar fields.
    """
    from mcopt.ot import GW

    data1 = load_bin(file1, shape, dtype)
    data2 = load_bin(file2, shape, dtype)

    mpn1 = _build_mpn_from_grid(data1)
    mpn2 = _build_mpn_from_grid(data2)

    coupling, dist = GW(mpn1, mpn2)
    return coupling, dist


# ===========================================================================
# Demo entry point
# ===========================================================================

if __name__ == '__main__':
    import os

    # ------------------------------------------------------------------
    # For the demo we generate two tiny synthetic .bin files so the script
    # can be run without any pre-existing data.  Replace the FILE1/FILE2
    # paths at the top of this file to use your own data.
    # ------------------------------------------------------------------
    _demo_shape = (20, 20)
    rng = np.random.default_rng(0)

    _demo_file1 = '/tmp/demo_field1.bin'
    _demo_file2 = '/tmp/demo_field2.bin'

    field1 = rng.random(_demo_shape).astype('float32')
    field2 = rng.random(_demo_shape).astype('float32')
    field1.tofile(_demo_file1)
    field2.tofile(_demo_file2)

    print("Generated synthetic demo .bin files:")
    print(f"  {_demo_file1}  shape={_demo_shape}  dtype=float32")
    print(f"  {_demo_file2}  shape={_demo_shape}  dtype=float32")
    print()

    if USE_TTK:
        print("=== Full TTK pipeline ===")
        coupling, dist = compute_distance_ttk(
            _demo_file1, _demo_file2,
            shape=_demo_shape, dtype='float32',
            persistence_threshold=PERSISTENCE_THRESHOLD,
            sample_rate=SAMPLE_RATE,
        )
        print(f"GW distance (TTK): {dist:.6f}")
    else:
        print("=== Lightweight path (no TTK) ===")
        print("(Set USE_TTK = True at the top of this file to use the full "
              "Morse-complex pipeline.)")
        coupling, dist = compute_distance_lightweight(
            _demo_file1, _demo_file2,
            shape=_demo_shape, dtype='float32',
        )
        print(f"GW distance (lightweight): {dist:.6f}")
        print()
        print("Optimal coupling matrix:")
        print(np.array(coupling))
