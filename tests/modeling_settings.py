"""Fixed sizes, seeds and probe values used by the modeling checks."""

import numpy as np

TOLERANCE = 1e-9
POOLS_IN_SAMPLE = 4
BOOKED_POSITIONS: list[int] = [3, 5, 7]
SESSION_RANKS: list[float] = [1.0, 2.0, 30.0]
SMALL_SAMPLE = 200
LARGE_SAMPLE = 20000
PAIRED_SAMPLE = 500
RELATIVE_POOLS = 5
PROJECTION_ROWS = 500
PROJECTION_DIMENSION = 6
PROJECTION_SCALE = 5.0
PROJECTION_SEED = 3
DOMINANT_WEIGHT = 0.9
TOP_THREE = 3
COMPONENTS = 2
TOP_TEN = 10
LOW_SCORES = np.array([0.1, 0.2, 0.8, 0.9])
HIGH_SCORES = np.array([0.9, 0.8, 0.2, 0.1])
AUC_LABELS = np.array([False, False, True, True])
WEIGHT_MARGINS = np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.2])
WEIGHT_PROPENSITIES = np.array([0.5, 0.25, 0.4, 0.6, 0.3, 0.2])
WEIGHT_MEAN_TOLERANCE = 0.5
