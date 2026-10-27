"""Separate shape change from translation and rigid rotation in a vertex trajectory."""
import numpy as np


def shape_deformation(initial, current):
    reference = initial - initial.mean(axis=0)
    observed = current - current.mean(axis=0)
    # Kabsch alignment removes rigid motion without fitting scale or shear.
    u, _, vh = np.linalg.svd(reference.T @ observed)
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] *= -1
    aligned = reference @ (u @ vh)
    return float(np.linalg.norm(observed - aligned, axis=1).max())
