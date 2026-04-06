import numpy as np


def sdf_sphere(p: np.ndarray, radius: float = 1.0) -> float:
    """Sphere signed distance function."""
    return float(np.linalg.norm(p) - radius)


def sdf_box(p: np.ndarray, size: np.ndarray) -> float:
    """Axis-aligned box signed distance function.

    Args:
        p: Query point (shape: [3]).
        size: Half-size of the box in each axis (shape: [3]).
    """
    q = np.abs(p) - size
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], max(q[1], q[2])), 0.0)
    return float(outside + inside)


if __name__ == "__main__":
    p = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    print("sdf_sphere:", sdf_sphere(p, radius=1.0))
    print("sdf_box:", sdf_box(p, size=np.array([0.6, 0.6, 0.6], dtype=np.float32)))
