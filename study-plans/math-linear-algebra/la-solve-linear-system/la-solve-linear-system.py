import numpy as np

def solve_linear_system(A: list, b: list) -> np.ndarray:
    """
    Returns the float64 solution vector.
    """
    matrix = np.copy(A).astype(np.float64)
    b = np.asarray(b, dtype=np.float64)

    return np.linalg.solve(matrix, b)
    