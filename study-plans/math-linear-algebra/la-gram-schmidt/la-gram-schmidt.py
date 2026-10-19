import numpy as np

def gram_schmidt(vectors: list) -> np.ndarray:
    """
    Returns a float64 array whose rows form the ordered orthonormal basis.
    """
    source = np.copy(vectors).astype(np.float64)
    basis = np.zeros_like(source, dtype=np.float64)

    n = source.shape[0]

    for i in range(n):
        residual = source[i].copy()
        for j in range(i):
            residual -= np.dot(residual, basis[j]) * basis[j]
        basis[i] = residual / np.linalg.norm(residual)
    return basis