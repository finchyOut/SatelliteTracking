import numpy as np
import numpy as np
from scipy.spatial.distance import pdist

#gets the dot product between two lists
def dotKernel(x, y, n =0, m=0, sigma = 0):
    dot_product = np.dot(x, y)
    return dot_product

#Applies the name of kernel
def applyKernel(kern, x, y, n=0, m =0, sigma =0):
    return kern(x,y, n, m, sigma)

#gets the median distance between the two
def median_heuristic_sigma2(X, Y):
    """
    Compute σ^2 via pooled median heuristic on the combined dataset Z = [X; Y].
    X, Y: arrays of shape (n_samples, n_features)
    Returns a float (sigma^2).
    """
    Z = np.vstack([np.asarray(X), np.asarray(Y)])
    dists = pdist(Z, metric="euclidean")        # condensed vector of pairwise distances
    sigma = np.median(dists)
    # guard tiny values to avoid division by zero
    if sigma <= 0:
        sigma = 1e-12
    return float(sigma**2)

def quantile_heuristic_sigma2(X, Y, q=0.5):
    Z = np.vstack([np.asarray(X), np.asarray(Y)])
    dists = pdist(Z, metric="euclidean")
    sigma = np.quantile(dists, q)
    if sigma <= 0:
        sigma = 1e-12
    return float(sigma**2)

#computes the gaussian kernel, uses the median kernel if sigma not identified
def gaussianKernel(x, y, sigma):
    """
    RBF kernel: k(x,y) = exp(-||x - y||^2 / (2 * sigma2))
    x, y: 1D arrays
    sigma2: scalar (σ^2)
    """
    x = np.asarray(x); y = np.asarray(y)
    sq_dist = np.sum((x - y)**2)
    return np.exp(-sq_dist / (2.0 * sigma**2))

#computes the laplacian kernel
def laplacianKernel(x, y, sigma):
    """
    Laplacian (L1) kernel:
        k(x, y) = exp(- ||x - y||_1 / sigma)
    where || · ||_1 is the L1 / Manhattan norm.

    Args:
        x, y : 1D arrays of equal length
        sigma: > 0, length scale (note: scikit-learn often uses gamma = 1/sigma)

    Returns:
        float kernel value in (0, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    l1 = np.linalg.norm(x - y, ord=1)
    return float(np.exp(-l1 / sigma))

#computes the cosine kernel
def cosineKernel(x, y, sigma =0):
    """
    Cosine similarity kernel:
    k(x,y) = (x^T y) / (||x|| * ||y||)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0  # define similarity with zero vector as 0
    
    return dot / (norm_x * norm_y) 

#computes the unbiased mmd calculation with specified kernel
def MMD(kern, X, Y, sigma = 0):
    """
    This returns MMD squared. The MMD package via ignite returns just MMD
    Unbiased MMD^2(P,Q;k) with k parameterized by sigma2 when needed.
    X, Y: arrays of shape (B, D). (For parity with Ignite, use same B in a batch.)
    kern: callable (x, y, sigma2) -> scalar
    sigma2: scalar passed to the kernel
    """

    X = np.asarray(X); Y = np.asarray(Y)
    n, m = len(X), len(Y)
    if sigma == 0:
        sigma = quantile_heuristic_sigma2(X, Y)
    if n < 2 or m < 2:
        raise ValueError("Unbiased MMD^2 requires n>=2 and m>=2")
    # within X (exclude diagonal)
    s_x = 0.0
    for i in range(n):
        for j in range(n):
            if j != i:
                s_x += kern(X[i], X[j], sigma)
    Finals_x = s_x / (n * (n - 1))

    # within Y (exclude diagonal)
    s_y = 0.0
    for i in range(m):
        for j in range(m):
            if j != i:
                s_y += kern(Y[i], Y[j], sigma)
    Finals_y = s_y / (m * (m - 1))

    # cross term: all pairs, factor 2/(nm)
    s_xy = 0.0
    for i in range(n):
        for j in range(m):
            s_xy += kern(X[i], Y[j], sigma)
    Finals_xy = s_xy*(2.0 / (n * m))

    return Finals_x + Finals_y - Finals_xy


"""
x = [[-0.80324818, -0.95768364, -0.03807209],
                [-0.11059691, -0.38230813, -0.4111988],
                [-0.8864329, -0.02890403, -0.60119252],
                [-0.68732452, -0.12854739, -0.72095073],
                [-0.62604613, -0.52368328, -0.24112842]]
y =[[0.0686768, 0.80502737, 0.53321717],
                [0.83849465, 0.59099726, 0.76385441],
                [0.68688272, 0.56833803, 0.98100778],
                [0.55267761, 0.13084654, 0.45382906],
                [0.0754253, 0.70317304, 0.4756805]]
distance = MMD(gaussianKernel, x, y)
print(distance)

"""