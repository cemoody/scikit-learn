from libc cimport math
cimport cython
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
cdef extern from "numpy/npy_math.h":
    float NPY_INFINITY


cdef float EPSILON_DBL = 1e-8
cdef float PERPLEXITY_TOLERANCE = 1e-5

@cython.boundscheck(True)
cpdef np.ndarray[np.float32_t, ndim=2] _binary_search_perplexity(
        np.ndarray[np.float32_t, ndim=2] affinities,
        float desired_perplexity,
        int verbose,
        int use_neighbors):
    """Binary search for sigmas of conditional Gaussians.


    Parameters
    ----------
    affinities : array-like, shape (n_samples, n_samples) or (n_samples, K)
        Distances between training samples. If this matrix is square, then
        compute the full pair-wise P(i|j) for all i and j. If this matrix is
        not square then we compute a restricted set of P(i|j). For example,
        we assume that affinities is defined such that affinities[i, 0] is
        distance to the nearest neighbor to point `i` and affinities[i, 1] is
        the next closest. By only considering the nearest neighbors, we reduce
        the computational complexity from O(N) to O(uN) where u `u` is the
        fraction of points considered to be neighbors. The memory consumption
        is also reduced. In practice, the number of nearest neighbors to
        include is a multiple of the perplexity, usually 3.

    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : array, shape (n_samples, n_samples)
        Probabilities of conditional Gaussian distributions p_i|j.
    """
    # Maximum number of binary search steps
    cdef long n_steps = 100

    cdef long n_samples = affinities.shape[0]
    # Precisions of conditional Gaussian distrubutions
    cdef float beta
    cdef float beta_min
    cdef float beta_max
    cdef float beta_sum = 0.0
    # Now we go to log scale
    cdef float desired_entropy = math.log(desired_perplexity)
    cdef float entropy_diff

    cdef float entropy
    cdef float sum_Pi
    cdef float sum_disti_Pi
    cdef long i, j, k, l = 0

    cdef long K = affinities.shape[1]

    # This array is later used as a 32bit array. It has multiple intermediate
    # floating point additions that benefit from the extra precision
    cdef np.ndarray[np.float64_t, ndim=2] P = np.zeros((n_samples, K),
                                                       dtype=np.float64)

    for i in range(n_samples):
        beta_min = -NPY_INFINITY
        beta_max = NPY_INFINITY
        beta = 1.0

        # Binary search of precision for i-th conditional distribution
        # Compute current entropy and corresponding probabilities
        # computed just over the nearest neighbors or over all data
        # if we're not using neighbors
        for l in range(n_steps):
            for j in range(K):
                P[i, j] = math.exp(-affinities[i, j] * beta)
            if not use_neighbors:
                P[i, i] = 0.0
            sum_Pi = 0.0
            for j in range(K):
                sum_Pi += P[i, j]
            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0
            for j in range(K):
                P[i, j] /= sum_Pi
                sum_disti_Pi += affinities[i, j] * P[i, j]
            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == NPY_INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -NPY_INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        beta_sum += beta

        if verbose and ((i + 1) % 1000 == 0 or i + 1 == n_samples):
            print("[t-SNE] Computed conditional probabilities for sample "
                  "%d / %d" % (i + 1, n_samples))

    if verbose:
        print("[t-SNE] Mean sigma: %f"
              % np.mean(math.sqrt(n_samples / beta_sum)))
    return P

def _uncompress_sparse(np.ndarray[np.int32_t, ndim=1] indices, 
                       np.ndarray[np.int32_t, ndim=1] indptr,
                       np.ndarray[np.float32_t, ndim=1] data, 
                       np.ndarray[np.float32_t, ndim=2] P, 
                       np.ndarray[np.int32_t, ndim=2] neighbors):
    # This expands a CSR matrix so that each row is a fixed size
    # and we can use normal dense numpy arrays
    cdef int rows = indices.shape[0]
    cdef int row, j, idx
    for row in range(rows):
        for j, idx in enumerate(range(indptr[row], indptr[row + 1])):
            P[row, j] = data[idx]
            neighbors[row, j] = indices[idx]
