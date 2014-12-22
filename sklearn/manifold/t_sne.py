# Author: Alexander Fabisch  -- <afabisch@informatik.uni-bremen.de>
# License: BSD 3 clause (C) 2014

# This is the standard t-SNE implementation. There are faster modifications of
# the algorithm:
# * Barnes-Hut-SNE: reduces the complexity of the gradient computation from
#   N^2 to N log N (http://arxiv.org/abs/1301.3342)
# * Fast Optimization for t-SNE:
#   http://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf

import numpy as np
import math
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from ..base import BaseEstimator
from ..externals.joblib import Parallel, delayed
from ..utils import check_array
from ..utils import check_random_state
from ..utils.extmath import _ravel
from ..decomposition import RandomizedPCA
from ..metrics.pairwise import pairwise_distances
from . import _utils
from . import bhtsne


MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : array, shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose)
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P


def _kl_divergence(params, P, alpha, n_samples, n_components):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.

    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    alpha : float
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    n = pdist(X_embedded, "sqeuclidean")
    n += 1.
    n /= alpha
    n **= (alpha + 1.0) / -2.0
    Q = np.maximum(n / (2.0 * np.sum(n)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    kl_divergence = 2.0 * np.dot(P, np.log(P / Q))

    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components))
    PQd = squareform((P - Q) * n)
    for i in range(n_samples):
        np.dot(_ravel(PQd[i]), X_embedded[i] - X_embedded, out=grad[i])
    grad = grad.ravel()
    c = 2.0 * (alpha + 1.0) / alpha
    grad *= c

    return kl_divergence, grad


def _kl_divergence_error(params, P, alpha, n_samples, n_components):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.

    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    alpha : float
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    n = pdist(X_embedded, "sqeuclidean")
    n += 1.
    n /= alpha
    n **= (alpha + 1.0) / -2.0
    Q = np.maximum(n / (2.0 * np.sum(n)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if len(P.shape) == 2:
        P = squareform(P)
    kl_divergence = 2.0 * np.dot(P, np.log(P / Q))

    return kl_divergence


def _kl_divergence_bh(params, P, alpha, n_samples, n_components, theta=0.5,
                      verbose=False, threadpool=None):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.
    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2)

    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.

    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    alpha : float
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    threadpool : joblib.Parallel object
        Use this threadpool to launch many threads

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components).astype(np.float32)

    if len(P.shape) == 1:
        sP = squareform(P).astype(np.float32)
    else:
        sP = P

    width = X_embedded.max(axis=0) - X_embedded.min(axis=0)
    sum_Q = np.zeros(1, dtype=np.float32)
    if threadpool:
        n_jobs = threadpool.n_jobs
        if verbose > 15:
            print("[t-SNE] Running in parallel with %i threads" % n_jobs)
        pos_f = np.zeros(X_embedded.shape, dtype=np.float32)
        args = (width.astype(np.float32), sP.astype(np.float32),
                X_embedded.astype(np.float32), pos_f.astype(np.float32),
                theta, verbose)
        bhtsne.gradient_positive(*args)
        grad_delayed = delayed(bhtsne.gradient_negative)
        start, stop = 0, 0
        n = sP.shape[0]
        step = int(math.ceil(n * 1.0 / n_jobs))
        jobs = []
        neg_fs = []
        for j in range(n_jobs):
            stop += step
            stop = min(stop, n)
            neg_f = np.zeros(X_embedded.shape, dtype=np.float32)
            neg_fs.append(neg_f)
            args = (width.astype(np.float32), sP.astype(np.float32),
                    X_embedded.astype(np.float32), neg_f.astype(np.float32),
                    theta, start, stop, verbose)
            jobs.append(grad_delayed(*args))
            start = stop
        # Launch all of the jobs on separate threads
        neg_f = np.vstack(neg_f)
        # Note that gradient_negative updates the grad array inplace
        # as well as returning sum_Q
        sum_Q = threadpool(jobs)
        sum_Q = np.sum(sum_Q)
        grad = pos_f - (neg_f / sum_Q)
    else:
        grad = np.zeros(X_embedded.shape, dtype=np.float32)
        bhtsne.gradient(width, sP, X_embedded, grad, theta, verbose)

    c = 2.0 * (alpha + 1.0) / alpha
    grad = grad.ravel()
    grad *= c

    return None, grad


def _gradient_descent(objective, p0, it, n_iter, objective_error=None,
                      n_iter_check=1, n_iter_without_progress=50,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                      args=[], kwargs={}):
    """Batch gradient descent with momentum and individual gains.

    Parameters
    ----------
    objective : function or callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.

    p0 : array-like, shape (n_params,)
        Initial parameter vector.

    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).

    n_iter : int
        Maximum number of gradient descent iterations.

    n_iter_check : int
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.

    objective_error : function or callable
        Should return a tuple of cost and gradient for a given parameter
        vector.

    n_iter_without_progress : int, optional (default: 30)
        Maximum number of iterations without progress before we abort the
        optimization.

    momentum : float, within (0.0, 1.0), optional (default: 0.5)
        The momentum generates a weight for previous gradients that decays
        exponentially.

    learning_rate : float, optional (default: 1000.0)
        The learning rate should be extremely high for t-SNE! Values in the
        range [100.0, 1000.0] are common.

    min_gain : float, optional (default: 0.01)
        Minimum individual gain for each parameter.

    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be aborted.

    min_error_diff : float, optional (default: 1e-7)
        If the absolute difference of two successive cost function values
        is below this threshold, the optimization will be aborted.

    verbose : int, optional (default: 0)
        Verbosity level.

    args : sequence
        Arguments to pass to objective function.

    kwargs : dict
        Keyword arguments to pass to objective function.

    Returns
    -------
    p : array, shape (n_params,)
        Optimum parameters.

    error : float
        Optimum.

    i : int
        Last iteration.
    """
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = 0

    for i in range(it, n_iter):
        new_error, grad = objective(p, *args, **kwargs)

        inc = update * grad >= 0.0
        dec = np.invert(inc)
        gains[inc] += 0.05
        gains[dec] *= 0.95
        np.clip(gains, min_gain, np.inf)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if (i+1) % n_iter_check == 0:
            grad_norm = linalg.norm(grad)
            if new_error is None:
                new_error = objective_error(p, *args)
            error_diff = np.abs(new_error - error)
            error = new_error

            if verbose >= 2:
                m = "[t-SNE] Iteration %d: error = %.7f, gradient norm = %.7f"
                print(m % (i + 1, error, grad_norm))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break
            if error_diff <= min_error_diff:
                if verbose >= 2:
                    m = "[t-SNE] Iteration %d: error difference %f. Finished."
                    print(m % (i + 1, error_diff))
                break

        if new_error is not None:
            error = new_error

    return p, error, i


def trustworthiness(X, X_embedded, n_neighbors=5, precomputed=False):
    """Expresses to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in U^{(k)}_i (r(i, j) - k)}

    where :math:`r(i, j)` is the rank of the embedded datapoint j
    according to the pairwise distances between the embedded datapoints,
    :math:`U^{(k)}_i` is the set of points that are in the k nearest
    neighbors in the embedded space but not in the original space.

    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten

    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : array, shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, optional (default: 5)
        Number of neighbors k that will be considered.

    precomputed : bool, optional (default: False)
        Set this flag if X is a precomputed square distance matrix.

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    if precomputed:
        dist_X = X
    else:
        dist_X = pairwise_distances(X, squared=True)
    dist_X_embedded = pairwise_distances(X_embedded, squared=True)
    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1]

    n_samples = X.shape[0]
    t = 0.0
    ranks = np.zeros(n_neighbors)
    for i in range(n_samples):
        for j in range(n_neighbors):
            ranks[j] = np.where(ind_X[i] == ind_X_embedded[i, j])[0][0]
        ranks -= n_neighbors
        t += np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t


class TSNE(BaseEstimator):
    """t-distributed Stochastic Neighbor Embedding.

    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.

    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selcting a value
        between 5 and 50. The choice is not extremely critical since t-SNE
        is quite insensitive to this parameter.

    early_exaggeration : float, optional (default: 4.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float, optional (default: 1000)
        The learning rate can be a critical parameter. It should be
        between 100 and 1000. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high. If the cost function gets stuck in a bad local
        minimum increasing the learning rate helps sometimes.

    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at
        least 200.

    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    init : string, optional (default: "random")
        Initialization of embedding. Possible options are 'random' and 'pca'.
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton. Note that different initializations
        might result in different local minima of the cost function.

    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.

    training_data_ : array-like, shape (n_samples, n_features)
        Stores the training data.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = TSNE(n_components=2, random_state=0)
    >>> model.fit_transform(X) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([[  887.28...,   238.61...],
           [ -714.79...,  3243.34...],
           [  957.30..., -2505.78...],
           [-1130.28...,  -974.78...])

    References
    ----------

    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        http://homepage.tudelft.nl/19j49/t-SNE.html
    """
    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', n_jobs=1):
        if init not in ["pca", "random"]:
            raise ValueError("'init' must be either 'pca' or 'random'")
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.n_jobs = n_jobs

    def _fit(self, X):
        """Fit the model using X as training data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is "
                             "%f" % self.early_exaggeration)

        if self.n_iter < 200:
            raise ValueError("n_iter should be at least 200")

        if self.metric == "precomputed":
            if self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be used "
                                 "with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")
            distances = X
        else:
            if self.verbose:
                print("[t-SNE] Computing pairwise distances...")
            if self.metric == "euclidean":
                distances = pairwise_distances(X, metric=self.metric,
                                               squared=True)
            else:
                distances = pairwise_distances(X, metric=self.metric)

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # alpha = n_components - 1 comes from "Learning a Parametric Embedding
        # by Preserving Local Structure" Laurens van der Maaten, 2009.
        alpha = self.n_components - 1.0
        n_samples = X.shape[0]
        self.training_data_ = X

        P = _joint_probabilities(distances, self.perplexity, self.verbose)
        if self.init == 'pca':
            pca = RandomizedPCA(n_components=self.n_components,
                                random_state=random_state)
            X_embedded = pca.fit_transform(X)
        elif self.init == 'random':
            X_embedded = None
        else:
            raise ValueError("Unsupported initialization scheme: %s"
                             % self.init)

        self.embedding_ = self._tsne(P, alpha, n_samples, random_state,
                                     X_embedded=X_embedded)

    def _tsne(self, P, alpha, n_samples, random_state, X_embedded=None):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with three stages:
        # * early exaggeration with momentum 0.5
        # * early exaggeration with momentum 0.8
        # * final optimization with momentum 0.8
        # The embedding is initialized with iid samples from Gaussians with
        # standard deviation 1e-4.

        if X_embedded is None:
            # Initialize embedding randomly
            X_embedded = 1e-4 * random_state.randn(n_samples,
                                                   self.n_components)
        params = X_embedded.ravel()

        kwargs = {}
        kwargs['n_iter'] = 50
        kwargs['momentum'] = 0.5
        kwargs['learning_rate'] = self.learning_rate
        kwargs['verbose'] = self.verbose
        kwargs['it'] = 0
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            objective_error = _kl_divergence_error
            sP = squareform(P).astype(np.float32)
            kwargs['args'] = [sP, alpha, n_samples, self.n_components]
            kwargs['min_grad_norm'] = 1e-3
            kwargs['n_iter_without_progress'] = 30
            if self.n_jobs > 1 or self.n_jobs == -1:
                tp = Parallel(n_jobs=self.n_jobs, backend="threading")
                kwargs['kwargs'] = dict(threadpool=tp, verbose=self.verbose)
            # Don't always calculate the cost since that calculation
            # can be nearly as expensive as the gradient
            kwargs['n_iter_check'] = 25
            kwargs['objective_error'] = objective_error
        else:
            obj_func = _kl_divergence
            kwargs['args'] = [P, alpha, n_samples, self.n_components]
            kwargs['min_error_diff'] = 0.0
            kwargs['min_grad_norm'] = 0.0

        # Early exaggeration
        P *= self.early_exaggeration
        params, error, it = _gradient_descent(obj_func, params, **kwargs)
        kwargs['n_iter'] = 100
        kwargs['momentum'] = 0.8
        kwargs['it'] += it
        params, error, it = _gradient_descent(obj_func, params, **kwargs)
        if self.verbose:
            print("[t-SNE] Error after %d iterations with early "
                  "exaggeration: %f" % (it + 1, error))

        # Final optimization
        kwargs['n_iter'] = self.n_iter
        kwargs['it'] += it
        params, error, it = _gradient_descent(obj_func, params, **kwargs)
        if self.verbose:
            print("[t-SNE] Error after %d iterations: %f" % (it + 1, error))

        X_embedded = params.reshape(n_samples, self.n_components)

        return X_embedded

    def fit_transform(self, X):
        """Transform X to the embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self._fit(X)
        return self.embedding_
