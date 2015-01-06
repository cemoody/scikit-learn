import sys
import warnings
from sklearn.externals.six.moves import cStringIO as StringIO
import numpy as np
import scipy.sparse as sp
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raises_regexp
from sklearn.utils import check_random_state
from sklearn.manifold.t_sne import _joint_probabilities
from sklearn.manifold.t_sne import _kl_divergence
from sklearn.manifold.t_sne import _gradient_descent
from sklearn.manifold.t_sne import trustworthiness
from sklearn.manifold.t_sne import TSNE
from sklearn.manifold import _barnes_hut_tsne
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.manifold._utils import _binary_search_perplexity_nn
from scipy.optimize import check_grad
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances


def test_gradient_descent_stops():
    """Test stopping conditions of gradient descent."""
    class ObjectiveSmallGradient:
        def __init__(self):
            self.it = -1

        def __call__(self, _):
            self.it += 1
            return (10 - self.it) / 10.0, np.array([1e-5])

    def flat_function(_):
        return 0.0, np.ones(1)

    # Gradient norm
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(
            ObjectiveSmallGradient(), np.zeros(1), 0, n_iter=100,
            n_iter_without_progress=100, momentum=0.0, learning_rate=0.0,
            min_gain=0.0, min_grad_norm=1e-5, min_error_diff=0.0, verbose=2)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert_equal(error, 1.0)
    assert_equal(it, 0)
    assert("gradient norm" in out)

    # Error difference
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(
            ObjectiveSmallGradient(), np.zeros(1), 0, n_iter=100,
            n_iter_without_progress=100, momentum=0.0, learning_rate=0.0,
            min_gain=0.0, min_grad_norm=0.0, min_error_diff=0.2, verbose=2)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert_equal(error, 0.9)
    assert_equal(it, 1)
    assert("error difference" in out)

    # Maximum number of iterations without improvement
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(
            flat_function, np.zeros(1), 0, n_iter=100,
            n_iter_without_progress=10, momentum=0.0, learning_rate=0.0,
            min_gain=0.0, min_grad_norm=0.0, min_error_diff=-1.0, verbose=2)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert_equal(error, 0.0)
    assert_equal(it, 11)
    assert("did not make any progress" in out)

    # Maximum number of iterations
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(
            ObjectiveSmallGradient(), np.zeros(1), 0, n_iter=11,
            n_iter_without_progress=100, momentum=0.0, learning_rate=0.0,
            min_gain=0.0, min_grad_norm=0.0, min_error_diff=0.0, verbose=2)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert_equal(error, 0.0)
    assert_equal(it, 10)
    assert("Iteration 10" in out)


def test_binary_search():
    """Test if the binary search finds Gaussians with desired perplexity."""
    random_state = check_random_state(0)
    distances = random_state.randn(50, 2)
    # Distances shouldn't be negative
    distances = np.abs(distances.dot(distances.T))
    np.fill_diagonal(distances, 0.0)
    desired_perplexity = 25.0
    P = _binary_search_perplexity(distances, desired_perplexity, verbose=0)
    P = np.maximum(P, np.finfo(np.double).eps)
    mean_perplexity = np.mean([np.exp(-np.sum(P[i] * np.log(P[i])))
                               for i in range(P.shape[0])])
    assert_almost_equal(mean_perplexity, desired_perplexity, decimal=3)


def test_binary_search_neighbors():
    """Test if the binary perplexity search is approximately equal to the
       slow method when the slow method uses all points as nighbors"""
    n_samples = 500
    desired_perplexity = 25.0
    random_state = check_random_state(0)
    distances = random_state.randn(n_samples, 2)
    # Distances shouldn't be negative
    distances = np.abs(distances.dot(distances.T))
    np.fill_diagonal(distances, 0.0)
    P1 = _binary_search_perplexity(distances, desired_perplexity,
                                   verbose=0)

    # Test that when we use all the neighbors the results are identical
    k = n_samples
    neighbors_nn = np.argsort(distances, axis=1)[:, :k]
    P2 = _binary_search_perplexity_nn(distances, neighbors_nn,
                                      desired_perplexity, verbose=0)
    assert_array_almost_equal(P1, P2, decimal=6)

    # Test that the highest P_ij are the same when few neighbors are used
    for k in np.linspace(80, n_samples, 10):
        k = int(k)
        topn = k * 10  # check the top 10 *k entries out of k * k entries
        neighbors_nn = np.argsort(distances, axis=1)[:, :k]
        P2k = _binary_search_perplexity_nn(distances, neighbors_nn,
                                           desired_perplexity, verbose=0)
        idx = np.argsort(P1.ravel())[::-1]
        P1top = P1.ravel()[idx][:topn]
        P2top = P2k.ravel()[idx][:topn]
        assert_array_almost_equal(P1top, P2top, decimal=2)


def test_gradient():
    """Test gradient of Kullback-Leibler divergence."""
    random_state = check_random_state(0)

    n_samples = 50
    n_features = 2
    n_components = 2
    alpha = 1.0

    distances = random_state.randn(n_samples, n_features)
    distances = distances.dot(distances.T)
    np.fill_diagonal(distances, 0.0)
    X_embedded = random_state.randn(n_samples, n_components)

    P = _joint_probabilities(distances, desired_perplexity=25.0,
                             verbose=0)
    fun = lambda params: _kl_divergence(params, P, alpha, n_samples,
                                        n_components)[0]
    grad = lambda params: _kl_divergence(params, P, alpha, n_samples,
                                         n_components)[1]
    assert_almost_equal(check_grad(fun, grad, X_embedded.ravel()), 0.0,
                        decimal=5)


def test_trustworthiness():
    """Test trustworthiness score."""
    random_state = check_random_state(0)

    # Affine transformation
    X = random_state.randn(100, 2)
    assert_equal(trustworthiness(X, 5.0 + X / 10.0), 1.0)

    # Randomly shuffled
    X = np.arange(100).reshape(-1, 1)
    X_embedded = X.copy()
    random_state.shuffle(X_embedded)
    assert_less(trustworthiness(X, X_embedded), 0.6)

    # Completely different
    X = np.arange(5).reshape(-1, 1)
    X_embedded = np.array([[0], [2], [4], [1], [3]])
    assert_almost_equal(trustworthiness(X, X_embedded, n_neighbors=1), 0.2)


def test_preserve_trustworthiness_approximately():
    """Nearest neighbors should be preserved approximately."""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    # The Barnes-Hut approximation uses a different method to estimate
    # P_ij using only a a number of nearest neighbors instead of all
    # particles (so that k = 3 * perplexity). As a result we set the
    # perplexity=5, so that the number of neighbors is 5%.
    methods = ['standard', 'barnes_hut']
    for init in ('random', 'pca'):
        for method in methods:
            tsne = TSNE(n_components=2, perplexity=2, learning_rate=100.0,
                        init=init, random_state=0, method=method)
            X_embedded = tsne.fit_transform(X)
            T = trustworthiness(X, X_embedded, n_neighbors=1),
            assert_almost_equal(T, 1.0, decimal=1)


def test_fit_csr_matrix():
    """X can be a sparse matrix."""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    X[(np.random.randint(0, 100, 50), np.random.randint(0, 2, 50))] = 0.0
    X_csr = sp.csr_matrix(X)
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=100.0,
                random_state=0, method='standard')
    X_embedded = tsne.fit_transform(X_csr)
    assert_almost_equal(trustworthiness(X_csr, X_embedded, n_neighbors=1), 1.0,
                        decimal=1)


def test_preserve_trustworthiness_approximately_with_precomputed_distances():
    """Nearest neighbors should be preserved approximately."""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    D = squareform(pdist(X), "sqeuclidean")
    tsne = TSNE(n_components=2, perplexity=2, learning_rate=100.0,
                metric="precomputed", random_state=0, verbose=0)
    X_embedded = tsne.fit_transform(D)
    assert_almost_equal(trustworthiness(D, X_embedded, n_neighbors=1,
                                        precomputed=True), 1.0, decimal=1)


def test_early_exaggeration_too_small():
    """Early exaggeration factor must be >= 1."""
    tsne = TSNE(early_exaggeration=0.99)
    assert_raises_regexp(ValueError, "early_exaggeration .*",
                         tsne.fit_transform, np.array([[0.0]]))


def test_too_few_iterations():
    """Number of gradient descent iterations must be at least 200."""
    tsne = TSNE(n_iter=199)
    assert_raises_regexp(ValueError, "n_iter .*", tsne.fit_transform,
                         np.array([[0.0]]))


def test_non_square_precomputed_distances():
    """Precomputed distance matrices must be square matrices."""
    tsne = TSNE(metric="precomputed")
    assert_raises_regexp(ValueError, ".* square distance matrix",
                         tsne.fit_transform, np.array([[0.0], [1.0]]))


def test_init_not_available():
    """'init' must be 'pca', 'random' or a NumPy array"""
    m = "'init' must be 'pca', 'random' or a NumPy array"
    assert_raises_regexp(ValueError, m, TSNE, init="not available")


def test_distance_not_available():
    """'metric' must be valid."""
    tsne = TSNE(metric="not available")
    assert_raises_regexp(ValueError, "Unknown metric not available.*",
                         tsne.fit_transform, np.array([[0.0], [1.0]]))


def test_pca_initialization_not_compatible_with_precomputed_kernel():
    """Precomputed distance matrices must be square matrices."""
    tsne = TSNE(metric="precomputed", init="pca")
    assert_raises_regexp(ValueError, "The parameter init=\"pca\" cannot be "
                         "used with metric=\"precomputed\".",
                         tsne.fit_transform, np.array([[0.0], [1.0]]))


def test_answer_gradient_two_particles():
    """ This case with two particles test the tree with only a single
        set of children.

        These tests & answers have been checked against
        the reference implementation by LvdM
    """
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0]])
    pos_output = np.array([[-4.961291e-05, -1.072243e-04],
                           [9.259460e-05, 2.702024e-04]])
    neighbors = np.array([[1],
                          [0]])
    grad_output = np.array([[-2.37012478e-05, -6.29044398e-05],
                            [2.37012478e-05, 6.29044398e-05]])
    yield _run_answer_test, pos_input, pos_output, neighbors, grad_output


def test_answer_gradient_four_particles():
    """ This case with two particles test the tree with
        multiple levels of children

        These tests & answers have been checked against
        the reference implementation by LvdM
    """
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0],
                          [5.0, 2.0], [7.3, 2.2]])
    pos_output = np.array([[6.080564e-05, -7.120823e-05],
                           [-1.718945e-04, -4.000536e-05],
                           [-2.271720e-04, 8.663310e-05],
                           [-1.032577e-04, -3.582033e-05]])
    neighbors = np.array([[1, 2, 3],
                          [0, 2, 3],
                          [1, 0, 3],
                          [1, 2, 0]])
    grad_output = np.array([[5.81128448e-05, -7.78033454e-06],
                            [-5.81526851e-05, 7.80976444e-06],
                            [4.24275173e-08, -3.69569698e-08],
                            [-2.58720939e-09, 7.52706374e-09]])
    yield _run_answer_test, pos_input, pos_output, neighbors, grad_output


def test_skip_num_points_gradient():
    """ Skip num points should make it such that the Barnes_hut gradient
        is not calculated for indices below skip_num_point.
    """
    # Aside from skip_num_points=2 and the first two gradient rows
    # being set to zero, these data points are the same as in
    # test_answer_gradient_four_particles()
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0],
                          [5.0, 2.0], [7.3, 2.2]])
    pos_output = np.array([[6.080564e-05, -7.120823e-05],
                           [-1.718945e-04, -4.000536e-05],
                           [-2.271720e-04, 8.663310e-05],
                           [-1.032577e-04, -3.582033e-05]])
    neighbors = np.array([[1, 2, 3],
                          [0, 2, 3],
                          [1, 0, 3],
                          [1, 2, 0]])
    grad_output = np.array([[0.0, 0.0],
                            [0.0, 0.0],
                            [4.24275173e-08, -3.69569698e-08],
                            [-2.58720939e-09, 7.52706374e-09]])
    yield (_run_answer_test, pos_input, pos_output, neighbors, grad_output,
           False, 0.1, 2)


def _run_answer_test(pos_input, pos_output, neighbors, grad_output,
                     verbose=False, perplexity=0.1, skip_num_points=0):
    distances = pairwise_distances(pos_input)
    args = distances, perplexity, verbose
    pos_output = pos_output.astype(np.float32)
    neighbors = neighbors.astype(np.int64)
    pij_input = _joint_probabilities(*args)
    pij_input = squareform(pij_input).astype(np.float32)
    grad_bh = np.zeros(pos_output.shape, dtype=np.float32)

    _barnes_hut_tsne.gradient(pij_input, pos_output, neighbors,
                              grad_bh, 0.5, 2, 1, skip_num_points=0)
    assert_array_almost_equal(grad_bh, grad_output, decimal=4)


def test_verbose():
    random_state = check_random_state(0)
    tsne = TSNE(verbose=2)
    X = random_state.randn(5, 2)

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        tsne.fit_transform(X)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    assert("[t-SNE]" in out)
    assert("Computing pairwise distances" in out)
    assert("Computed conditional probabilities" in out)
    assert("Mean sigma" in out)
    assert("Finished" in out)
    assert("early exaggeration" in out)
    assert("Finished" in out)


def test_chebyshev_metric():
    """t-SNE should allow metrics that cannot be squared (issue #3526)."""
    random_state = check_random_state(0)
    tsne = TSNE(verbose=2, metric="chebyshev")
    X = random_state.randn(5, 2)
    tsne.fit_transform(X)


def test_no_sparse_on_barnes_hut():
    """No sparse matrices allowed on Barnes-Hut"""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    X[(np.random.randint(0, 100, 50), np.random.randint(0, 2, 50))] = 0.0
    X_csr = sp.csr_matrix(X)
    tsne = TSNE(n_iter=199, method='barnes_hut')
    assert_raises_regexp(TypeError, ".*sparse.*standard.*", tsne.fit_transform,
                         X_csr)


def test_no_4D_on_barnes_hut():
    """No sparse matrices allowed on Barnes-Hut"""
    random_state = check_random_state(0)
    X = random_state.randn(5, 2)
    for nc in [4, 100]:
        tsne = TSNE(n_iter=199, method='barnes_hut', n_components=nc)
        m = ".*method='barnes_hut' only available for.*"
        assert_raises_regexp(ValueError, m, tsne.fit_transform, X)


def test_64bit():
    """Test to ensure 64bit arrays are handled correctly"""
    random_state = check_random_state(0)
    methods = ['barnes_hut', 'standard']
    for method in methods:
        for dt in [np.float32, np.float64]:
            X = random_state.randn(100, 2).astype(dt)
            tsne = TSNE(n_components=2, perplexity=2, learning_rate=100.0,
                        random_state=0, method=method)
            tsne.fit_transform(X)


def test_transform_before_fit():
    """transform() cannot be called before fit()"""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    tsne = TSNE(n_components=2, perplexity=2, learning_rate=100.0,
                random_state=0, method='barnes_hut')
    m = ".*Cannot call `transform` unless `fit` has.*"
    assert_raises_regexp(ValueError, m, tsne.transform, X)


def test_transform_warning():
    """Raises a warning if fit and transform encountered the same data """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        random_state = check_random_state(0)
        X = random_state.randn(100, 2)
        tsne = TSNE(n_components=2, perplexity=2, learning_rate=100.0,
                    random_state=0, method='barnes_hut')
        tsne.fit(X)
        tsne.transform(X)
        m = str(w[-1].message)
    assert "The transform input appears to be similar" in m


def test_quadtree_similar_point():
    """
    Test that a point can be introduced into a quad tree
    where a similar point already exists.

    Test will hang if it doesn't complete.
    """

    Xs = []
    # check the case where points are actually different
    Xs.append(np.array([[1, 2], [3, 4]]))
    # check the case where points are the same on X axis
    Xs.append(np.array([[-9.368728, 10.264389], [-9.368728, 11.264389]]))
    # check the case where points are arbitraryily close on X axis
    Xs.append(np.array([[-9.368728, 10.264389], [-9.368761, 11.264389]]))
    # check the case where points are the same on Y axis
    Xs.append(np.array([[-10.368728, 3.264389], [-11.368761, 3.264389]]))
    # check the case where points are arbitrarily close on Y axis
    Xs.append(np.array([[-10.368728, 3.264339], [-11.368761, 3.264389]]))
    # check the case where points are arbitraryily close on both axes
    Xs.append(np.array([[-9.368728, 3.264389], [-9.368761, 3.264389]]))

    for X in Xs:
        counts = np.zeros(3, dtype='int64')
        _barnes_hut_tsne.check_quadtree(X, counts)
        m = "Tree consistency failed: unexpected number of points at root node"
        assert counts[0] == counts[1], m
        m = "Tree consistency failed: unexpected number of points on the tree"
        assert counts[0] == counts[2], m
