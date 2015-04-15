"""
===================================
Comparison of t-SNE methods and options
===================================

A comparison of t-SNE applied to the Olivetti faces data. The default 'barnes-hut' 
methods is visualized, as well as init='pca' and init='random'. In addition to 
varying the init parameter, the script also compares preprocessing the input data 
by reducing the dimensionality via PCA.

The key learnings are that all the three configurations yield similar 
arrangements of the data. PCA-reduced features can speed up analysis 
while preserving most of the pairwise distances of the original feature space. 
In particular, the BallTree data structure used to compute the neighborhood in the 
original feature space is known to work faster than brute force only for lower 
dimensional space (e.g. less than a couple hundreds of dimensions). PCA
initialization on the other hand yields a global arrangement that is more 
consistent with the long range dependencies between groups.

For a discussion and comparison of these algorithms, see the
:ref:`manifold module page <manifold>`

"""
# Authors: Christopher Erick Moody <chrisemoody@gmail.com>
# License: BSD 3 clause

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import RandomizedPCA
from sklearn import manifold
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np
from time import time


faces = fetch_olivetti_faces()

verbose = 5

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(dataset, X, title=None, ax=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y = dataset.target
    if ax is None:
        plt.figure(figsize=(18, 15))
        ax = plt.gca()
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(dataset.target[i]),
                color=plt.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 30})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(dataset.data.shape[0]):
            shown_images = np.r_[shown_images, [X[i]]]
            r, g, b, a = plt.cm.Paired(i * 1.0 / dataset.data.shape[0])
            gray = dataset.images[i]
            rgb_image = np.zeros((64, 64, 3))
            rgb_image[:, :, 0] = gray * r
            rgb_image[:, :, 1] = gray * g
            rgb_image[:, :, 2] = gray * b
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(rgb_image),
                X[i], frameon=False)
            ax.add_artist(imagebox)
    ax.set_xticks([]), ax.set_yticks([])
    if title is not None:
        ax.set_title(title, loc='left')


#----------------------------------------------------------------------
# Fit t-SNE with random initialization.
tsne = manifold.TSNE(n_components=2, init='random', random_state=3,
                     n_iter=1000, verbose=verbose, early_exaggeration=50.0,
                     learning_rate=100, perplexity=50)
time_start = time()
embedded = tsne.fit_transform(faces.data)
time_end = time()
dt = time_end - time_start


title = ("Barnes-Hut t-SNE Visualization of Olivetti Faces\n" +
         "in %1.1f seconds initialized randomly")
title = title % dt
plot_embedding(faces, embedded, title=title)


#-------------------------------------------------------------------------
# Fit t-SNE, but use PCA to initialize the embedding
# Initializing the embedding with PCA allows the final emebedding to preserve
# global structure, leaving t-SNE to optimize the local structure
tsne = manifold.TSNE(n_components=2, init='pca', random_state=3,
                     n_iter=1000, verbose=verbose, early_exaggeration=50.0,
                     learning_rate=100, perplexity=5)
time_start = time()
embedded = tsne.fit_transform(faces.data)
time_end = time()
dt = time_end - time_start


title = ("Barnes-Hut t-SNE Visualization of Olivetti Faces\n" +
         "in %1.1f seconds & initialized embedding with PCA ")
title = title % dt
plot_embedding(faces, embedded, title=title)



#----------------------------------------------------------------------
# Fit t-SNE on 50 PCA-reduced dimensions with PCA initialization.
# We will frequently want to reduce the dimensionality when we have many
# dimensions since the t-SNE complexity grows linearly with the dimensionality.
tsne = manifold.TSNE(n_components=2, init='random', random_state=3,
                     n_iter=1000, verbose=verbose, early_exaggeration=50.0,
                     learning_rate=100, perplexity=50)
rpca = RandomizedPCA(n_components=50)
time_start = time()
reduced = rpca.fit_transform(faces.data)
embedded = tsne.fit_transform(reduced)
time_end = time()
dt = time_end - time_start


title = ("Barnes-Hut t-SNE Visualization of Olivetti Faces\n" +
         "in %1.1f seconds on 50-dimensional PCA-reduced data ")
title = title % dt
plot_embedding(faces, embedded, title=title)

plt.show()
