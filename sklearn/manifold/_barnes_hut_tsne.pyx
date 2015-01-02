# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
from cython.parallel cimport prange, parallel
from libc.stdio cimport printf
cimport numpy as np
cimport cython

# Implementation by Chris Moody & Nick Travers
# original code by Laurens van der Maaten
# for reference implemenations and papers describing the technique:
# http://homepage.tudelft.nl/19j49/t-SNE.html


from libc.stdlib cimport malloc, free, abs

cdef extern from "math.h":
    double sqrt(double x) nogil


cdef extern from "time.h":
    # Declare only what is necessary from `tm` structure.
    ctypedef long clock_t
    clock_t clock() nogil
    double CLOCKS_PER_SEC


cdef struct Node:
    # Keep track of the center of mass
    float[3] cum_com
    # If this is a leaf, the position of the particle within this leaf 
    float[3] cur_pos
    # The number of particles including all 
    # nodes below this one
    long cum_size
    # Number of particles at this node
    long size
    # Index of the particle at this node
    long point_index
    # level = 0 is the root node
    # And each subdivision adds 1 to the level
    long level
    # Left edge of this node, normalized to [0,1]
    float[3] le
    # The center of this node, equal to le + w/2.0
    float[3] c
    # The width of this node -- used to calculate the opening
    # angle. Equal to width = re - le
    float[3] w

    # Does this node have children?
    # Default to leaf until we add particles
    int is_leaf
    # Keep pointers to the child nodes
    Node *children[2][2][2]
    # Keep a pointer to the parent
    Node *parent
    # Pointer to the tree this node belongs too
    Tree* tree

cdef struct Tree:
    # Holds a pointer to the root node
    Node* root_node 
    # Number of dimensions in the ouput
    int dimension
    # Total number of cells
    long num_cells
    # Total number of particles
    long num_part
    # Spit out diagnostic information?
    int verbose

cdef Tree* init_tree(float[:] width, int dimension, int verbose) nogil:
    # tree is freed by free_tree
    cdef Tree* tree = <Tree*> malloc(sizeof(Tree))
    tree.dimension = dimension
    tree.num_cells = 0
    tree.num_part = 0
    tree.verbose = verbose
    tree.root_node = create_root(width, dimension)
    tree.root_node.tree = tree
    tree.num_cells += 1
    return tree

cdef Node* create_root(float[:] width, int dimension) nogil:
    # Create a default root node
    cdef int ax
    # root is freed by free_tree
    root = <Node*> malloc(sizeof(Node))
    root.is_leaf = 1
    root.parent = NULL
    root.level = 0
    root.cum_size = 0
    root.size = 0
    root.point_index = -1
    for ax in range(dimension):
        root.w[ax] = width[ax]
        root.le[ax] = 0. - root.w[ax] / 2.0
        root.c[ax] = 0.0
        root.cum_com[ax] = 0.
        root.cur_pos[ax] = -1.
    return root

cdef Node* create_child(Node *parent, int[3] offset) nogil:
    # Create a new child node with default parameters
    cdef int ax
    # these children are freed by free_recursive
    child = <Node *> malloc(sizeof(Node))
    child.is_leaf = 1
    child.parent = parent
    child.level = parent.level + 1
    child.size = 0
    child.cum_size = 0
    child.point_index = -1
    child.tree = parent.tree
    for ax in range(parent.tree.dimension):
        child.w[ax] = parent.w[ax] / 2.0
        child.le[ax] = parent.le[ax] + offset[ax] * parent.w[ax] / 2.0
        child.c[ax] = child.le[ax] + child.w[ax] / 2.0
        child.cum_com[ax] = 0.
        child.cur_pos[ax] = -1.
    child.tree.num_cells += 1
    return child

cdef Node* select_child(Node *node, float[3] pos) nogil:
    # Find which sub-node a position should go into
    # And return the appropriate node
    cdef int[3] offset
    cdef int ax
    # In case we don't have 3D data, set it to zero
    for ax in range(3):
        offset[ax] = 0
    for ax in range(node.tree.dimension):
        offset[ax] = (pos[ax] - (node.le[ax] + node.w[ax] / 2.0)) > 0.
        # if we are talking about the same point on this axis,
        # then flip the offset on this axis
        if pos[ax] == node.cur_pos[ax]:
            offset[ax] = 1 - offset[ax]
    return node.children[offset[0]][offset[1]][offset[2]]

cdef void subdivide(Node* node) nogil:
    # This instantiates 4 or 8 nodes for the current node
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int[3] offset
    node.is_leaf = False
    for ax in range(3):
        offset[ax] = 0
    if node.tree.dimension > 2:
        krange = 2
    else:
        krange = 1
    for i in range(2):
        offset[0] = i
        for j in range(2):
            offset[1] = j
            for k in range(krange):
                offset[2] = k
                node.children[i][j][k] = create_child(node, offset)

cdef void insert(Node *root, float pos[3], long point_index) nogil:
    # Introduce a new point into the tree
    # by recursively inserting it and subdividng as necessary
    cdef Node *child
    cdef long i
    cdef int ax
    cdef int dimension = root.tree.dimension
    # Increment the total number points including this
    # node and below it
    root.cum_size += 1
    # Evaluate the new center of mass, weighting the previous
    # center of mass against the new point data
    cdef double frac_seen = <double>(root.cum_size - 1) / (<double> root.cum_size)
    cdef double frac_new  = 1.0 / <double> root.cum_size
    for ax in range(dimension):
        root.cum_com[ax] *= frac_seen
    for ax in range(dimension):
        root.cum_com[ax] += pos[ax] * frac_new
    # If this node is unoccupied, fill it.
    # Otherwise, we need to insert recursively.
    # Two insertion scenarios: 
    # 1) Insert into this node if it is a leaf and empty
    # 2) Subdivide this node if it is currently occupied
    if (root.size == 0) & root.is_leaf:
        for ax in range(dimension):
            root.cur_pos[ax] = pos[ax]
        root.point_index = point_index
        root.size = 1
    else:
        # If necessary, subdivide this node before
        # descending
        if root.is_leaf:
            subdivide(root)
        # We have two points to relocate: the one previously
        # at this node, and the new one we're attempting
        # to insert
        if root.size > 0:
            child = select_child(root, root.cur_pos)
            insert(child, root.cur_pos, root.point_index)
            # Remove the point from this node
            for ax in range(dimension):
                root.cur_pos[ax] = -1
            root.size = 0
            root.point_index = -1
        # Insert the new point
        child = select_child(root, pos)
        insert(child, pos, point_index)

cdef void insert_many(Tree* tree, float[:,:] pos_array) nogil:
    # Insert each data point into the tree one at a time
    cdef long nrows = pos_array.shape[0]
    cdef long i
    cdef int ax
    cdef float row[3]
    for i in range(nrows):
        for ax in range(tree.dimension):
            row[ax] = pos_array[i, ax]
        insert(tree.root_node, row, i)
        tree.num_part += 1

cdef int free_tree(Tree* tree) nogil:
    cdef int check
    cdef long* cnt = <long*> malloc(sizeof(long) * 3)
    for i in range(3):
        cnt[i] = 0
    free_recursive(tree, tree.root_node, cnt)
    free(tree.root_node)
    free(tree)
    check = cnt[0] == tree.num_cells
    check &= cnt[2] == tree.num_part
    free(cnt)
    free(tree)
    return check

cdef void free_recursive(Tree* tree, Node *root, long* counts) nogil:
    # Free up all of the tree nodes recursively
    # while counting the number of nodes visited
    # and total number of data points removed
    cdef int i, j, krange
    cdef int k = 0
    cdef Node* child
    if root.tree.dimension > 2:
        krange = 2
    else:
        krange = 1
    if not root.is_leaf:
        for i in range(2):
            for j in range(2):
                for k in range(krange):
                    child = root.children[i][j][k]
                    free_recursive(tree, child, counts)
                    counts[0] += 1
                    if child.is_leaf:
                        counts[1] += 1
                        if child.size > 0:
                            counts[2] +=1
                    free(child)


cdef long count_points(Node* root, long count) nogil:
    # Walk through the whole tree and count the number 
    # of points at the leaf nodes
    cdef Node* child
    cdef int i, j
    if root.tree.dimension > 2:
        krange = 2
    else:
        krange = 1
    for i in range(2):
        for j in range(2):
            for k in range(krange):
                child = root.children[i][j][k]
                if child.is_leaf and child.size > 0:
                    count += 1
                elif not child.is_leaf:
                    count = count_points(child, count)
                # else case is we have an empty leaf node
                # which happens when we create a quadtree for
                # one point, and then the other neighboring cells
                # don't get filled in
    return count


cdef void compute_gradient(float[:,:] val_P,
                           float[:,:] pos_reference,
                           long[:,:] neighbors,
                           float[:,:] tot_force,
                           Node* root_node,
                           float theta,
                           long start,
                           long stop) nogil:
    # Having created the tree, calculate the gradient
    # in two components, the positive and negative forces
    cdef long i, coord
    cdef int ax
    cdef long n = pos_reference.shape[0]
    cdef int dimension = root_node.tree.dimension
    if root_node.tree.verbose > 11:
        printf("Allocating %i elements in force arrays",
                n * dimension * 2)
    cdef float* sum_Q = <float*> malloc(sizeof(float))
    cdef float* neg_f = <float*> malloc(sizeof(float) * n * dimension)
    cdef float* pos_f = <float*> malloc(sizeof(float) * n * dimension)
    cdef clock_t t1, t2

    sum_Q[0] = 0.0
    if root_node.tree.verbose > 11:
        printf("computing positive gradient")
    t1 = clock()
    compute_gradient_positive_nn(val_P, pos_reference, neighbors, pos_f, dimension)
    t2 = clock()
    if root_node.tree.verbose > 15:
        printf("  nn pos: %e ticks\n", ((float) (t2 - t1)))
    if root_node.tree.verbose > 11:
        printf("computing negative gradient")
    t1 = clock()
    compute_gradient_negative(val_P, pos_reference, neg_f, root_node, sum_Q, 
                              theta, start, stop)
    t2 = clock()
    if root_node.tree.verbose > 15:
        printf("negative: %e ticks\n", ((float) (t2 - t1)))
    for i in range(n):
        for ax in range(dimension):
            coord = i * dimension + ax
            tot_force[i, ax] = pos_f[coord] - (neg_f[coord] / sum_Q[0])
    free(sum_Q)
    free(neg_f)
    free(pos_f)

cdef void compute_gradient_positive(float[:,:] val_P,
                                    float[:,:] pos_reference,
                                    float* pos_f,
                                    int dimension) nogil:
    # Sum over the following expression for i not equal to j
    # grad_i = p_ij (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
    # This is equivalent to compute_edge_forces in the authors' code
    cdef:
        int ax
        long i, j, temp
        long n = val_P.shape[0]
        float buff[3]
        float D
    for i in range(n):
        for ax in range(dimension):
            pos_f[i * dimension + ax] = 0.0
        for j in range(n):
            if i == j : 
                continue
            D = 0.0
            for ax in range(dimension):
                buff[ax] = pos_reference[i, ax] - pos_reference[j, ax]
                D += buff[ax] ** 2.0  
            D = val_P[i, j] / (1.0 + D)
            for ax in range(dimension):
                pos_f[i * dimension + ax] += D * buff[ax]
                temp = i * dimension + ax


cdef void compute_gradient_positive_nn(float[:,:] val_P,
                                       float[:,:] pos_reference,
                                       long[:,:] neighbors,
                                       float* pos_f,
                                       int dimension) nogil:
    # Sum over the following expression for i not equal to j
    # grad_i = p_ij (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
    # This is equivalent to compute_edge_forces in the authors' code
    # It just goes over the nearest neighbors instead of all the data points
    # (unlike the non-nearest neighbors version of `compute_gradient_positive')
    cdef:
        int ax
        long i, j, k
        long K = neighbors.shape[1]
        long n = val_P.shape[0]
        float[3] buff
        float D
    for i in range(n):
        for ax in range(dimension):
            pos_f[i * dimension + ax] = 0.0
        for k in range(K):
            j = neighbors[i, k]
            # we don't need to exclude the i==j case since we've 
            # already thrown it out from the list of neighbors
            D = 0.0
            for ax in range(dimension):
                buff[ax] = pos_reference[i, ax] - pos_reference[j, ax]
                D += buff[ax] ** 2.0  
            D = val_P[i, j] / (1.0 + D)
            for ax in range(dimension):
                pos_f[i * dimension + ax] += D * buff[ax]



cdef void compute_gradient_negative(float[:,:] val_P, 
                                    float[:,:] pos_reference,
                                    float* neg_f,
                                    Node *root_node,
                                    float* sum_Q,
                                    float theta, 
                                    long start, 
                                    long stop) nogil:
    if stop == -1:
        stop = pos_reference.shape[0] 
    cdef:
        int ax
        long i
        long n = stop - start
        float* force
        float* iQ 
        float* pos
        int dimension = root_node.tree.dimension

    iQ = <float*> malloc(sizeof(float))
    force = <float*> malloc(sizeof(float) * dimension)
    pos = <float*> malloc(sizeof(float) * dimension)
    for i in range(start, stop):
        # Clear the arrays
        for ax in range(dimension): 
            force[ax] = 0.0
            pos[ax] = pos_reference[i, ax]
        iQ[0] = 0.0
        compute_non_edge_forces(root_node, theta, iQ, i,
                                pos, force)
        sum_Q[0] += iQ[0]
        # Save local force into global
        for ax in range(dimension): 
            neg_f[i * dimension + ax] = force[ax]
    free(iQ)
    free(force)
    free(pos)


cdef void compute_non_edge_forces(Node* node, 
                                  float theta,
                                  float* sum_Q,
                                  long point_index,
                                  float* pos,
                                  float* force) nogil:
    # Compute the t-SNE force on the point in pos given by point_index
    cdef:
        Node* child
        int i, j, krange
        int summary = 0
        int dimension = node.tree.dimension
        float dist2, mult, qijZ
        float wmax = 0.0
        float* delta  = <float*> malloc(sizeof(float) * dimension)
    
    if node.tree.dimension > 2:
        krange = 2
    else:
        krange = 1

    for i in range(dimension):
        delta[i] = 0.0

    # There are no points below this node if cum_size == 0
    # so do not bother to calculate any force contributions
    # Also do not compute self-interactions
    if node.cum_size > 0 and not (node.is_leaf and (node.point_index == point_index)):
        dist2 = 0.0
        # Compute distance between node center of mass and the reference point
        for i in range(dimension):
            delta[i] += pos[i] - node.cum_com[i] 
            dist2 += delta[i]**2.0
        # Check whether we can use this node as a summary
        # It's a summary node if the angular size as measured from the point
        # is relatively small (w.r.t. to theta) or if it is a leaf node.
        # If it can be summarized, we use the cell center of mass 
        # Otherwise, we go a higher level of resolution and into the leaves.
        for i in range(dimension):
            wmax = max(wmax, node.w[i])

        summary = (wmax / sqrt(dist2) < theta)

        if node.is_leaf or summary:
            # Compute the t-SNE force between the reference point and the current node
            qijZ = 1.0 / (1.0 + dist2)
            sum_Q[0] += node.cum_size * qijZ
            mult = node.cum_size * qijZ * qijZ
            for ax in range(dimension):
                force[ax] += mult * delta[ax]
        else:
            # Recursively apply Barnes-Hut to child nodes
            for i in range(dimension):
                for j in range(dimension):
                    for k in range(krange):
                        child = node.children[i][j][k]
                        if child.cum_size == 0: 
                            continue
                        compute_non_edge_forces(child, theta, sum_Q, 
                                                     point_index,
                                                     pos, force)

    free(delta)


def gradient(float[:] width, 
             float[:,:] pij_input, 
             float[:,:] pos_output, 
             long[:,:] neighbors, 
             float[:,:] forces, 
             float theta,
             int dimension,
             int verbose):
    # This function is designed to be called from external Python
    # it passes the 'forces' array by reference and fills thats array
    # up in-place
    n = pos_output.shape[0]
    assert width.itemsize == 4
    assert pij_input.itemsize == 4
    assert pos_output.itemsize == 4
    assert forces.itemsize == 4
    m = "Number of neighbors must be < # of points - 1"
    assert n - 1 >= neighbors.shape[1], m
    m = "neighbors array and pos_output shapes are incompatible"
    assert n == neighbors.shape[0], m
    m = "Forces array and pos_output shapes are incompatible"
    assert n == forces.shape[0], m
    m = "Pij and pos_output shapes are incompatible"
    assert n == pij_input.shape[0], m
    m = "Pij and pos_output shapes are incompatible"
    assert n == pij_input.shape[1], m
    m = "Only 2D and 3D embeddings supported. Width array must be size 2 or 3"
    assert width.shape[0] <= 3, m
    if verbose > 10:
        printf("Initializing tree of dimension %i\n", dimension)
    cdef Tree* qt = init_tree(width, dimension, verbose)
    if verbose > 10:
        printf("Inserting %i points\n", pos_output.shape[0])
    insert_many(qt, pos_output)
    if verbose > 10:
        printf("Computing gradient\n")
    compute_gradient(pij_input, pos_output, neighbors, forces, qt.root_node, theta, 0, -1)
    if verbose > 10:
        printf("Checking tree consistency \n")
    cdef long count = count_points(qt.root_node, 0)
    m = "Tree consistency failed: unexpected number of points at root node"
    assert count == qt.root_node.cum_size, m 
    m = "Tree consistency failed: unexpected number of points on the tree"
    assert count == qt.num_part, m
    free_tree(qt)
