# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
from cython.parallel cimport prange, parallel
from libc.stdio cimport printf
cimport numpy as np
cimport cython
cimport openmp

# Implementation by Chris Moody & Nick Travers
# original code by Laurens van der Maaten
# for reference implemenations and papers describing the technique:
# http://homepage.tudelft.nl/19j49/t-SNE.html


from libc.stdlib cimport malloc, free, abs

cdef extern from "math.h":
    double sqrt(double x) nogil
    double ceil(double x) nogil

cdef extern from "omp.h":
    int omp_get_thread_num() nogil

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
    int cum_size
    # Number of particles at this node
    int size
    # Index of the particle at this node
    int point_index
    # level = 0 is the root node
    # And each subdivision adds 1 to the level
    int level
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
    int num_cells
    # Total number of particles
    int num_part
    # Spit out diagnostic information?
    int verbose

cdef Tree* init_tree(float[:] width, int dimension, int verbose) nogil:
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
    cdef int offset[3]
    cdef int ax
    # In case we don't have 3D data, set it to zero
    offset[2] = 0
    for ax in range(node.tree.dimension):
        offset[ax] = (pos[ax] - (node.le[ax] + node.w[ax] / 2.0)) > 0.
    return node.children[offset[0]][offset[1]][offset[2]]

cdef void subdivide(Node* node) nogil:
    # This instantiates 4 or 8 nodes for the current node
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int[3] offset
    node.is_leaf = False
    offset[0] = 0
    offset[1] = 0
    offset[2] = 0
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

cdef void insert(Node *root, float pos[3], int point_index) nogil:
    # Introduce a new point into the tree
    # by recursively inserting it and subdividng as necessary
    cdef Node *child
    cdef int i
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
    cdef int nrows = pos_array.shape[0]
    cdef int i, ax
    cdef float row[3]
    for i in range(nrows):
        for ax in range(tree.dimension):
            row[ax] = pos_array[i, ax]
        insert(tree.root_node, row, i)
        tree.num_part += 1

cdef int free_tree(Tree* tree) nogil:
    cdef int check
    cdef int* cnt = <int*> malloc(sizeof(int) * 3)
    for i in range(3):
        cnt[i] = 0
    free_recursive(tree, tree.root_node, cnt)
    free(tree.root_node)
    check = cnt[0] == tree.num_cells
    check &= cnt[2] == tree.num_part
    free(cnt)
    return check

cdef void free_recursive(Tree* tree, Node *root, int* counts) nogil:
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

cdef int check_consistency(Tree* tree) nogil:
    # Ensure that the number of cells and data
    # points removed are equal to the number
    # removed
    cdef int count 
    cdef int check
    count = 0
    count = count_points(tree.root_node, count)
    check = count == tree.root_node.cum_size
    check &= count == tree.num_part
    return check

cdef int count_points(Node* root, int count) nogil:
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
                           float[:,:] tot_force,
                           Node* root_node,
                           float theta,
                           int start,
                           int stop) nogil:
    # Having created the tree, calculate the gradient
    # in two components, the positive and negative forces
    cdef int i, ax, coord
    cdef int n = pos_reference.shape[0]
    cdef int dimension = root_node.tree.dimension
    cdef float* sum_Q = <float*> malloc(sizeof(float))
    cdef float* neg_f = <float*> malloc(sizeof(float) * n * dimension)
    cdef float* pos_f = <float*> malloc(sizeof(float) * n * dimension)
    cdef clock_t t1, t2

    sum_Q[0] = 0.0
    t1 = clock()
    compute_gradient_positive(val_P, pos_reference, pos_f, dimension)
    t2 = clock()
    if root_node.tree.verbose > 15:
        printf("positive: %e sec\n", ((float) (t2 - t1)))
    t1 = clock()
    compute_gradient_negative(val_P, pos_reference, neg_f, root_node, sum_Q, 
                              theta, start, stop)
    t2 = clock()
    if root_node.tree.verbose > 15:
        printf("negative: %e sec\n", ((float) (t2 - t1)))

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
        int i, j, ax
        int n = val_P.shape[0]
        float buff[3]
        float D
        int temp
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


cdef void compute_gradient_positive_parallel(float[:,:] val_P,
                                             float[:,:] pos_reference,
                                             float* pos_f,
                                             int dimension) nogil:
    # Sum over the following expression for i not equal to j
    # grad_i = p_ij (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
    cdef:
        int i, j, ax
        int n = val_P.shape[0]
        float* buff
        float* pos_f_buff
        float* D
    with parallel():
        buff = <float*> malloc(sizeof(float) * dimension)
        pos_f_buff = <float*> malloc(sizeof(float) * dimension)
        D = <float*> malloc(sizeof(float) )
        for i in prange(n, schedule='static'):
            for ax in range(dimension):
                pos_f_buff[ax] = 0.0
            for j in range(n):
                if i == j : 
                    continue
                D[0] = 0.0
                for ax in range(dimension):
                    buff[ax] = pos_reference[i, ax] - pos_reference[j, ax]
                    D[0] += buff[ax] ** 2.0  
                D[0] = val_P[i, j] / (1.0 + D[0])
                for ax in range(dimension):
                    pos_f_buff[ax] += D[0] * buff[ax]
            for ax in range(dimension):
                pos_f[i * dimension + ax] = pos_f_buff[ax]
        free(buff)
        free(pos_f_buff)
        free(D)


cdef void compute_gradient_negative(float[:,:] val_P, 
                                    float[:,:] pos_reference,
                                    float* neg_f,
                                    Node *root_node,
                                    float* sum_Q,
                                    float theta, 
                                    int start, 
                                    int stop) nogil:
    if stop == -1:
        stop = pos_reference.shape[0] 
    cdef:
        int ax, i
        int n = stop - start
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


cdef void compute_gradient_negative_parallel(float[:,:] val_P, 
                                             float[:,:] pos_reference,
                                             float* neg_f,
                                             Node *root_node,
                                             float* sum_Q,
                                             float theta, 
                                             int start, 
                                             int stop) nogil:
    if stop == -1:
        stop = pos_reference.shape[0] 
    cdef:
        int ax, i
        int n = stop - start
        float* force
        float* iQ 
        float* pos
        int dimension = root_node.tree.dimension
        int step = <int> (ceil(n / 4.0))

    with parallel():
        iQ = <float*> malloc(sizeof(float))
        force = <float*> malloc(sizeof(float) * dimension)
        pos = <float*> malloc(sizeof(float) * dimension)
        for i in prange(start, stop, schedule='static'):
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
                                  int point_index,
                                  float* pos,
                                  float* force) nogil:
    # Compute the t-SNE force on the point in pos given by point_index
    cdef:
        Node* child
        int i, j
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


def gradient(float[:] width, 
             float[:,:] pij_input, 
             float[:,:] pos_output, 
             float[:,:] forces, 
             float theta,
             int dimension,
             int verbose):
    # This function is designed to be called from external Python
    # it passes the 'forces' array by reference and fills thats array
    # up in-place
    assert width.itemsize == 4
    assert pij_input.itemsize == 4
    assert pos_output.itemsize == 4
    assert forces.itemsize == 4
    cdef Tree* qt = init_tree(width, dimension, verbose)
    insert_many(qt, pos_output)
    compute_gradient(pij_input, pos_output, forces, qt.root_node, theta, 0, -1)
    check_consistency(qt)
    free_tree(qt)
