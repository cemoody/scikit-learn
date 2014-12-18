# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from cython.parallel cimport prange, parallel, threadid
cimport numpy as np
cimport cython
cimport openmp
# Implementation by Chris Moody
# original code by Laurens van der Maaten
# for reference implemenations and papers describing the technique:
# http://homepage.tudelft.nl/19j49/t-SNE.html


# TODO:
# Include usage documentation
# Find all references to embedding in sklearn & see where else we can document
# Incorporate into SKLearn
# PEP8 the code
# Am I using memviews or returning fulla rrays appropriately?
# DONE:
# Cython deallocate memory

from libc.stdlib cimport malloc, free, abs

cdef extern from "math.h":
    double sqrt(double x) nogil

cdef struct QuadNode:
    # Keep track of the center of mass
    float[2] cum_com
    # If this is a leaf, the position of the particle within this leaf 
    float[2] cur_pos
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
    float[2] le
    # The center of this node, equal to le + w/2.0
    float[2] c
    # The width of this node -- used to calculate the opening
    # angle. Equal to width = re - le
    float[2] w

    # Does this node have children?
    # Default to leaf until we add particles
    int is_leaf
    # Keep pointers to the child nodes
    QuadNode *children[2][2]
    # Keep a pointer to the parent
    QuadNode *parent
    # Keep a pointer to the node
    # to visit next when visting every QuadNode
    # QuadNode *next

cdef class QuadTree:
    # Holds a pointer to the root node
    cdef QuadNode* root_node 
    # Total number of cells
    cdef int num_cells
    # Total number of particles
    cdef int num_part
    # Spit out diagnostic information?
    cdef int verbose
    
    def __cinit__(self, float[:] width, int verbose=0):
        self.root_node = self.create_root(width)
        self.num_cells = 0
        self.num_part = 0
        self.verbose = verbose

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline QuadNode* create_root(self, float[:] width) nogil:
        # Create a default root node
        cdef int ax
        root = <QuadNode*> malloc(sizeof(QuadNode))
        root.is_leaf = 1
        root.parent = NULL
        root.level = 0
        root.cum_size = 0
        root.size = 0
        root.point_index = -1
        for ax in range(2):
            root.w[ax] = width[ax]
            root.le[ax] = 0. - root.w[ax] / 2.0
            root.c[ax] = 0.0
            root.cum_com[ax] = 0.
            root.cur_pos[ax] = -1.
        self.num_cells += 1
        return root

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline QuadNode* create_child(self, QuadNode *parent, int[2] offset) nogil:
        # Create a new child node with default parameters
        cdef int ax
        child = <QuadNode *> malloc(sizeof(QuadNode))
        child.is_leaf = 1
        child.parent = parent
        child.level = parent.level + 1
        child.size = 0
        child.cum_size = 0
        child.point_index = -1
        for ax in range(2):
            child.w[ax] = parent.w[ax] / 2.0
            child.le[ax] = parent.le[ax] + offset[ax] * parent.w[ax] / 2.0
            child.c[ax] = child.le[ax] + child.w[ax] / 2.0
            child.cum_com[ax] = 0.
            child.cur_pos[ax] = -1.
        self.num_cells += 1
        return child

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline QuadNode* select_child(self, QuadNode *node, float[2] pos) nogil:
        # Find which sub-node a position should go into
        # And return the appropriate node
        cdef int offset[2]
        cdef int ax
        for ax in range(2):
            offset[ax] = (pos[ax] - (node.le[ax] + node.w[ax] / 2.0)) > 0.
        return node.children[offset[0]][offset[1]]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void subdivide(self, QuadNode *node) nogil:
        # This instantiates 4 nodes for the current node
        cdef int i = 0
        cdef int j = 0
        cdef int[2] offset
        node.is_leaf = False
        offset[0] = 0
        offset[1] = 0
        for i in range(2):
            offset[0] = i
            for j in range(2):
                offset[1] = j
                node.children[i][j] = self.create_child(node, offset)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert(self, QuadNode *root, float pos[2], int point_index) nogil:
        # Introduce a new point into the quadtree
        # by recursively inserting it and subdividng as necessary
        cdef QuadNode *child
        cdef int i
        cdef int ax
        # Increment the total number points including this
        # node and below it
        root.cum_size += 1
        # Evaluate the new center of mass, weighting the previous
        # center of mass against the new point data
        cdef double frac_seen = <double>(root.cum_size - 1) / (<double> root.cum_size)
        cdef double frac_new  = 1.0 / <double> root.cum_size
        for ax in range(2):
            root.cum_com[ax] *= frac_seen
        for ax in range(2):
            root.cum_com[ax] += pos[ax] * frac_new
        # If this node is unoccupied, fill it.
        # Otherwise, we need to insert recursively.
        # Two insertion scenarios: 
        # 1) Insert into this node if it is a leaf and empty
        # 2) Subdivide this node if it is currently occupied
        if (root.size == 0) & root.is_leaf:
            for ax in range(2):
                root.cur_pos[ax] = pos[ax]
            root.point_index = point_index
            root.size = 1
        else:
            # If necessary, subdivide this node before
            # descending
            if root.is_leaf:
                self.subdivide(root)
            # We have two points to relocate: the one previously
            # at this node, and the new one we're attempting
            # to insert
            if root.size > 0:
                child = self.select_child(root, root.cur_pos)
                self.insert(child, root.cur_pos, root.point_index)
                # Remove the point from this node
                for ax in range(2):
                    root.cur_pos[ax] = -1
                root.size = 0
                root.point_index = -1
            # Insert the new point
            child = self.select_child(root, pos)
            self.insert(child, pos, point_index)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert_many(self, float[:,:] pos_array):
        cdef int nrows = pos_array.shape[0]
        cdef int i, ax
        cdef float row[2]
        for i in range(nrows):
            for ax in range(2):
                row[ax] = pos_array[i, ax]
            self.insert(self.root_node, row, i)
            self.num_part += 1

    cdef int free(self):
        cdef int check
        cdef int[:] cnt = np.zeros(3, dtype=np.int32)
        self.free_recursive(self.root_node, cnt)
        free(self.root_node)
        check = cnt[0] == self.num_cells
        check &= cnt[2] == self.num_part
        return check

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void free_recursive(self, QuadNode *root, int[:] counts):
        cdef int i, j    
        cdef QuadNode* child
        if not root.is_leaf:
            for i in range(2):
                for j in range(2):
                    child = root.children[i][j]
                    self.free_recursive(child, counts)
                    counts[0] += 1
                    if child.is_leaf:
                        counts[1] += 1
                        if child.size > 0:
                            counts[2] +=1
                    free(child)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef float[:,:] compute_gradient(self, float theta,
                                     float[:,:] val_P,
                                     float[:,:] pos_reference):
        cdef int ax = 0
        cdef int i = 0
        cdef int j = 0
        cdef int n = pos_reference.shape[0]
        cdef float[:,:] pos_force = np.zeros((n, 2), dtype=np.float32)
        cdef float[:,:] neg_force = np.zeros((n, 2), dtype=np.float32)
        cdef float[:,:] tot_force = np.zeros((n, 2), dtype=np.float32)
        cdef float* force
        cdef float* iQ 
        #cdef float[:] force = np.zeros(2, dtype=np.float32)
        cdef int point_index
        cdef float sum_Q = 0.0
        self.compute_edge_forces(val_P, pos_reference, pos_force)
        for point_index in range(n):
            # Clear force array
            force = <float*> malloc(sizeof(float) * 2)
            for ax in range(2): force[ax] = 0.0
            iQ = <float*> malloc(sizeof(float))
            iQ[0] = 0.0
            self.compute_non_edge_forces(self.root_node, theta, iQ, point_index,
                                         pos_reference, force)
            sum_Q += iQ[0]
            # Save local force into global
            for ax in range(2): neg_force[point_index, ax] = force[ax]

        for i in range(pos_force.shape[0]):
            for j in range(pos_force.shape[1]):
                tot_force[i, j] = pos_force[i, j] - (neg_force[i, j] / sum_Q)

        return tot_force

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef float[:,:] compute_gradient_parallel(self, float theta,
                                              float[:,:] val_P,
                                              float[:,:] pos_reference):
        cdef int ax = 0
        cdef int i = 0
        cdef int j = 0
        cdef int n = pos_reference.shape[0]
        cdef float[:,:] pos_force = np.zeros((n, 2), dtype=np.float32)
        cdef float[:,:] neg_force = np.zeros((n, 2), dtype=np.float32)
        cdef float[:,:] tot_force = np.zeros((n, 2), dtype=np.float32)
        cdef float* force
        cdef float* iQ 
        cdef int point_index
        cdef float sum_Q = 0.0
        cdef float[:] sum_Qs = np.zeros(n, dtype=np.float32)
        cdef int skip, start
        self.compute_edge_forces(val_P, pos_reference, pos_force)
        with nogil, parallel():
            skip = openmp.omp_get_num_threads()
            start = threadid()
            force = <float*> malloc(sizeof(float) * 2)
            iQ = <float*> malloc(sizeof(float))
            root_node = <QuadNode*> malloc(sizeof(QuadNode))
            root_node = self.root_node
            for point_index in prange(0, n, schedule='static'):
                # Clear force array
                for ax in range(2): force[ax] = 0.0
                iQ[0] = 0.0
                self.compute_non_edge_forces(root_node, theta, iQ, point_index,
                                             pos_reference, force)
                sum_Qs[point_index] = iQ[0]
                # Save local force into global
                for ax in range(2): neg_force[point_index, ax] = force[ax]
            free(iQ)
            free(force)
        for i in range(n):
            sum_Q += sum_Qs[i]

        for i in range(pos_force.shape[0]):
            for j in range(pos_force.shape[1]):
                tot_force[i, j] = pos_force[i, j] - (neg_force[i, j] / sum_Q)
        return tot_force


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef float[:,:] compute_gradient_exact(self, float theta,
                                           float[:,:] val_P,
                                           float[:,:] pos_reference):
        cdef float sum_Q = 0.0
        cdef float mult = 0.0
        cdef int i, j

        cdef int n = pos_reference.shape[0]
        cdef float[:,:] dC = np.zeros((n, 2), dtype=np.float32)
        cdef float[:,:] DD = pairwise_distances(pos_reference)
        cdef float[:,:] Q = np.zeros((n, n), dtype=np.float32)

        # Computation of the Q matrix & normalization sum
        for i in range(pos_reference.shape[0]):
            for j in range(pos_reference.shape[0]):
                Q[i, j] = 1. / (1. + DD[i, j])
                sum_Q += Q[i, j]

        # Computation of the gradient
        for i in range(pos_reference.shape[0]):
            for j in range(pos_reference.shape[0]):
                if i == j: 
                    continue
                mult = (val_P[i, j] - (Q[i, j] / sum_Q)) * Q[i, j]
                for ax in range(2):
                    dC[i, ax] += mult * (pos_reference[i, ax] - pos_reference[j, ax])
        return dC
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void compute_edge_forces(self, float[:,:] val_P,
                                  float[:,:] pos_reference,
                                  float[:,:] force):
        # Sum over the following expression for i not equal to j
        # grad_i = p_ij (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
        cdef int i, j, dim
        cdef float buff[2]
        cdef float D
        for i in range(pos_reference.shape[0]):
            for j in range(pos_reference.shape[0]):
                if i == j : 
                    continue
                D = 0.0
                for dim in range(2):
                    buff[dim] = pos_reference[i, dim] - pos_reference[j, dim]
                    D += buff[dim] ** 2.0  
                D = val_P[i, j] / (1.0 + D)
                for dim in range(2):
                    force[i, dim] += D * buff[dim]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double compute_non_edge_forces(self, QuadNode* node, 
                                 float theta,
                                 float* sum_Q,
                                 int point_index,
                                 float[:, :] pos_reference,
                                 float* force) nogil:
        # Compute the t-SNE force on the point in pos_reference given by point_index
        cdef QuadNode* child
        cdef int i, j
        cdef int summary = 0
        cdef float dist2, mult, qijZ
        cdef float delta[2] 
        cdef float wmax = 0.0

        for i in range(2):
            delta[i] = 0.0

        # There are no points below this node if cum_size == 0
        # so do not bother to calculate any force contributions
        # Also do not compute self-interactions
        if node.cum_size > 0 and not (node.is_leaf and (node.point_index == point_index)):
            dist2 = 0.0
            # Compute distance between node center of mass and the reference point
            for i in range(2):
                delta[i] += pos_reference[point_index, i] - node.cum_com[i] 
                dist2 += delta[i]**2.0
            # Check whether we can use this node as a summary
            # It's a summary node if the angular size as measured from the point
            # is relatively small (w.r.t. to theta) or if it is a leaf node.
            # If it can be summarized, we use the cell center of mass 
            # Otherwise, we go a higher level of resolution and into the leaves.
            for i in range(2):
                wmax = max(wmax, node.w[i])

            summary = (wmax / sqrt(dist2) < theta)

            if node.is_leaf or summary:
                # Compute the t-SNE force between the reference point and the current node
                qijZ = 1.0 / (1.0 + dist2)
                sum_Q[0] += node.cum_size * qijZ
                mult = node.cum_size * qijZ * qijZ
                for ax in range(2):
                    force[ax] += mult * delta[ax]
            else:
                # Recursively apply Barnes-Hut to child nodes
                for i in range(2):
                    for j in range(2):
                        child = node.children[i][j]
                        if child.cum_size == 0: 
                            continue
                        self.compute_non_edge_forces(child, theta, sum_Q, 
                                                     point_index,
                                                     pos_reference, force)
        
    cdef int check_consistency(self):
        cdef int count 
        cdef int check
        count = 0
        count = self.count_points(self.root_node, count)
        check = count == self.root_node.cum_size
        check &= count == self.num_part
        return check

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int count_points(self, QuadNode* root, int count):
        # Walk through the whole tree and count the number 
        # of points at the leaf nodes
        cdef QuadNode* child
        cdef int i, j
        for i in range(2):
            for j in range(2):
                child = root.children[i][j]
                if child.is_leaf and child.size > 0:
                    count += 1
                elif not child.is_leaf:
                    count = self.count_points(child, count)
                # else case is we have an empty leaf node
                # which happens when we create a quadtree for
                # one point, and then the other neighboring cells
                # don't get filled in
        return count


cdef QuadTree create_quadtree(pos_output, verbose):
    pos_output = pos_output.astype(np.float32)
    width = pos_output.max(axis=0) - pos_output.min(axis=0)
    width = width.astype(np.float32)
    qt = QuadTree(width, verbose=verbose)
    qt.insert_many(pos_output)
    return qt


def consistency_checks(pos_output, verbose=0):
    pos_output = pos_output.astype(np.float32)
    qt = create_quadtree(pos_output, verbose=verbose)
    assert qt.check_consistency()
    assert qt.free()
    return True

def compute_gradient(pij_input, pos_output, theta=0.5, verbose=0, do_parallel=0):
    assert pij_input.dtype == np.float32
    assert pos_output.dtype == np.float32
    qt = create_quadtree(pos_output, verbose=verbose)
    if do_parallel:
        forces = qt.compute_gradient_parallel(theta, pij_input, pos_output)
    else:
        forces = qt.compute_gradient(theta, pij_input, pos_output)
    f = np.zeros(pos_output.shape, dtype=np.float32)
    f[:,:] = forces
    assert qt.check_consistency()
    assert qt.free()
    return f
