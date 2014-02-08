# cython: experimental_cpp_class_def=True

import numpy as np
cimport numpy as np

from libcpp.map cimport map
from libcpp.pair cimport pair

DEF DEBUG_CHECKS = True   # true if laborious parameter checks are needed 

# This is the energy type; should match the EnergyType and
# EnergyTermType in GCOptimization.h
DEF NRG_TYPE_STR = float

IF NRG_TYPE_STR == int:
    ctypedef np.int32_t NRG_DTYPE_t
    ctypedef int NRG_TYPE
    ctypedef map[pair[int,int],int] PW_MAP_T   # map (s1, s2) -> strength
ELSE:
    ctypedef np.float64_t NRG_DTYPE_t
    ctypedef double NRG_TYPE
    ctypedef map[pair[int,int],double] PW_MAP_T   # map (s1, s2) -> strength
    

np.import_array()

cdef extern from "GCoptimization.h":
    cdef cppclass GCoptimizationGridGraph:
        cppclass SmoothCostFunctor:
            NRG_TYPE compute(int s1, int s2, int l1, int l2)
			
        GCoptimizationGridGraph(int width, int height, int n_labels)
        void setDataCost(NRG_TYPE *)
        void setSmoothCost(NRG_TYPE *)
        NRG_TYPE expansion(int n_iterations)
        NRG_TYPE swap(int n_iterations)
        void setSmoothCostVH(NRG_TYPE* pairwise, NRG_TYPE* V, NRG_TYPE* H)
        void setSmoothCostFunctor(SmoothCostFunctor* f)
        int whatLabel(int node)
        void setLabelCost(NRG_TYPE *)
        void setLabel(int node, int label)
        NRG_TYPE compute_energy()

    cdef cppclass GCoptimizationGeneralGraph:
        GCoptimizationGeneralGraph(int n_vertices, int n_labels)
        void setDataCost(NRG_TYPE *)
        void setSmoothCost(NRG_TYPE *)
        void setNeighbors(int, int)
        void setNeighbors(int, int, NRG_TYPE)
        NRG_TYPE expansion(int n_iterations)
        NRG_TYPE swap(int n_iterations)
        void setSmoothCostFunctor(GCoptimizationGridGraph.SmoothCostFunctor* f) # yep, it works
        int whatLabel(int node)
        void setLabelCost(NRG_TYPE *)
        void setLabel(int node, int label)
        NRG_TYPE compute_energy()
        
        
cdef cppclass GeneralizedPottsFunctor(GCoptimizationGridGraph.SmoothCostFunctor):
    PW_MAP_T data_
    
    __init__(object data):
        this.data_ = data
    
    NRG_TYPE compute(int s1, int s2, int l1, int l2):
        if l1 != l2: 
            return 0
        else:
            pair = tuple(sorted([s1,s2]))
            return -this.data_[pair]  

    
def cut_from_graph_gen_potts(
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] unary_cost,
        object pairwise_cost, n_iter=5,
        algorithm='expansion',
        np.ndarray[NRG_DTYPE_t, ndim=1, mode='c'] label_cost=None):
    """
    Apply multi-label graphcuts to arbitrary graph given by `edges`.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(n_vertices, n_labels)
        Unary potentials
    pairwise_cost: dict: (site1, site2) -> strength, where site1 < site2.
        The order of nodes is the same as in unary_cost
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    cdef int n_vertices = unary_cost.shape[0]
    cdef int n_labels = unary_cost.shape[1]

    if label_cost is not None and (label_cost.shape[0] != n_labels):
        raise ValueError("label_cost must be an array of size n_labels.\n")

    cdef GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(n_vertices, n_labels)
    for edge, strength in pairwise_cost.items():
        gc.setNeighbors(edge[0], edge[1])
        if edge[0] >= edge[1]:
            raise ValueError("The order of sites in the edge (%d,%d) should be ascending" % edge)
        if strength < 0:
            raise ValueError("Pairwise potential for the edge (%d,%d) is negative, "
                             "which is not allowed in generalized Potts" % edge)

    if label_cost is not None:
        gc.setLabelCost(<NRG_TYPE*>label_cost.data)
        
    gc.setDataCost(<NRG_TYPE*>unary_cost.data)
    cdef GeneralizedPottsFunctor* functor = new GeneralizedPottsFunctor(pairwise_cost)
    gc.setSmoothCostFunctor(functor)
    cdef NRG_TYPE nrg
    if algorithm == 'swap':
        nrg = gc.swap(n_iter)
    elif algorithm == 'expansion':
        nrg = gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_vertices
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_vertices):
        result_ptr[i] = gc.whatLabel(i)
        
    del gc
    del functor
    return result, nrg
