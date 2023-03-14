//
// Created by user on 14/03/23.
//

#include "algebra.h"

#include "tensor_key.h"
#include "lie_key.h"
#include "free_tensor.h"
#include "shuffle_tensor.h"
#include "lie.h"
#include "lie_key_iterator.h"
#include "tensor_key_iterator.h"



void rpy::python::init_algebra(pybind11::module_ &m) {

    py::enum_<algebra::VectorType>(m, "VectorType")
        .value("DenseVector", algebra::VectorType::Dense)
        .value("SparseVector", algebra::VectorType::Sparse)
        .export_values();

    init_py_tensor_key(m);
    init_py_lie_key(m);
    init_tensor_key_iterator(m);
    init_lie_key_iterator(m);
    init_context(m);

    init_free_tensor(m);
    init_shuffle_tensor(m);
    init_lie(m);


}
