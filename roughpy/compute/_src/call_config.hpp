#ifndef ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP
#define ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP

#include "py_headers.h"
#include "py_obj_handle.hpp"

#include <roughpy_compute/common/basis.hpp>

namespace rpy::compute {

struct DegreeBounds
{
    int32_t max_degree = -1;
    int32_t min_degree = 0;
};

struct CallConfig
{
    DegreeBounds* degree_bounds = nullptr;
    BasisBase const* const* basis_data = nullptr;
    void* ops;
};

bool update_algebra_params(CallConfig& config, npy_intp n_args, npy_intp const* arg_basis_mapping);


PyObjHandle to_basis(PyObject* basis_obj, TensorBasis& basis);
PyObjHandle to_basis(PyObject* basis_obj, LieBasis& basis);

}// namespace rpy::compute


#endif //ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP