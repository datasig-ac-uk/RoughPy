#ifndef ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP
#define ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP

#include "py_headers.h"
#include "py_obj_handle.hpp"

#include <roughpy_compute/common/basis.hpp>

namespace rpy::compute {

struct CallConfig
{
    int32_t out_max_degree = -1;
    int32_t out_min_degree = 0;
    int32_t lhs_max_degree = -1;
    int32_t rhs_max_degree = -1;
    int32_t lhs_min_degree = 0;
    int32_t rhs_min_degree = 0;
    BasisBase const* basis_data = nullptr;
    void const* lhs_op = nullptr;
    void const* rhs_op = nullptr;
};

bool update_algebra_params(CallConfig& config);


PyObjHandle to_basis(PyObject* basis_obj, TensorBasis& basis);
PyObjHandle to_basis(PyObject* basis_obj, LieBasis& basis);

}// namespace rpy::compute


#endif //ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP