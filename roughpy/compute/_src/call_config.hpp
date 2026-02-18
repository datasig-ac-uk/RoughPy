#ifndef ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP
#define ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP

#include <roughpy/pycore/py_headers.h>
#include <roughpy/pycore/object_handle.hpp>

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


struct LieBasisArrayHolder
{
    PyArrayObject* degree_begin = nullptr;
    PyArrayObject* data = nullptr;

    LieBasisArrayHolder() = default;

    LieBasisArrayHolder(LieBasisArrayHolder&& old) noexcept
        : degree_begin(old.degree_begin), data(old.data)
    {
        old.degree_begin = nullptr;
        old.data = nullptr;
    }

    ~LieBasisArrayHolder()
    {
        Py_XDECREF(degree_begin);
        Py_XDECREF(data);
    }

    explicit operator bool() const noexcept
    {
        return degree_begin != nullptr && data != nullptr;
    }

};

LieBasisArrayHolder to_basis(PyObject* basis_obj, LieBasis& basis);

}// namespace rpy::compute


#endif //ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP