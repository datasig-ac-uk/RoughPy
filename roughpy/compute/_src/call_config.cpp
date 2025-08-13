#include "call_config.hpp"


#include <roughpy_compute/common/basis.hpp>


using namespace rpy::compute;


bool rpy::compute::update_algebra_params(CallConfig& config)
{
    auto* basis = config.basis_data;
    if (config.lhs_max_degree == -1 || config.lhs_max_degree >= basis->depth) {
        config.lhs_max_degree = basis->depth;
    }

    if (config.lhs_max_degree < config.lhs_min_degree) {
        PyErr_SetString(PyExc_ValueError,
                        "lhs_min_degree must be less than lhs_max_degree");
        return false;
    }

    if (config.rhs_max_degree == -1 || config.rhs_max_degree >= basis->depth) {
        config.rhs_max_degree = basis->depth;
    }
    if (config.rhs_max_degree < config.rhs_min_degree) {
        PyErr_SetString(PyExc_ValueError,
                        "rhs_min_degree must be less than rhs_max_degree");
        return false;
    }

    if (config.out_max_degree == -1 || config.out_max_degree >= basis->depth) {
        config.out_max_degree = basis->depth;
    }
    if (config.out_max_degree < config.out_min_degree) {
        PyErr_SetString(PyExc_ValueError,
                        "out_min_degree must be less than out_max_degree");
    }

    return true;
}


static bool width_and_depth_from_obj(PyObject* basis_obj,
                                     int32_t& width,
                                     int32_t& depth) noexcept
{
    PyObject* width_obj = PyObject_GetAttrString(basis_obj, "width");
    if (width_obj == nullptr) { return false; }
    width = PyLong_AsInt(width_obj);
    if (width == -1) {
        // Error already set
        return false;
    }

    PyObject* data_obj = PyObject_GetAttrString(basis_obj, "depth");
    if (data_obj == nullptr) { return false; }
    depth = PyLong_AsInt(data_obj);
    if (depth == -1) {
        // Error already set
        return false;
    }

    return true;
}

static
PyObjHandle degree_begin_from_obj(PyObject* basis_obj, int32_t depth) noexcept
{
    PyObjHandle degree_begin;
    if (PyObject_HasAttrString(basis_obj, "degree_begin")) {
        degree_begin = PyObject_GetAttrString(basis_obj, "degree_begin");

        if (!PyArray_Check(degree_begin.obj())) {
            PyErr_SetString(PyExc_TypeError,
                            "degree_begin must be a numpy array");
            return PyObjHandle();
        }

        auto* degree_begin_arr = reinterpret_cast<PyArrayObject*>(degree_begin.
            obj());
        auto const ndim = PyArray_NDIM(degree_begin_arr);

        if (ndim != 1) {
            PyErr_SetString(PyExc_TypeError, "degree_begin must be a 1D array");
            return PyObjHandle();
        }

        auto const* shape = PyArray_DIMS(degree_begin_arr);

        if (shape[0] < depth + 2) {
            PyErr_SetString(PyExc_ValueError,
                            "degree_begin must have at least depth + 2 elements");
            return PyObjHandle();
        }
    }

    return degree_begin;
}

PyObjHandle rpy::compute::to_basis(PyObject* basis_obj, TensorBasis& basis)
{

    if (basis_obj == nullptr) {
        PyErr_SetString(PyExc_ValueError, "basis object is null");
        return PyObjHandle();
    }

    if (!width_and_depth_from_obj(basis_obj, basis.width, basis.depth)) {
        return PyObjHandle();
    }

    auto degree_begin = degree_begin_from_obj(basis_obj, basis.depth);

    if (!degree_begin) {
        // For the tensor basis, we can just construct the degree_begin array // directly 
        npy_intp shape[1] = {basis.depth + 2};
        degree_begin = PyArray_SimpleNew(1, shape, NPY_INTP);
        if (!degree_begin) { return PyObjHandle(); }

        auto* degree_begin_arr = reinterpret_cast<PyArrayObject*>(degree_begin.
            obj());

        auto* data = static_cast<npy_intp*>(PyArray_DATA(degree_begin_arr));
        data[0] = 0;

        for (int32_t i = 1; i <= basis.depth + 1; ++i) {
            data[i] = 1 + basis.width * data[i - 1];
        }

        basis.degree_begin = data;
    } else {
        auto* degree_begin_arr = reinterpret_cast<PyArrayObject*>(degree_begin.
            obj());
        basis.degree_begin = static_cast<npy_intp*>(PyArray_DATA(degree_begin_arr));
    }

    return degree_begin;
}

PyObjHandle rpy::compute::to_basis(PyObject* basis_obj, LieBasis& basis)
{
    return PyObjHandle();
}
