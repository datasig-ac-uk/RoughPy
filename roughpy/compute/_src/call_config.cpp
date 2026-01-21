#include "call_config.hpp"
#include <roughpy/pycore/compat.h>

#include <roughpy_compute/common/basis.hpp>


using namespace rpy;
using namespace rpy::compute;


bool rpy::compute::update_algebra_params(CallConfig& config, npy_intp n_args, npy_intp const* arg_basis_mapping)
{
    for (npy_intp i=0; i<n_args; ++i) {
        auto& degree_bounds = config.degree_bounds[i];

        const auto* basis = config.basis_data[arg_basis_mapping[i]];

        if (degree_bounds.max_degree == -1 || degree_bounds.max_degree > basis->depth) {
            degree_bounds.max_degree = basis->depth;
        }

        if (degree_bounds.min_degree > degree_bounds.max_degree) {
            PyErr_Format(PyExc_ValueError,
                "invalid degree bounds for argument %zd:"
                " min_degree must not exceed max_degree", i);
            return false;
        }
    }

    return true;
}


static bool width_and_depth_from_obj(PyObject* basis_obj,
                                     int32_t& width,
                                     int32_t& depth) noexcept
{
    PyObject* width_obj = PyObject_GetAttrString(basis_obj, "width");
    if (width_obj == nullptr) { return false; }
    int ret = PyLong_AsInt32(width_obj, &width);
    Py_DECREF(width_obj);
    if (ret == -1) {
        // Error already set
        return false;
    }

    PyObject* data_obj = PyObject_GetAttrString(basis_obj, "depth");
    if (data_obj == nullptr) { return false; }
    ret = PyLong_AsInt32(data_obj, &depth);
    Py_DECREF(data_obj);
    if (ret == -1) {
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
        // PyObject_GetAttrString returns a new reference to the attribute
        degree_begin.reset(PyObject_GetAttrString(basis_obj, "degree_begin"), false);

        if (!PyArray_Check(degree_begin.obj())) {
            return PyObjHandle();
        }

        auto* degree_begin_arr = reinterpret_cast<PyArrayObject*>(degree_begin.
            obj());
        auto const ndim = PyArray_NDIM(degree_begin_arr);

        if (ndim != 1) {
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

LieBasisArrayHolder rpy::compute::to_basis(PyObject* basis_obj, LieBasis& basis)
{
    if (basis_obj == nullptr) {
        PyErr_SetString(PyExc_ValueError, "basis object is null");
        return {};
    }

    if (!width_and_depth_from_obj(basis_obj, basis.width, basis.depth)) {
        return {};
    }

    LieBasisArrayHolder array_holder;
    /*
     * The main challenge here is unpacking the Lie basis data to a numpy array.
     * If there is no data attribute then it is not a valid basis object. Unlike
     * the tensor basis, we cannot just fill in the gaps by ourselves because
     * the nature of the basis might not be the standard one that we provide.
     */
    PyObject* data_obj = PyObject_GetAttrString(basis_obj, "data");
    if (data_obj == nullptr) { return {}; }

    /*
     * Let's make sure the data we do get is contiguous and of the correct shape
     * otherwise we're wasting our time.
     */
    PyArray_Descr* intp_descr = PyArray_DescrFromType(NPY_INTP);
    Py_INCREF(intp_descr);
    array_holder.data = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
            data_obj,
            intp_descr,
            2,
            2,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
            nullptr
    ));
    Py_DECREF(data_obj);

    if (array_holder.data == nullptr) {
        Py_DECREF(intp_descr);
        return {};
    }

    const npy_intp* data_shape = PyArray_SHAPE(array_holder.data);
    if (data_shape[1] != 2) {
        PyErr_SetString(PyExc_ValueError, "data array must have shape (n, 2)");
        Py_DECREF(intp_descr);
        return {};
    }
    basis.data = static_cast<const npy_intp*>(PyArray_DATA(array_holder.data));

    PyObject* degree_begin_obj = PyObject_GetAttrString(basis_obj, "degree_begin");
    if (degree_begin_obj == nullptr) {
        Py_DECREF(intp_descr);
        return {};
    }

    array_holder.degree_begin = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
                    degree_begin_obj,
                    intp_descr,
                    1,
                    1,
                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                    nullptr
            ));
    Py_DECREF(degree_begin_obj);
    // At this point we have had intp_descr stolen twice, so it is no longer valid
    intp_descr = nullptr;

    if (array_holder.degree_begin == nullptr) {
        return {};
    }

    if (PyArray_SHAPE(array_holder.degree_begin)[0] < basis.depth + 1) {
        PyErr_SetString(PyExc_ValueError,
            "degree_begin array must have at least depth + 1 elements");
        return {};
    }

    basis.degree_begin = static_cast<const npy_intp*>(PyArray_DATA(array_holder.degree_begin));

    return array_holder;
}
