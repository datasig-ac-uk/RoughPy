//
// Created by sam on 1/17/24.
//

#include "scalar.h"
#include "r_py_polynomial.h"
#include "scalar_type.h"
#include "scalars.h"

#include <functional>

using namespace rpy;
using namespace rpy::python;

scalars::Scalar pyobject_to_scalar(PyObject* obj);

void PyScalarProxy::do_conversion()
{
    if (!PyScalar_Check(p_object)) {
        m_converted = scalars::Scalar();
        assign_py_object_to_scalar(*m_converted, p_object);
    }
}

namespace {

template <typename Op>
inline PyObject*
do_arithmetic(PyScalarProxy lhs, PyScalarProxy rhs, Op&& op) noexcept
{
    try {
        return PyScalar_FromScalar(op(lhs.ref(), rhs.ref()));
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_ArithmeticError, exc.what());
        return nullptr;
    }
}

struct ScalarInplaceAdd {
    scalars::Scalar&
    operator()(scalars::Scalar& lhs, const scalars::Scalar& rhs) const
    {
        return lhs += rhs;
    }

    scalars::Scalar
    operator()(const scalars::Scalar& lhs, const scalars::Scalar& rhs) const
    {
        return lhs + rhs;
    }
};

struct ScalarInplaceSub {
    scalars::Scalar&
    operator()(scalars::Scalar& lhs, const scalars::Scalar& rhs) const
    {
        return lhs -= rhs;
    }
    scalars::Scalar
    operator()(const scalars::Scalar& lhs, const scalars::Scalar& rhs) const
    {
        return lhs - rhs;
    }
};

struct ScalarInplaceMul {
    scalars::Scalar&
    operator()(scalars::Scalar& lhs, const scalars::Scalar& rhs) const
    {
        return lhs *= rhs;
    }
    scalars::Scalar
    operator()(const scalars::Scalar& lhs, const scalars::Scalar& rhs) const
    {
        return lhs * rhs;
    }
};

struct ScalarInplaceDiv {
    scalars::Scalar&
    operator()(scalars::Scalar& lhs, const scalars::Scalar& rhs) const
    {
        return lhs /= rhs;
    }
    scalars::Scalar
    operator()(const scalars::Scalar& lhs, const scalars::Scalar& rhs) const
    {
        return lhs / rhs;
    }
};

template <typename Op>
inline PyObject*
do_inplace_arithmetic(PyScalarProxy lhs, PyScalarProxy rhs, Op&& op)
{
    if (!lhs.is_scalar()) {
        return do_arithmetic(
                std::move(lhs),
                std::move(rhs),
                std::forward<Op>(op)
        );
    }

    try {
        op(lhs.mut_ref(), rhs.ref());
        // Py_INCREF(lhs.object());
        // return lhs.object();
        Py_RETURN_NONE;
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_ArithmeticError, exc.what());
        return nullptr;
    }
}

template <typename Comp>
PyObject* do_compare(PyScalarProxy lhs, PyScalarProxy rhs, Comp&& comp) noexcept
{
    try {
        if (comp(lhs.ref(), rhs.ref())) { Py_RETURN_TRUE; }
        Py_RETURN_FALSE;
    } catch (...) {
        Py_RETURN_NOTIMPLEMENTED;
    }
}

}// namespace

extern "C" {

static PyObject* PyScalar_type(PyObject* self)
{
    const auto& scalar = cast_pyscalar(self);
    auto tp = scalar.type();
    if (tp) { return PyScalarType_FromScalarType(*tp); }

    Py_INCREF(&PyLong_Type);
    return reinterpret_cast<PyObject*>(&PyLong_Type);
}

static PyObject* PyScalar_to_float(PyObject* self)
{
    const auto& scalar = cast_pyscalar(self);

    try {
        return PyFloat_FromDouble(scalars::scalar_cast<scalar_t>(scalar));
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, "unable to convert to float");
        return nullptr;
    }
}

static PyMethodDef PyScalar_methods[] = {
        {"type", (PyCFunction) PyScalar_type, METH_METHOD | METH_NOARGS},
        {"to_float", (PyCFunction) PyScalar_to_float, METH_METHOD | METH_NOARGS
        },
        {nullptr, nullptr, 0, nullptr}
};

static PyObject* PyScalar_add(PyObject* self, PyObject* other)
{
    std::plus<scalars::Scalar> op;
    return do_arithmetic(self, other, op);
}

static PyObject* PyScalar_sub(PyObject* self, PyObject* other)
{
    std::minus<scalars::Scalar> op;
    return do_arithmetic(self, other, op);
}

static PyObject* PyScalar_mul(PyObject* self, PyObject* other)
{
    std::multiplies<scalars::Scalar> op;
    return do_arithmetic(self, other, op);
}

static PyObject*
PyScalar_rem(PyObject* RPY_UNUSED_VAR self, PyObject* RPY_UNUSED_VAR other)
{
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyScalar_pow(
        PyObject* RPY_UNUSED_VAR self,
        PyObject* RPY_UNUSED_VAR other,
        PyObject* RPY_UNUSED_VAR modulo
)
{
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyScalar_neg(PyObject* self)
{
    try {
        return PyScalar_FromScalar(-cast_pyscalar(self));
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_ArithmeticError, exc.what());
        return nullptr;
    }
}

static PyObject* PyScalar_pos(PyObject* self)
{
    try {
        return PyScalar_FromScalar(scalars::Scalar(cast_pyscalar(self)));
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_ArithmeticError, exc.what());
        return nullptr;
    }
}

static int PyScalar_bool(PyObject* self)
{
    return cast_pyscalar(self).is_zero() ? 0 : 1;
}

static PyObject* PyScalar_inplace_add(PyObject* self, PyObject* other)
{
    ScalarInplaceAdd add;
    return do_inplace_arithmetic(self, other, add);
}

static PyObject* PyScalar_inplace_sub(PyObject* self, PyObject* other)
{
    ScalarInplaceSub sub;
    return do_inplace_arithmetic(self, other, sub);
}

static PyObject* PyScalar_inplace_mul(PyObject* self, PyObject* other)
{
    ScalarInplaceMul mul;
    return do_inplace_arithmetic(self, other, mul);
}

static PyObject* PyScalar_inplace_rem(
        PyObject* RPY_UNUSED_VAR self,
        PyObject* RPY_UNUSED_VAR other
)
{
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyScalar_inplace_pow(
        PyObject* RPY_UNUSED_VAR self,
        PyObject* RPY_UNUSED_VAR other,
        PyObject* RPY_UNUSED_VAR modulo
)
{
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject*
PyScalar_floordiv(PyObject* RPY_UNUSED_VAR self, PyObject* RPY_UNUSED_VAR other)
{
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyScalar_div(PyObject* self, PyObject* other)
{
    ScalarInplaceDiv div;
    return do_inplace_arithmetic(self, other, div);
}

static PyObject* PyScalar_inplace_floordiv(
        PyObject* RPY_UNUSED_VAR self,
        PyObject* RPY_UNUSED_VAR other
)
{
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject* PyScalar_inplace_div(
        PyObject* RPY_UNUSED_VAR self,
        PyObject* RPY_UNUSED_VAR other
)
{
    Py_RETURN_NOTIMPLEMENTED;
}

static PyNumberMethods PyScalar_number{
        (binaryfunc) PyScalar_add,              /* nb_add */
        (binaryfunc) PyScalar_sub,              /* nb_subtract */
        (binaryfunc) PyScalar_mul,              /* nb_multiply */
        (binaryfunc) PyScalar_rem,              /* nb_remainder */
        nullptr,                                /* nb_divmod */
        (ternaryfunc) PyScalar_pow,             /* nb_power */
        PyScalar_neg,                           /* nb_negative */
        PyScalar_pos,                           /* nb_positive */
        nullptr,                                /* nb_absolute */
        (inquiry) PyScalar_bool,                /* nb_bool */
        nullptr,                                /* nb_invert */
        nullptr,                                /* nb_lshift */
        nullptr,                                /* nb_rshift */
        nullptr,                                /* nb_and */
        nullptr,                                /* nb_xor */
        nullptr,                                /* nb_or */
        nullptr,                                /* nb_int */
        nullptr,                                /* nb_reserved */
        nullptr,                                /* nb_float */
        (binaryfunc) PyScalar_inplace_add,      /* nb_inplace_add */
        (binaryfunc) PyScalar_inplace_sub,      /* nb_inplace_subtract */
        (binaryfunc) PyScalar_inplace_mul,      /* nb_inplace_multiply */
        (binaryfunc) PyScalar_inplace_rem,      /* nb_inplace_remainder */
        (ternaryfunc) PyScalar_inplace_pow,     /* nb_inplace_power */
        nullptr,                                /* nb_inplace_lshift */
        nullptr,                                /* nb_inplace_rshift */
        nullptr,                                /* nb_inplace_and */
        nullptr,                                /* nb_inplace_xor */
        nullptr,                                /* nb_inplace_or */
        (binaryfunc) PyScalar_floordiv,         /* nb_floor_divide */
        (binaryfunc) PyScalar_div,              /* nb_true_divide */
        (binaryfunc) PyScalar_inplace_floordiv, /* nb_inplace_floor_divide */
        (binaryfunc) PyScalar_inplace_div,      /* nb_inplace_true_div */
        nullptr,                                /* nb_index */
        nullptr,                                /* nb_matrix_multiply */
        nullptr                                 /* nb_inplace_matrix_multiply */
};

static void PyScalar_finalize(PyObject* self)
{
    cast_pyscalar_mut(self).~Scalar();
}

static PyObject* PyScalar_repr(PyObject* self)
{
    try {
        std::stringstream ss;
        ss << cast_pyscalar(self);
        return PyUnicode_FromString(ss.str().c_str());
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}

static PyObject* PyScalar_str(PyObject* self)
{
    try {
        std::stringstream ss;
        ss << cast_pyscalar(self);
        return PyUnicode_FromString(ss.str().c_str());
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}

static PyObject*
PyScalar_richcmp(PyObject* self, PyObject* other, const int optype)
{
    switch (optype) {
        case Py_EQ:
            return do_compare(self, other, std::equal_to<scalars::Scalar>());
        case Py_NE:
            return do_compare(
                    self,
                    other,
                    std::not_equal_to<scalars::Scalar>()
            );
        case Py_LT:
        case Py_LE:
        case Py_GT:
        case Py_GE:
        default: break;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

PyTypeObject rpy::python::PyScalar_Type = {
        PyVarObject_HEAD_INIT(nullptr, 0)//
        "roughpy.Scalar",                /* tp_name */
        sizeof(PyScalar),                /* tp_basicsize */
        0,                               /* tp_itemsize */
        nullptr,                         /* tp_dealloc */
        0,                               /* tp_vectorcall_offset */
        nullptr,                         /* tp_getattr */
        nullptr,                         /* tp_setattr */
        nullptr,                         /* tp_as_async */
        PyScalar_repr,                   /* tp_repr */
        &PyScalar_number,                /* tp_as_number */
        nullptr,                         /* tp_as_sequence */
        nullptr,                         /* tp_as_mapping */
        nullptr,                         /* tp_hash */
        nullptr,                         /* tp_call */
        PyScalar_str,                    /* tp_str */
        nullptr,                         /* tp_getattro */
        nullptr,                         /* tp_setattro */
        nullptr,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,              /* tp_flags */
        PyDoc_STR("My objects"),         /* tp_doc */
        nullptr,                         /* tp_traverse */
        nullptr,                         /* tp_clear */
        PyScalar_richcmp,                /* tp_richcompare */
        0,                               /* tp_weaklistoffset */
        nullptr,                         /* tp_iter */
        nullptr,                         /* tp_iternext */
        nullptr,                         /* tp_methods */
        nullptr,                         /* tp_members */
        nullptr,                         /* tp_getset */
        nullptr,                         /* tp_base */
        nullptr,                         /* tp_dict */
        nullptr,                         /* tp_descr_get */
        nullptr,                         /* tp_descr_set */
        0,                               /* tp_dictoffset */
        nullptr,                         /* tp_init */
        nullptr,                         /* tp_alloc */
        nullptr,                         /* tp_new */
        nullptr,                         /* tp_free */
        nullptr,                         /* tp_is_gc */
        nullptr,                         /* tp_bases */
        nullptr,                         /* tp_mro */
        nullptr,                         /* tp_cache */
        nullptr,                         /* tp_subclasses */
        nullptr,                         /* tp_weaklist */
        nullptr,                         /* tp_del */
        0,                               /* tp_version_tag */
        PyScalar_finalize,               /* tp_finalize */
        nullptr                          /* tp_vectorcall */
};
}

scalars::Scalar pyobject_to_scalar(PyObject* obj)
{
    scalars::Scalar result;

    if (RPyPolynomial_Check(obj)) {
        result = RPyPolynomial_cast(obj);
    } else if (PyFloat_CheckExact(obj)) {
        result = PyFloat_AsDouble(obj);
    } else if (PyLong_CheckExact(obj)) {
        result = PyLong_AsLongLong(obj);
    } else {
        result = PyFloat_AsDouble(obj);
    }

    return result;
}
