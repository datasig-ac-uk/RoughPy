//
// Created by sam on 1/17/24.
//

#ifndef ROUGHPY_SCALAR_H
#define ROUGHPY_SCALAR_H

#include "roughpy_module.h"

#include <roughpy/scalars/scalar.h>

namespace rpy {
namespace python {

extern "C" PyTypeObject PyScalar_Type;

struct PyScalar {
    PyObject_HEAD scalars::Scalar m_content;
};

inline bool PyScalar_Check(PyObject* obj) noexcept
{
    return Py_TYPE(obj) == &PyScalar_Type;
}

inline scalars::Scalar& cast_pyscalar_mut(PyObject* obj) noexcept
{
    return reinterpret_cast<PyScalar*>(obj)->m_content;
}

inline const scalars::Scalar& cast_pyscalar(PyObject* obj) noexcept
{
    return reinterpret_cast<PyScalar*>(obj)->m_content;
}

class PyScalarProxy
{
    optional<scalars::Scalar> m_converted;
    PyObject* p_object;
    bool m_needs_conversion;

    void do_conversion();

public:
    PyScalarProxy(PyObject* obj)
        : p_object(obj),
          m_needs_conversion(PyScalar_Check(obj))
    {}

    bool is_scalar() const noexcept { return !m_needs_conversion; }

    const scalars::Scalar& ref()
    {
        if (m_needs_conversion) {
            if (!m_converted) { do_conversion(); }
            RPY_DBG_ASSERT(m_converted);
            return *m_converted;
        }
        return reinterpret_cast<PyScalar*>(p_object)->m_content;
    }

    scalars::Scalar& mut_ref()
    {
        if (m_needs_conversion) {
            if (!m_converted) { do_conversion(); }
            RPY_DBG_ASSERT(m_converted);
            return *m_converted;
        }
        return reinterpret_cast<PyScalar*>(p_object)->m_content;
    }

    PyObject* object() const noexcept { return p_object; }
};

// PyObject* PyScalar_FromScalar(scalars::Scalar&& obj);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_SCALAR_H
