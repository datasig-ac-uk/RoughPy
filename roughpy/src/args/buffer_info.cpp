//
// Created by sam on 2/29/24.
//

#include "buffer_info.h"

#include "roughpy_module.h"
#include <roughpy/platform/errors.h>

using namespace rpy;
using namespace python;

BufferInfo::BufferInfo(PyObject* object)
{
    if (PyObject_GetBuffer(object, &m_view, PyBUF_FULL_RO) != 0) {
        RPY_THROW(py::buffer_error, "invalid buffer object");
    }
}

BufferInfo::~BufferInfo() { PyBuffer_Release(&m_view); }

BufferFormat BufferInfo::format() const
{
    BufferFormat fmt;
    return fmt;
}
