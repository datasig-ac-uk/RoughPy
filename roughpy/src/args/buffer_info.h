//
// Created by sam on 2/29/24.
//

#ifndef ROUGHPY_BUFFER_INFO_H
#define ROUGHPY_BUFFER_INFO_H

#include "roughpy_python.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/devices/core.h>

#include <boost/container/small_vector.hpp>

namespace rpy {
namespace python {

struct BufferFormat {
    boost::container::small_vector<devices::TypeInfo, 1> types;
};

class BufferInfo
{
    // This is mutable because the Python API always passes mutable pointers
    mutable Py_buffer m_view;
    optional<Py_ssize_t> m_size;

public:
    explicit BufferInfo(PyObject* object);
    ~BufferInfo();

    RPY_NO_DISCARD const byte* data() const noexcept
    {
        return reinterpret_cast<const byte*>(m_view.buf);
    }

    RPY_NO_DISCARD Py_ssize_t size() const noexcept;

    RPY_NO_DISCARD bool is_contiguous() const noexcept
    {
        return static_cast<bool>(PyBuffer_IsContiguous(&m_view, 'A'));
    }

    RPY_NO_DISCARD idimn_t itemsize() const noexcept
    {
        return static_cast<idimn_t>(m_view.itemsize);
    }
    RPY_NO_DISCARD BufferFormat format() const;

    RPY_NO_DISCARD idimn_t ndim() const noexcept
    {
        return static_cast<idimn_t>(m_view.itemsize);
    }

    RPY_NO_DISCARD const Py_ssize_t* shape() const noexcept
    {
        return m_view.shape;
    }

    RPY_NO_DISCARD const Py_ssize_t* strides() const noexcept
    {
        return m_view.strides;
    }

    RPY_NO_DISCARD const Py_ssize_t* suboffsets() const noexcept
    {
        return m_view.suboffsets;
    }

    RPY_NO_DISCARD const byte* ptr(Py_ssize_t* index) const noexcept
    {
        return reinterpret_cast<const byte*>(PyBuffer_GetPointer(&m_view, index)
        );
    }

    RPY_NO_DISCARD boost::container::small_vector<Py_ssize_t, 2>
    new_index() const
    {
        return boost::container::small_vector<Py_ssize_t, 2>(m_view.ndim);
    }
};

}// namespace python
}// namespace rpy

#endif// ROUGHPY_BUFFER_INFO_H
