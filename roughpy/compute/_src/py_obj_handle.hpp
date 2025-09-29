#ifndef ROUGHPY_COMPUTE__SRC_PY_OBJ_HANDLE_HPP
#define ROUGHPY_COMPUTE__SRC_PY_OBJ_HANDLE_HPP

#include "py_headers.h"

namespace rpy::compute {

class PyObjHandle
{
    PyObject* ptr = nullptr;

public:

    PyObjHandle() = default;

    PyObjHandle(PyObject* ptr, bool incref=true) : ptr(ptr)
    {
        if (incref) {
            Py_INCREF(ptr);
        }
    }

    ~PyObjHandle()
    {
        Py_XDECREF(ptr);
    }

    PyObjHandle(PyObjHandle&& other) noexcept : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }

    PyObjHandle& operator=(PyObjHandle&& other) noexcept
    {
        Py_XSETREF(ptr, other.ptr);
        other.ptr = nullptr;
        return *this;
    }

    void reset(PyObject* new_obj, bool incref=true)
    {
        if (incref) { Py_INCREF(ptr); }
        Py_XSETREF(ptr, new_obj);
    }

    // Must be followed by inc_ref if the object should be preserved
    constexpr PyObject*& obj() { return ptr; }
    void drop() noexcept
    {
        Py_XDECREF(ptr);
        ptr = nullptr;
    }

    void inc_ref() noexcept { Py_INCREF(ptr); }

    PyObject* release() noexcept
    {
        auto tmp = ptr;
        ptr = nullptr;
        return tmp;
    }

    explicit constexpr operator bool() const noexcept
    {
        return ptr != nullptr;
    }

    PyObjHandle& operator=(PyObject* obj) noexcept
    {
        Py_INCREF(obj);
        Py_XSETREF(ptr, obj);
        // Py_XDECREF(ptr);
        // ptr = obj;
        // Py_INCREF(ptr);
        return *this;
    }
};


} // namespace rpy::compute

#endif //ROUGHPY_COMPUTE__SRC_PY_OBJ_HANDLE_HPP