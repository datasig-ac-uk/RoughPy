#ifndef ROUGHPY_PYCORE_OBJECT_HANDLE_HPP
#define ROUGHPY_PYCORE_OBJECT_HANDLE_HPP

#include "py_headers.h"

namespace rpy {

class PyObjHandle
{
    PyObject* ptr_ = nullptr;

public:

    PyObjHandle() = default;

    PyObjHandle(PyObject* ptr, bool incref=true) : ptr_(ptr)
    {
        if (incref) {
            Py_INCREF(ptr_);
        }
    }

    ~PyObjHandle()
    {
        Py_XDECREF(ptr_);
    }

    PyObjHandle(PyObjHandle&& other) noexcept : ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
    }

    PyObjHandle& operator=(PyObjHandle&& other) noexcept
    {
        Py_XSETREF(ptr_, other.ptr_);
        other.ptr_ = nullptr;
        return *this;
    }

    void reset(PyObject* new_obj, bool incref=true)
    {
        if (incref) { Py_INCREF(ptr_); }
        Py_XSETREF(ptr_, new_obj);
    }

    // Must be followed by inc_ref if the object should be preserved
    constexpr PyObject*& obj() { return ptr_; }
    void drop() noexcept
    {
        Py_XDECREF(ptr_);
        ptr_ = nullptr;
    }

    void inc_ref() noexcept { Py_INCREF(ptr_); }

    PyObject* release() noexcept
    {
        auto tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

    explicit constexpr operator bool() const noexcept
    {
        return ptr_ != nullptr;
    }

    PyObjHandle& operator=(PyObject* obj) noexcept
    {
        Py_INCREF(obj);
        Py_XSETREF(ptr_, obj);
        // Py_XDECREF(ptr);
        // ptr = obj;
        // Py_INCREF(ptr);
        return *this;
    }
};




} // namespace rpy


#endif//ROUGHPY_PYCORE_OBJECT_HANDLE_HPP
