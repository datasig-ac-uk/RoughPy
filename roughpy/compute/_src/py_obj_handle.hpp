#ifndef ROUGHPY_COMPUTE__SRC_PY_OBJ_HANDLE_HPP
#define ROUGHPY_COMPUTE__SRC_PY_OBJ_HANDLE_HPP

#include "py_headers.h"

#include <exception>

namespace rpy::compute {



class ObjectRef;
class MutableObjectRef;

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

    PyObjHandle(const PyObjHandle& other) noexcept
        : ptr_(other.ptr_)
    {
        Py_XINCREF(ptr_);
    }

    PyObjHandle(PyObjHandle&& other) noexcept : ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
    }

    PyObjHandle& operator=(const PyObjHandle& other) noexcept
    {
        Py_XINCREF(ptr_);
        Py_XSETREF(ptr_, other.ptr_);
        return *this;
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


    constexpr PyObject* obj() const noexcept { return ptr_;}
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

    PyObjHandle& operator+=(const ObjectRef& other);
    PyObjHandle& operator+=(const PyObjHandle& other);
    PyObjHandle& operator-=(const ObjectRef& other);
    PyObjHandle& operator-=(const PyObjHandle& other);
    PyObjHandle& operator*=(const ObjectRef& other);
    PyObjHandle& operator*=(const PyObjHandle& other);
    PyObjHandle& operator/=(const ObjectRef& other);
    PyObjHandle& operator/=(const PyObjHandle& other);
};



class ObjectRef {
    PyObject** const data_;

    friend class MutableObjectRef;
    friend class PyObjHandle;

public:
    constexpr explicit ObjectRef(PyObject** const data) : data_(data) {}

    PyObjHandle strong() const noexcept { return PyObjHandle(*data_); }

    PyObject* obj() const noexcept { return *data_; }

};



class MutableObjectRef : public ObjectRef {
    friend class ObjectRef;
    friend class PyObjHandle;

public:
    constexpr explicit MutableObjectRef(PyObject** const data) : ObjectRef(data) {}

    MutableObjectRef& operator=(const ObjectRef & rhs) noexcept
    {
        Py_SETREF(*data_, *rhs.data_);
        return *this;
    }

    MutableObjectRef& operator=(PyObjHandle&& rhs) noexcept
    {
        Py_SETREF(*data_, rhs.release());
        return *this;
    }

    MutableObjectRef& operator=(const PyObjHandle& rhs) noexcept
    {
        Py_SETREF(*data_, rhs.obj());
        return *this;
    }

#define INPLACE_BINOP(sym, pyname)                                             \
    MutableObjectRef& operator sym(const ObjectRef & rhs)                      \
    {                                                                          \
        PyObject* ret = PyNumber_##pyname(*data_, *rhs.data_);                 \
        if (ret == nullptr) { throw PyErrAlreadySet(); }                       \
        if (ret != *data_) { Py_SETREF(*data_, ret); }                         \
        return *this;                                                          \
    }                                                                          \
    MutableObjectRef& operator sym(const PyObjHandle & rhs)                    \
    {                                                                          \
        PyObject* ret = PyNumber_##pyname(*data_, rhs.obj());                  \
        if (ret == nullptr) { throw PyErrAlreadySet(); }                       \
        if (ret != *data_) { Py_SETREF(*data_, ret); }                         \
        return *this;                                                          \
    }

    INPLACE_BINOP(+=, InPlaceAdd)
    INPLACE_BINOP(-=, InPlaceSubtract)
    INPLACE_BINOP(/=, InPlaceTrueDivide)
    INPLACE_BINOP(*=, InPlaceMultiply)

#undef INPLACE_BINOP
};

#define INPLACE_BINOP(sym, pyname)                                             \
    inline PyObjHandle& PyObjHandle::operator sym(const ObjectRef & other)     \
    {                                                                          \
        PyObject* ret = PyNumber_##pyname(ptr_, *other.data_);                 \
        if (ret == nullptr) { throw PyErrAlreadySet(); }                       \
        if (ret != ptr_) { Py_SETREF(ptr_, ret); }                             \
        return *this;                                                          \
    }                                                                          \
    inline PyObjHandle& PyObjHandle::operator sym(const PyObjHandle & other)   \
    {                                                                          \
        PyObject* ret = PyNumber_##pyname(ptr_, other.ptr_);                   \
        if (ret == nullptr) { throw PyErrAlreadySet(); }                       \
        if (ret != ptr_) { Py_SETREF(ptr_, ret); }                             \
        return *this;                                                          \
    }

INPLACE_BINOP(+=, InPlaceAdd)
INPLACE_BINOP(-=, InPlaceSubtract)
INPLACE_BINOP(/=, InPlaceTrueDivide)
INPLACE_BINOP(*=, InPlaceMultiply)

#undef INPLACE_BINOP

#define BINOP(sym, pyname)                                                     \
    inline PyObjHandle operator sym(                                           \
            const ObjectRef& lhs,                                              \
            const ObjectRef& rhs                                               \
    )                                                                          \
    {                                                                          \
        PyObjHandle ret(PyNumber_##pyname(lhs.obj(), rhs.obj()), false);       \
        if (!ret) { throw PyErrAlreadySet(); }                                 \
        return ret;                                                            \
    }                                                                          \
    inline PyObjHandle operator sym(                                           \
            const PyObjHandle& lhs,                                            \
            const ObjectRef& rhs                                               \
    )                                                                          \
    {                                                                          \
        PyObjHandle ret(PyNumber_##pyname(lhs.obj(), rhs.obj()), false);       \
        if (!ret) { throw PyErrAlreadySet(); }                                 \
        return ret;                                                            \
    }                                                                          \
    inline PyObjHandle operator sym(                                           \
            const ObjectRef& lhs,                                              \
            const PyObjHandle& rhs                                             \
    )                                                                          \
    {                                                                          \
        PyObjHandle ret(PyNumber_##pyname(lhs.obj(), rhs.obj()), false);       \
        if (!ret) { throw PyErrAlreadySet(); }                                 \
        return ret;                                                            \
    }                                                                          \
    inline PyObjHandle operator sym(                                           \
            const PyObjHandle& lhs,                                            \
            const PyObjHandle& rhs                                             \
    )                                                                          \
    {                                                                          \
        PyObjHandle ret(PyNumber_##pyname(lhs.obj(), rhs.obj()), false);       \
        if (!ret) { throw PyErrAlreadySet(); }                                 \
        return ret;                                                            \
    }                                                                          \
    inline PyObjHandle operator sym(PyObjHandle&& lhs, const ObjectRef& rhs)   \
    {                                                                          \
        lhs sym## = rhs;                                                       \
        return lhs;                                                            \
    }                                                                          \
    inline PyObjHandle operator sym(PyObjHandle&& lhs, const PyObjHandle& rhs) \
    {                                                                          \
        lhs sym## = rhs;                                                       \
        return lhs;                                                            \
    }                                                                          \

BINOP(+, Add)
BINOP(-, Subtract)
BINOP(*, Multiply)
BINOP(/, TrueDivide)

#undef BINOP


inline PyObjHandle operator-(const ObjectRef& arg)
{
    PyObjHandle ret(PyNumber_Negative(arg.obj()), false);
    if (!ret) { throw PyErrAlreadySet(); }
    return ret;
}

inline PyObjHandle operator-(const PyObjHandle& arg)
{
    PyObjHandle ret(PyNumber_Negative(arg.obj()), false);
    if (!ret) { throw PyErrAlreadySet(); }
    return ret;
}

} // namespace rpy::compute

#endif //ROUGHPY_COMPUTE__SRC_PY_