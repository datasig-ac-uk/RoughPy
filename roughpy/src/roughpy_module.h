// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 14/03/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_ROUGHPY_MODULE_H
#define ROUGHPY_ROUGHPY_SRC_ROUGHPY_MODULE_H

#include <stdexcept>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/errors.h>

#if defined(RPY_GCC)
#  define RPY_NO_EXPORT __attribute__((visibility("hidden")))
#else
#  define RPY_NO_EXPORT
#endif

#ifndef RPY_CPP_17
// `boost::optional` as an example -- can be any `std::optional`-like container
namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>>
    : public optional_caster<boost::optional<T>> {
};
}// namespace detail
}// namespace PYBIND11_NAMESPACE
#endif

namespace py = pybind11;

namespace rpy {
namespace python {

template <typename T>
inline PyObject* cast_to_object(T&& arg) noexcept
{
    return py::cast(std::forward<T>(arg)).release().ptr();
}

inline py::object kwargs_pop(py::kwargs& kwargs, const char* name)
{
    auto arg = py::reinterpret_borrow<py::object>(kwargs[name]);
    PyDict_DelItemString(kwargs.ptr(), name);
    return arg;
}

void check_for_excess_arguments(const py::kwargs& kwargs);

template <typename T>
enable_if_t<is_base_of_v<py::object, T>, T> steal_as(py::object& obj
) noexcept { return py::reinterpret_steal<T>(obj.release().ptr()); }


/**
 * A helper method that executes a given callable and catches any exceptions
 * that might occur during its execution. The method ensures that exceptions
 * are handled gracefully, and the result of the callable is returned if no
 * exceptions are thrown.
 *
 * @param fn The callable object (function, lambda, etc.) to be executed.
 *
 * @return Returns the result from the provided callable if executed without
 *         exceptions, or any result from the exception handler if applicable.
 */
template <typename F>
bool with_caught_exceptions(F&& fn) noexcept
{
    bool completed_successfully = false;
    try {
        fn();
        completed_successfully = true;
    // } catch (const py::error_already_set& e) { // Python exception
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::domain_error& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::length_error& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }catch (const std::out_of_range& e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const std::bad_alloc& e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
    // } catch (const py::builtin_exception& e) {
    //     e.set_error();
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return completed_successfully;
}

void init_roughpy_module(py::module_& m);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_ROUGHPY_SRC_ROUGHPY_MODULE_H