//
// Created by user on 14/03/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_ROUGHPY_MODULE_H
#define ROUGHPY_ROUGHPY_SRC_ROUGHPY_MODULE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <roughpy/config/implementation_types.h>

// `boost::optional` as an example -- can be any `std::optional`-like container
namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};
}}

namespace py = pybind11;
namespace rpy { namespace python {



}}


#endif//ROUGHPY_ROUGHPY_SRC_ROUGHPY_MODULE_H
