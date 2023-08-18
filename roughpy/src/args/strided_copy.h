//
// Created by sam on 09/08/23.
//

#ifndef ROUGHPY_STRIDED_COPY_H
#define ROUGHPY_STRIDED_COPY_H

#include "roughpy_module.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

namespace rpy { namespace python {

void stride_copy(void* RPY_RESTRICT dst,
                 const void* RPY_RESTRICT src,
                 const py::ssize_t itemsize,
                 const py::ssize_t ndim,
                 const py::ssize_t* shape_in,
                 const py::ssize_t* strides_in,
                 const py::ssize_t* strides_out
                 ) noexcept;



}}


#endif// ROUGHPY_STRIDED_COPY_H
