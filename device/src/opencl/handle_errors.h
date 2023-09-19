//
// Created by sam on 19/09/23.
//

#ifndef ROUGHPY_HANDLE_ERRORS_H
#define ROUGHPY_HANDLE_ERRORS_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "opencl_headers.h"

namespace rpy {
namespace device {
namespace cl {

RPY_NO_RETURN void
handle_cl_error(cl_int err, const char* filename, int lineno, const char* func);

}// namespace cl
}// namespace device
}// namespace rpy

#define RPY_HANDLE_OCL_ERROR(ERR)                                              \
    ::rpy::device::cl::handle_cl_error(                                        \
            ERR, RPY_FILE_NAME, __LINE__, RPY_FUNC_NAME                        \
    )

#endif// ROUGHPY_HANDLE_ERRORS_H
