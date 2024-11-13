// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 11/10/23.
//

#include "ocl_handle_errors.h"

#include "roughpy/core/check.h"  // for throw_exception

#include <roughpy/platform/errors.h>
#include <sstream>

void rpy::devices::cl::handle_cl_error(
        cl_int err, const char* filename, int lineno, const char* func
)
{
    std::stringstream msg;

    switch (err) {
        case CL_DEVICE_NOT_FOUND:
        case CL_DEVICE_NOT_AVAILABLE:
        case CL_COMPILER_NOT_AVAILABLE:
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY:
        case CL_PROFILING_INFO_NOT_AVAILABLE:
        case CL_MEM_COPY_OVERLAP:
        case CL_IMAGE_FORMAT_MISMATCH:
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        case CL_BUILD_PROGRAM_FAILURE:
        case CL_MAP_FAILURE:
#ifdef CL_VERSION_1_1
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
#endif
#ifdef CL_VERSION_1_2
        case CL_COMPILE_PROGRAM_FAILURE:
        case CL_LINKER_NOT_AVAILABLE:
        case CL_LINK_PROGRAM_FAILURE:
        case CL_DEVICE_PARTITION_FAILED:
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
#endif

        case CL_INVALID_VALUE:
        case CL_INVALID_DEVICE_TYPE:
        case CL_INVALID_PLATFORM:
        case CL_INVALID_DEVICE:
        case CL_INVALID_CONTEXT:
        case CL_INVALID_QUEUE_PROPERTIES:
        case CL_INVALID_COMMAND_QUEUE:
        case CL_INVALID_HOST_PTR:
        case CL_INVALID_MEM_OBJECT:
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        case CL_INVALID_IMAGE_SIZE:
        case CL_INVALID_SAMPLER:
        case CL_INVALID_BINARY:
        case CL_INVALID_BUILD_OPTIONS:
        case CL_INVALID_PROGRAM:
        case CL_INVALID_PROGRAM_EXECUTABLE:
        case CL_INVALID_KERNEL_NAME:
        case CL_INVALID_KERNEL_DEFINITION:
        case CL_INVALID_KERNEL:
        case CL_INVALID_ARG_INDEX:
        case CL_INVALID_ARG_VALUE:
        case CL_INVALID_KERNEL_ARGS:
        case CL_INVALID_WORK_DIMENSION:
        case CL_INVALID_WORK_GROUP_SIZE:
        case CL_INVALID_WORK_ITEM_SIZE:
        case CL_INVALID_GLOBAL_OFFSET:
        case CL_INVALID_EVENT_WAIT_LIST:
        case CL_INVALID_EVENT:
        case CL_INVALID_OPERATION:
        case CL_INVALID_GL_OBJECT:
        case CL_INVALID_BUFFER_SIZE:
        case CL_INVALID_MIP_LEVEL:
        case CL_INVALID_GLOBAL_WORK_SIZE:
#ifdef CL_VERSION_1_1
        case CL_INVALID_PROPERTY:
#endif
#ifdef CL_VERSION_1_2
        case CL_INVALID_IMAGE_DESCRIPTOR:
        case CL_INVALID_COMPILER_OPTIONS:
        case CL_INVALID_LINKER_OPTIONS:
        case CL_INVALID_DEVICE_PARTITION_COUNT:
#endif
#ifdef CL_VERSION_2_0
        case CL_INVALID_PIPE_SIZE:
        case CL_INVALID_DEVICE_QUEUE:
#endif
#ifdef CL_VERSION_2_2
        case CL_INVALID_SPEC_ID:
        case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
#endif
        default:
            msg << "Error code " << err << " occurred";
            errors::throw_exception<std::runtime_error>(
                    msg.str(),
                    filename,
                    lineno,
                    func
            );
    }
}
