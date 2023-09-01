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
// Created by user on 24/08/23.
//

#ifndef ROUGHPY_DEVICE_SRC_OPENCL_OPEN_CL_RUNTIME_LIBRARY_H_
#define ROUGHPY_DEVICE_SRC_OPENCL_OPEN_CL_RUNTIME_LIBRARY_H_

#include <roughpy/platform/runtime_library.h>

#include <roughpy/device/buffer.h>
#include <roughpy/device/device_handle.h>
#include <roughpy/device/event.h>
#include <roughpy/device/kernel.h>
#include <roughpy/device/queue.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>
#include <CL/cl_platform.h>

#include <mutex>
#include <unordered_map>

#include <boost/functional/hash.hpp>

namespace rpy {
namespace device {

namespace cl {

extern "C" {
/* Platform API */
typedef cl_int(CL_API_CALL*
                       FclGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

/* Device APIs */
typedef cl_int(CL_API_CALL*
                       FclGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_2

typedef cl_int(CL_API_CALL*
                       FclCreateSubDevices)(cl_device_id, const cl_device_partition_property*, cl_uint, cl_device_id*, cl_uint*)
        CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* FclRetainDevice)(cl_device_id
) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* FclReleaseDevice)(cl_device_id
) CL_API_SUFFIX__VERSION_1_2;

#endif

#ifdef CL_VERSION_2_1

typedef cl_int(CL_API_CALL* FclSetDefaultDeviceCommandQueue)(
        cl_context, cl_device_id, cl_command_queue
) CL_API_SUFFIX__VERSION_2_1;

typedef cl_int(CL_API_CALL*
                       FclGetDeviceAndHostTimer)(cl_device_id, cl_ulong*, cl_ulong*)
        CL_API_SUFFIX__VERSION_2_1;

typedef cl_int(CL_API_CALL* FclGetHostTimer)(cl_device_id, cl_ulong*)
        CL_API_SUFFIX__VERSION_2_1;

#endif

/* Context APIs */
typedef cl_context(CL_API_CALL*
                           FclCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void(CL_CALLBACK*)(const char*, const void*, size_t c, void*), void*, cl_int*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_context(CL_API_CALL*
                           FclCreateContextFromType)(const cl_context_properties*, cl_device_type, void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclRetainContext)(cl_context
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclReleaseContext)(cl_context
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetContextInfo)(cl_context, cl_context_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_3_0

typedef cl_int(CL_API_CALL*
                       FclSetContextDestructorCallback)(cl_context, void(CL_CALLBACK*)(cl_context, void*), void*)
        CL_API_SUFFIX__VERSION_3_0;

#endif

/* Command Queue APIs */

#ifdef CL_VERSION_2_0

typedef cl_command_queue(CL_API_CALL*
                                 FclCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_queue_properties*, cl_int*)
        CL_API_SUFFIX__VERSION_2_0;

#endif

typedef cl_int(CL_API_CALL* FclRetainCommandQueue)(cl_command_queue
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclReleaseCommandQueue)(cl_command_queue
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetCommandQueueInfo)(cl_command_queue, cl_command_queue_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

/* Memory Object APIs */
typedef cl_mem(CL_API_CALL*
                       FclCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_1

typedef cl_mem(CL_API_CALL*
                       FclCreateSubBuffer)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void*, cl_int*)
        CL_API_SUFFIX__VERSION_1_1;

#endif

#ifdef CL_VERSION_1_2

typedef cl_mem(CL_API_CALL*
                       FclCreateImage)(cl_context, cl_mem_flags, const cl_image_format*, const cl_image_desc*, void*, cl_int*)
        CL_API_SUFFIX__VERSION_1_2;

#endif

#ifdef CL_VERSION_2_0

typedef cl_mem(CL_API_CALL*
                       FclCreatePipe)(cl_context, cl_mem_flags, cl_uint, cl_uint, const cl_pipe_properties*, cl_int*)
        CL_API_SUFFIX__VERSION_2_0;

#endif

#ifdef CL_VERSION_3_0

typedef cl_mem(CL_API_CALL*
                       FclCreateBufferWithProperties)(cl_context, const cl_mem_properties*, cl_mem_flags, size_t, void*, cl_int*)
        CL_API_SUFFIX__VERSION_3_0;

typedef cl_mem(CL_API_CALL*
                       FclCreateImageWithProperties)(cl_context, const cl_mem_properties*, cl_mem_flags, const cl_image_format*, const cl_image_desc*, void*, cl_int*)
        CL_API_SUFFIX__VERSION_3_0;

#endif

typedef cl_int(CL_API_CALL* FclRetainMemObject)(cl_mem
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclReleaseMemObject)(cl_mem
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetSupportedImageFormats)(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format*, cl_uint*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetMemObjectInfo)(cl_mem, cl_mem_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetImageInfo)(cl_mem, cl_image_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_2_0

typedef cl_int(CL_API_CALL*
                       FclGetPipeInfo)(cl_mem, cl_pipe_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_2_0;

#endif

#ifdef CL_VERSION_1_1

typedef cl_int(CL_API_CALL*
                       FclSetMemObjectDestructorCallback)(cl_mem, void(CL_CALLBACK*)(cl_mem, void*), void*)
        CL_API_SUFFIX__VERSION_1_1;

#endif

/* SVM Allocation APIs */

#ifdef CL_VERSION_2_0

typedef void*(CL_API_CALL* FclSVMAlloc)(
        cl_context, cl_svm_mem_flags, size_t, cl_uint
) CL_API_SUFFIX__VERSION_2_0;

typedef void(CL_API_CALL* FclSVMFree)(cl_context, void*)
        CL_API_SUFFIX__VERSION_2_0;

#endif

/* Sampler APIs */

#ifdef CL_VERSION_2_0

typedef cl_sampler(CL_API_CALL*
                           FclCreateSamplerWithProperties)(cl_context, const cl_sampler_properties*, cl_int*)
        CL_API_SUFFIX__VERSION_2_0;

#endif

typedef cl_int(CL_API_CALL* FclRetainSampler)(cl_sampler
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclReleaseSampler)(cl_sampler
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetSamplerInfo)(cl_sampler, cl_sampler_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

/* Program Object APIs */
typedef cl_program(CL_API_CALL*
                           FclCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_program(CL_API_CALL*
                           FclCreateProgramWithBinary)(cl_context, cl_uint, const cl_device_id*, const size_t*, const unsigned char**, cl_int*, cl_int*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_2

typedef cl_program(CL_API_CALL*
                           FclCreateProgramWithBuiltInKernels)(cl_context, cl_uint, const cl_device_id*, const char*, cl_int*)
        CL_API_SUFFIX__VERSION_1_2;

#endif

#ifdef CL_VERSION_2_1

typedef cl_program(CL_API_CALL*
                           FclCreateProgramWithIL)(cl_context, const void*, size_t, cl_int*)
        CL_API_SUFFIX__VERSION_2_1;

#endif

typedef cl_int(CL_API_CALL* FclRetainProgram)(cl_program
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclReleaseProgram)(cl_program
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void(CL_CALLBACK*)(cl_program, void*), void*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_2

typedef cl_int(CL_API_CALL*
                       FclCompileProgram)(cl_program, cl_uint, const cl_device_id*, const char*, cl_uint, const cl_program*, const char**, void(CL_CALLBACK*)(cl_program, void*), void*)
        CL_API_SUFFIX__VERSION_1_2;

typedef cl_program(CL_API_CALL*
                           FclLinkProgram)(cl_context, cl_uint, const cl_device_id*, const char*, cl_uint, const cl_program*, void(CL_CALLBACK*)(cl_program, void*), void*, cl_int*)
        CL_API_SUFFIX__VERSION_1_2;

#endif

#ifdef CL_VERSION_2_2

typedef CL_API_PREFIX__VERSION_2_2_DEPRECATED
cl_int(CL_API_CALL*
               FclSetProgramReleaseCallback)(cl_program, void(CL_CALLBACK*)(cl_program, void*), void*)
        CL_API_SUFFIX__VERSION_2_2_DEPRECATED;

typedef cl_int(CL_API_CALL*
                       FclSetProgramSpecializationConstant)(cl_program, cl_uint, size_t, const void*)
        CL_API_SUFFIX__VERSION_2_2;

#endif

#ifdef CL_VERSION_1_2

typedef cl_int(CL_API_CALL* FclUnloadPlatformCompiler)(cl_platform_id
) CL_API_SUFFIX__VERSION_1_2;

#endif

typedef cl_int(CL_API_CALL*
                       FclGetProgramInfo)(cl_program, cl_program_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

/* Kernel Object APIs */
typedef cl_kernel(CL_API_CALL*
                          FclCreateKernel)(cl_program, const char*, cl_int*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclCreateKernelsInProgram)(cl_program, cl_uint, cl_kernel*, cl_uint*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_2_1

typedef cl_kernel(CL_API_CALL* FclCloneKernel)(cl_kernel, cl_int*)
        CL_API_SUFFIX__VERSION_2_1;

#endif

typedef cl_int(CL_API_CALL* FclRetainKernel)(cl_kernel
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclReleaseKernel)(cl_kernel
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclSetKernelArg)(cl_kernel, cl_uint, size_t, const void*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_2_0

typedef cl_int(CL_API_CALL*
                       FclSetKernelArgSVMPointer)(cl_kernel, cl_uint, const void*)
        CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL*
                       FclSetKernelExecInfo)(cl_kernel, cl_kernel_exec_info, size_t, const void*)
        CL_API_SUFFIX__VERSION_2_0;

#endif

typedef cl_int(CL_API_CALL*
                       FclGetKernelInfo)(cl_kernel, cl_kernel_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_2

typedef cl_int(CL_API_CALL*
                       FclGetKernelArgInfo)(cl_kernel, cl_uint, cl_kernel_arg_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_2;

#endif
typedef cl_int(CL_API_CALL*
                       FclGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_2_1

typedef cl_int(CL_API_CALL*
                       FclGetKernelSubGroupInfo)(cl_kernel, cl_device_id, cl_kernel_sub_group_info, size_t, const void*, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_2_1;

#endif

/* Event Object APIs */
typedef cl_int(CL_API_CALL* FclWaitForEvents)(cl_uint, const cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclGetEventInfo)(cl_event, cl_event_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_1

typedef cl_event(CL_API_CALL* FclCreateUserEvent)(cl_context, cl_int*)
        CL_API_SUFFIX__VERSION_1_1;

#endif

typedef cl_int(CL_API_CALL* FclRetainEvent)(cl_event
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclReleaseEvent)(cl_event
) CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_1

typedef cl_int(CL_API_CALL* FclSetUserEventStatus)(cl_event, cl_int)
        CL_API_SUFFIX__VERSION_1_1;

typedef cl_int(CL_API_CALL*
                       FclSetEventCallback)(cl_event, cl_int, void(CL_CALLBACK*)(cl_event, cl_int, void*), void*)
        CL_API_SUFFIX__VERSION_1_1;

#endif

/* Profiling APIs */
typedef cl_int(CL_API_CALL*
                       FclGetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void*, size_t*)
        CL_API_SUFFIX__VERSION_1_0;

/* Flush and Finish APIs */
typedef cl_int(CL_API_CALL* FclFlush)(cl_command_queue
) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* FclFinish)(cl_command_queue
) CL_API_SUFFIX__VERSION_1_0;

/* Enqueued Commands APIs */
typedef cl_int(CL_API_CALL*
                       FclEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_1

typedef cl_int(CL_API_CALL*
                       FclEnqueueReadBufferRect)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_1;

#endif

typedef cl_int(CL_API_CALL*
                       FclEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_1

typedef cl_int(CL_API_CALL*
                       FclEnqueueWriteBufferRect)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_1;

#endif

#ifdef CL_VERSION_1_2

typedef cl_int(CL_API_CALL*
                       FclEnqueueFillBuffer)(cl_command_queue, cl_mem, const void*, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_2;

#endif

typedef cl_int(CL_API_CALL*
                       FclEnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_1

typedef cl_int(CL_API_CALL*
                       FclEnqueueCopyBufferRect)(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_1;

#endif

typedef cl_int(CL_API_CALL*
                       FclEnqueueReadImage)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueWriteImage)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_2

typedef cl_int(CL_API_CALL*
                       FclEnqueueFillImage)(cl_command_queue, cl_mem, const void*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_2;

#endif

typedef cl_int(CL_API_CALL*
                       FclEnqueueCopyImage)(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueCopyImageToBuffer)(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, size_t, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueCopyBufferToImage)(cl_command_queue, cl_mem, cl_mem, size_t, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

typedef void*(CL_API_CALL*
                      FclEnqueueMapBuffer)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event*, cl_event*, cl_int*)
        CL_API_SUFFIX__VERSION_1_0;

typedef void*(CL_API_CALL*
                      FclEnqueueMapImage)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t*, const size_t*, size_t*, size_t*, cl_uint, const cl_event*, cl_event*, cl_int*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueUnmapMemObject)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_2

typedef cl_int(CL_API_CALL*
                       FclEnqueueMigrateMemObjects)(cl_command_queue, cl_uint, const cl_mem*, cl_mem_migration_flags, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_2;

#endif

typedef cl_int(CL_API_CALL*
                       FclEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueNativeKernel)(cl_command_queue, void(CL_CALLBACK*)(void*), void*, size_t, cl_uint, const cl_mem*, const void**, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_2

typedef cl_int(CL_API_CALL*
                       FclEnqueueMarkerWithWaitList)(cl_command_queue, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL*
                       FclEnqueueBarrierWithWaitList)(cl_command_queue, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_1_2;

#endif

#ifdef CL_VERSION_2_0

typedef cl_int(CL_API_CALL*
                       FclEnqueueSVMFree)(cl_command_queue, cl_uint, void*[], void(CL_CALLBACK*)(cl_command_queue, cl_uint, void*[], void*), void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueSVMMemcpy)(cl_command_queue, cl_bool, void*, const void*, size_t, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueSVMMemFill)(cl_command_queue, void*, const void*, size_t, size_t, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueSVMMap)(cl_command_queue, cl_bool, cl_map_flags, void*, size_t, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL*
                       FclEnqueueSVMUnmap)(cl_command_queue, void*, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_2_0;

#endif

#ifdef CL_VERSION_2_1
typedef cl_int(CL_API_CALL*
                       FclEnqueueSVMMigrateMem)(cl_command_queue, cl_uint, const void**, const size_t*, cl_mem_migration_flags, cl_uint, const cl_event*, cl_event*)
        CL_API_SUFFIX__VERSION_2_1;

#endif

#ifdef CL_VERSION_1_2

typedef void*(CL_API_CALL*
                      FclGetExtensionFunctionAddressForPlatform)(cl_platform_id, const char*)
        CL_API_SUFFIX__VERSION_1_2;

#endif
}// extern "C"

}// namespace cl

#define DECL_CL_FUN(name) cl::F##name name

class OpenCLRuntimeLibrary : public platform::RuntimeLibrary
{
    mutable std::unordered_map<
            DeviceInfo, std::shared_ptr<DeviceHandle>, boost::hash<DeviceInfo>>
            m_device_cache;
    mutable std::mutex m_lock;

    std::unique_ptr<const BufferInterface> p_buffer_interface;
    std::unique_ptr<const EventInterface> p_event_interface;
    std::unique_ptr<const KernelInterface> p_kernel_interface;
    std::unique_ptr<const QueueInterface> p_queue_interface;

    void load_symbols();

public:
    explicit OpenCLRuntimeLibrary(const fs::path& path);

    RPY_NO_DISCARD const BufferInterface* buffer_interface() const noexcept
    {
        RPY_DBG_ASSERT(p_buffer_interface);
        return p_buffer_interface.get();
    }
    RPY_NO_DISCARD const EventInterface* event_interface() const noexcept
    {
        RPY_DBG_ASSERT(p_event_interface);
        return p_event_interface.get();
    }
    RPY_NO_DISCARD const KernelInterface* kernel_interface() const noexcept
    {
        RPY_DBG_ASSERT(p_kernel_interface);
        return p_kernel_interface.get();
    }
    RPY_NO_DISCARD const QueueInterface* queue_interface() const noexcept
    {
        RPY_DBG_ASSERT(p_queue_interface);
        return p_queue_interface.get();
    }

    string_view get_version() const noexcept;

    std::shared_ptr<DeviceHandle> get_device(DeviceInfo info) const;

    DECL_CL_FUN(clGetPlatformIDs);
    DECL_CL_FUN(clGetPlatformInfo);
    DECL_CL_FUN(clGetDeviceIDs);
    DECL_CL_FUN(clGetDeviceInfo);

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clCreateSubDevices);
    DECL_CL_FUN(clRetainDevice);
    DECL_CL_FUN(clReleaseDevice);
#endif

#ifdef CL_VERSION_2_1
    DECL_CL_FUN(clSetDefaultDeviceCommandQueue);
    DECL_CL_FUN(clGetDeviceAndHostTimer);
    DECL_CL_FUN(clGetHostTimer);
#endif

    DECL_CL_FUN(clCreateContext);
    DECL_CL_FUN(clCreateContextFromType);
    DECL_CL_FUN(clRetainContext);
    DECL_CL_FUN(clReleaseContext);
    DECL_CL_FUN(clGetContextInfo);

#ifdef CL_VERSION_3_0
    DECL_CL_FUN(clSetContextDestructorCallback);
#endif

#ifdef CL_VERSION_2_0
    DECL_CL_FUN(clCreateCommandQueueWithProperties);
#endif

    DECL_CL_FUN(clRetainCommandQueue);
    DECL_CL_FUN(clReleaseCommandQueue);
    DECL_CL_FUN(clGetCommandQueueInfo);

    DECL_CL_FUN(clCreateBuffer);

#ifdef CL_VERSION_1_1
    DECL_CL_FUN(clCreateSubBuffer);
#endif

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clCreateImage);
#endif

#ifdef CL_VERSION_2_0
    DECL_CL_FUN(clCreatePipe);
#endif

#ifdef CL_VERSION_3_0
    DECL_CL_FUN(clCreateBufferWithProperties);
    DECL_CL_FUN(clCreateImageWithProperties);
#endif

    DECL_CL_FUN(clRetainMemObject);
    DECL_CL_FUN(clReleaseMemObject);
    DECL_CL_FUN(clGetSupportedImageFormats);
    DECL_CL_FUN(clGetMemObjectInfo);
    DECL_CL_FUN(clGetImageInfo);

#ifdef CL_VERSION_2_0
    DECL_CL_FUN(clGetPipeInfo);
#endif

#ifdef CL_VERSION_1_1
    DECL_CL_FUN(clSetMemObjectDestructorCallback);
#endif

#ifdef CL_VERSION_2_0
    DECL_CL_FUN(clSVMAlloc);
    DECL_CL_FUN(clSVMFree);
#endif

#ifdef CL_VERSION_2_0
    DECL_CL_FUN(clCreateSamplerWithProperties);
#endif

    DECL_CL_FUN(clRetainSampler);
    DECL_CL_FUN(clReleaseSampler);
    DECL_CL_FUN(clGetSamplerInfo);

    DECL_CL_FUN(clCreateProgramWithSource);
    DECL_CL_FUN(clCreateProgramWithBinary);

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clCreateProgramWithBuiltInKernels);
#endif

#ifdef CL_VERSION_2_1
    DECL_CL_FUN(clCreateProgramWithIL);
#endif

    DECL_CL_FUN(clRetainProgram);
    DECL_CL_FUN(clReleaseProgram);
    DECL_CL_FUN(clBuildProgram);

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clCompileProgram);
    DECL_CL_FUN(clLinkProgram);
#endif

#ifdef CL_VERSION_2_2
    DECL_CL_FUN(clSetProgramReleaseCallback);
    DECL_CL_FUN(clSetProgramSpecializationConstant);
#endif

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clUnloadPlatformCompiler);
#endif

    DECL_CL_FUN(clGetProgramInfo);
    DECL_CL_FUN(clGetProgramBuildInfo);

    DECL_CL_FUN(clCreateKernel);
    DECL_CL_FUN(clCreateKernelsInProgram);

#ifdef CL_VERSION_2_1
    DECL_CL_FUN(clCloneKernel);
#endif

    DECL_CL_FUN(clRetainKernel);
    DECL_CL_FUN(clReleaseKernel);
    DECL_CL_FUN(clSetKernelArg);

#ifdef CL_VERSION_2_0
    DECL_CL_FUN(clSetKernelArgSVMPointer);
    DECL_CL_FUN(clSetKernelExecInfo);
#endif

    DECL_CL_FUN(clGetKernelInfo);

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clGetKernelArgInfo);
#endif

    DECL_CL_FUN(clGetKernelWorkGroupInfo);

#ifdef CL_VERSION_2_1
    DECL_CL_FUN(clGetKernelSubGroupInfo);
#endif

    DECL_CL_FUN(clWaitForEvents);
    DECL_CL_FUN(clGetEventInfo);

#ifdef CL_VERSION_1_1
    DECL_CL_FUN(clCreateUserEvent);
#endif

    DECL_CL_FUN(clRetainEvent);
    DECL_CL_FUN(clReleaseEvent);

#ifdef CL_VERSION_1_1
    DECL_CL_FUN(clSetUserEventStatus);
    DECL_CL_FUN(clSetEventCallback);
#endif

    DECL_CL_FUN(clGetEventProfilingInfo);
    DECL_CL_FUN(clFlush);
    DECL_CL_FUN(clFinish);

    DECL_CL_FUN(clEnqueueReadBuffer);

#ifdef CL_VERSION_1_1
    DECL_CL_FUN(clEnqueueReadBufferRect);
#endif

    DECL_CL_FUN(clEnqueueWriteBuffer);

#ifdef CL_VERSION_1_1
    DECL_CL_FUN(clEnqueueWriteBufferRect);
#endif

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clEnqueueFillBuffer);
#endif

    DECL_CL_FUN(clEnqueueCopyBuffer);

#ifdef CL_VERSION_1_1
    DECL_CL_FUN(clEnqueueCopyBufferRect);
#endif

    DECL_CL_FUN(clEnqueueReadImage);
    DECL_CL_FUN(clEnqueueWriteImage);

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clEnqueueFillImage);
#endif

    DECL_CL_FUN(clEnqueueCopyImage);
    DECL_CL_FUN(clEnqueueCopyImageToBuffer);
    DECL_CL_FUN(clEnqueueCopyBufferToImage);

    DECL_CL_FUN(clEnqueueMapBuffer);
    DECL_CL_FUN(clEnqueueMapImage);
    DECL_CL_FUN(clEnqueueUnmapMemObject);

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clEnqueueMigrateMemObjects);
#endif

    DECL_CL_FUN(clEnqueueNDRangeKernel);
    DECL_CL_FUN(clEnqueueNativeKernel);

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clEnqueueMarkerWithWaitList);
    DECL_CL_FUN(clEnqueueBarrierWithWaitList);
#endif

#ifdef CL_VERSION_2_0
    DECL_CL_FUN(clEnqueueSVMFree);
    DECL_CL_FUN(clEnqueueSVMMemcpy);
    DECL_CL_FUN(clEnqueueSVMMemFill);
    DECL_CL_FUN(clEnqueueSVMMap);
    DECL_CL_FUN(clEnqueueSVMUnmap);
#endif

#ifdef CL_VERSION_2_1
    DECL_CL_FUN(clEnqueueSVMMigrateMem);
#endif

#ifdef CL_VERSION_1_2
    DECL_CL_FUN(clGetExtensionFunctionAddressForPlatform);
#endif
};

#undef DECL_CL_FUN

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_OPENCL_OPEN_CL_RUNTIME_LIBRARY_H_
