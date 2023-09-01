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

#include "open_cl_runtime_library.h"


#include "open_cl_device.h"


#include <vector>

using namespace rpy;
using namespace rpy::device;

#define SET_CL_FUN(name)                                                       \
    do {                                                                       \
        if (RPY_LIKELY(has(#name))) {                                          \
            name = get<cl::F##name>(#name);                                    \
        } else {                                                               \
            name = nullptr;                                                    \
        }                                                                      \
    } while (0)

void OpenCLRuntimeLibrary::load_symbols()
{

    SET_CL_FUN(clGetPlatformIDs);
    SET_CL_FUN(clGetPlatformInfo);
    SET_CL_FUN(clGetDeviceIDs);
    SET_CL_FUN(clGetDeviceInfo);

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clCreateSubDevices);
    SET_CL_FUN(clRetainDevice);
    SET_CL_FUN(clReleaseDevice);
#endif

#ifdef CL_VERSION_2_1
    SET_CL_FUN(clSetDefaultDeviceCommandQueue);
    SET_CL_FUN(clGetDeviceAndHostTimer);
    SET_CL_FUN(clGetHostTimer);
#endif

    SET_CL_FUN(clCreateContext);
    SET_CL_FUN(clCreateContextFromType);
    SET_CL_FUN(clRetainContext);
    SET_CL_FUN(clReleaseContext);
    SET_CL_FUN(clGetContextInfo);

#ifdef CL_VERSION_3_0
    SET_CL_FUN(clSetContextDestructorCallback);
#endif

#ifdef CL_VERSION_2_0
    SET_CL_FUN(clCreateCommandQueueWithProperties);
#endif

    SET_CL_FUN(clRetainCommandQueue);
    SET_CL_FUN(clReleaseCommandQueue);
    SET_CL_FUN(clGetCommandQueueInfo);

    SET_CL_FUN(clCreateBuffer);

#ifdef CL_VERSION_1_1
    SET_CL_FUN(clCreateSubBuffer);
#endif

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clCreateImage);
#endif

#ifdef CL_VERSION_2_0
    SET_CL_FUN(clCreatePipe);
#endif

#ifdef CL_VERSION_3_0
    SET_CL_FUN(clCreateBufferWithProperties);
    SET_CL_FUN(clCreateImageWithProperties);
#endif

    SET_CL_FUN(clRetainMemObject);
    SET_CL_FUN(clReleaseMemObject);
    SET_CL_FUN(clGetSupportedImageFormats);
    SET_CL_FUN(clGetMemObjectInfo);
    SET_CL_FUN(clGetImageInfo);

#ifdef CL_VERSION_2_0
    SET_CL_FUN(clGetPipeInfo);
#endif

#ifdef CL_VERSION_1_1
    SET_CL_FUN(clSetMemObjectDestructorCallback);
#endif

#ifdef CL_VERSION_2_0
    SET_CL_FUN(clSVMAlloc);
    SET_CL_FUN(clSVMFree);
#endif

#ifdef CL_VERSION_2_0
    SET_CL_FUN(clCreateSamplerWithProperties);
#endif

    SET_CL_FUN(clRetainSampler);
    SET_CL_FUN(clReleaseSampler);
    SET_CL_FUN(clGetSamplerInfo);

    SET_CL_FUN(clCreateProgramWithSource);
    SET_CL_FUN(clCreateProgramWithBinary);

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clCreateProgramWithBuiltInKernels);
#endif

#ifdef CL_VERSION_2_1
    SET_CL_FUN(clCreateProgramWithIL);
#endif

    SET_CL_FUN(clRetainProgram);
    SET_CL_FUN(clReleaseProgram);
    SET_CL_FUN(clBuildProgram);

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clCompileProgram);
    SET_CL_FUN(clLinkProgram);
#endif

#ifdef CL_VERSION_2_2
    SET_CL_FUN(clSetProgramReleaseCallback);
    SET_CL_FUN(clSetProgramSpecializationConstant);
#endif

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clUnloadPlatformCompiler);
#endif

    SET_CL_FUN(clGetProgramInfo);
    SET_CL_FUN(clGetProgramBuildInfo);

    SET_CL_FUN(clCreateKernel);
    SET_CL_FUN(clCreateKernelsInProgram);

#ifdef CL_VERSION_2_1
    SET_CL_FUN(clCloneKernel);
#endif

    SET_CL_FUN(clRetainKernel);
    SET_CL_FUN(clReleaseKernel);
    SET_CL_FUN(clSetKernelArg);

#ifdef CL_VERSION_2_0
    SET_CL_FUN(clSetKernelArgSVMPointer);
    SET_CL_FUN(clSetKernelExecInfo);
#endif

    SET_CL_FUN(clGetKernelInfo);

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clGetKernelArgInfo);
#endif

    SET_CL_FUN(clGetKernelWorkGroupInfo);

#ifdef CL_VERSION_2_1
    SET_CL_FUN(clGetKernelSubGroupInfo);
#endif

    SET_CL_FUN(clWaitForEvents);
    SET_CL_FUN(clGetEventInfo);

#ifdef CL_VERSION_1_1
    SET_CL_FUN(clCreateUserEvent);
#endif

    SET_CL_FUN(clRetainEvent);
    SET_CL_FUN(clReleaseEvent);

#ifdef CL_VERSION_1_1
    SET_CL_FUN(clSetUserEventStatus);
    SET_CL_FUN(clSetEventCallback);
#endif

    SET_CL_FUN(clGetEventProfilingInfo);
    SET_CL_FUN(clFlush);
    SET_CL_FUN(clFinish);

    SET_CL_FUN(clEnqueueReadBuffer);

#ifdef CL_VERSION_1_1
    SET_CL_FUN(clEnqueueReadBufferRect);
#endif

    SET_CL_FUN(clEnqueueWriteBuffer);

#ifdef CL_VERSION_1_1
    SET_CL_FUN(clEnqueueWriteBufferRect);
#endif

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clEnqueueFillBuffer);
#endif

    SET_CL_FUN(clEnqueueCopyBuffer);

#ifdef CL_VERSION_1_1
    SET_CL_FUN(clEnqueueCopyBufferRect);
#endif

    SET_CL_FUN(clEnqueueReadImage);
    SET_CL_FUN(clEnqueueWriteImage);

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clEnqueueFillImage);
#endif

    SET_CL_FUN(clEnqueueCopyImage);
    SET_CL_FUN(clEnqueueCopyImageToBuffer);
    SET_CL_FUN(clEnqueueCopyBufferToImage);

    SET_CL_FUN(clEnqueueMapBuffer);
    SET_CL_FUN(clEnqueueMapImage);
    SET_CL_FUN(clEnqueueUnmapMemObject);

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clEnqueueMigrateMemObjects);
#endif

    SET_CL_FUN(clEnqueueNDRangeKernel);
    SET_CL_FUN(clEnqueueNativeKernel);

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clEnqueueMarkerWithWaitList);
    SET_CL_FUN(clEnqueueBarrierWithWaitList);
#endif

#ifdef CL_VERSION_2_0
    SET_CL_FUN(clEnqueueSVMFree);
    SET_CL_FUN(clEnqueueSVMMemcpy);
    SET_CL_FUN(clEnqueueSVMMemFill);
    SET_CL_FUN(clEnqueueSVMMap);
    SET_CL_FUN(clEnqueueSVMUnmap);
#endif

#ifdef CL_VERSION_2_1
    SET_CL_FUN(clEnqueueSVMMigrateMem);
#endif

#ifdef CL_VERSION_1_2
    SET_CL_FUN(clGetExtensionFunctionAddressForPlatform);
#endif
}

#undef SET_CL_FUN

static constexpr cl_uint s_max_num_platforms = 16;
static constexpr cl_uint s_max_num_devices = 16;

OpenCLRuntimeLibrary::OpenCLRuntimeLibrary(const fs::path& path)
    : platform::RuntimeLibrary(path)
{
    RPY_CHECK(is_loaded());
    load_symbols();
}
string_view OpenCLRuntimeLibrary::get_version() const noexcept { return {}; }

std::shared_ptr<DeviceHandle> OpenCLRuntimeLibrary::get_device(DeviceInfo info
) const
{
    std::lock_guard<std::mutex> access(m_lock);
    auto& entry = m_device_cache[info];
    if (!entry) {

        cl_device_type search_type;
        switch (info.device_type) {
            case CPU:
                search_type = CL_DEVICE_TYPE_CPU;
                break;
            case CUDA: RPY_FALLTHROUGH;
            case CUDAHost: RPY_FALLTHROUGH;
            case CUDAManaged: RPY_FALLTHROUGH;
            case Metal: RPY_FALLTHROUGH;
            case ROCM: RPY_FALLTHROUGH;
            case ROCMHost: RPY_FALLTHROUGH;
            case Vulkan: RPY_FALLTHROUGH;
            case WebGPU:
                search_type = CL_DEVICE_TYPE_GPU;
                break;
            case ExtDev:
                RPY_THROW(std::runtime_error, "ExtDev is not supported");
            case VPI: RPY_FALLTHROUGH;
            case Hexagon:
                search_type = CL_DEVICE_TYPE_ACCELERATOR;
                break;
            case OpenCL: RPY_FALLTHROUGH;
            case OneAPI:
                search_type = CL_DEVICE_TYPE_DEFAULT;
        }


        cl_int rc;
        cl_uint num_platforms = 0;
        std::vector<cl_platform_id> plats(s_max_num_platforms);
        rc = clGetPlatformIDs(s_max_num_platforms, plats.data(), &num_platforms);

        RPY_CHECK(rc == CL_SUCCESS);

        std::vector<cl_device_id> devices(s_max_num_devices);
        std::vector<cl_device_id> candidates;
        candidates.reserve(s_max_num_devices);
        for (cl_uint i=0; i<num_platforms; ++i) {
            auto id = plats[i];
            cl_uint num_devices;
            rc = clGetDeviceIDs(id, search_type, s_max_num_devices,
                                devices.data(), &num_devices);
            if (rc != CL_SUCCESS || num_devices == 0) {
                continue;
            }

            for (cl_uint dev_idx=0; dev_idx < num_devices; ++dev_idx) {
                const auto& dev = devices[dev_idx];
                cl_bool is_available;
                rc = clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE,
                                     sizeof(is_available),
                                     &is_available, nullptr);
                if (rc != CL_SUCCESS || !is_available) { continue; }

                // Other checks?
                candidates.push_back(dev);
            }
        }

        if (candidates.empty()) {
            RPY_THROW(std::runtime_error, "could not find appropriate device");
        }

        entry = std::make_shared<OpenCLDevice>(this, candidates[0]);
    }
    return entry;
}
