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

using namespace rpy;
using namespace rpy::device;


#define SET_CL_FUN(name) name = get<cl::F ## name>(#name)


void OpenCLRuntimeLibrary::load_symbols() {

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

OpenCLRuntimeLibrary::OpenCLRuntimeLibrary(const fs::path& path)
    : platform::RuntimeLibrary(path)
{
    load_symbols();
}
const char* OpenCLRuntimeLibrary::get_version() const noexcept
{
    return nullptr;
}
