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

#include "ocl_device.h"

#include "ocl_buffer.h"
#include "ocl_event.h"
#include "ocl_handle_errors.h"
#include "ocl_helpers.h"
#include "ocl_kernel.h"
#include "ocl_queue.h"

#include <roughpy/platform/configuration.h>

#include <roughpy/device/buffer.h>
#include <roughpy/device/device_object_base.h>
#include <roughpy/device/event.h>
#include <roughpy/device/kernel.h>
#include <roughpy/device/queue.h>

#include <CL/cl_ext.h>
#include <boost/container/small_vector.hpp>

#include <fstream>

using namespace rpy;
using namespace rpy::devices;

Buffer OCLDeviceHandle::make_buffer(cl_mem buffer, bool move) const
{
    if (RPY_LIKELY(!move)) {
        auto ecode = clRetainMemObject(buffer);
        if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
    }
    return Buffer(std::make_unique<OCLBuffer>(buffer, this));
}
Event OCLDeviceHandle::make_event(cl_event event, bool move) const
{
    if (RPY_UNLIKELY(!move)) {
        auto ecode = clRetainEvent(event);
        if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
    }
    return Event(std::make_unique<OCLEvent>(event, this));
}
Kernel OCLDeviceHandle::make_kernel(cl_kernel kernel, bool move) const
{
    if (RPY_UNLIKELY(!move)) {
        auto ecode = clRetainKernel(kernel);
        if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
    }
    return Kernel(std::make_unique<OCLKernel>(kernel, this));
}
Queue OCLDeviceHandle::make_queue(cl_command_queue queue, bool move) const
{
    if (RPY_LIKELY(!move)) {
        auto ecode = clRetainCommandQueue(queue);
        if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
    }
    return Queue(std::make_unique<OCLQueue>(queue, this));
}

OCLDeviceHandle::OCLDeviceHandle(cl_device_id id) : m_device(id)
{
    const guard_type access(get_lock());
    cl_int ecode;

    m_ctx = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &ecode);

    if (RPY_UNLIKELY(m_ctx == nullptr)) { RPY_HANDLE_OCL_ERROR(ecode); }

    m_default_queue = clCreateCommandQueueWithProperties(
            m_ctx,
            m_device,
            nullptr,
            &ecode
    );

    if (RPY_UNLIKELY(m_default_queue == nullptr)) {
        RPY_HANDLE_OCL_ERROR(ecode);
    }

    const auto& config = get_config();
    auto kernel_dir = config.get_builtin_kernel_dir();
    if (exists(kernel_dir)) {
        fs::directory_iterator iter(config.get_builtin_kernel_dir());

        std::vector<string> sources;
        for (auto&& dir : iter) {
            std::ifstream istr(dir.path());
            sources.emplace_back();
        }
        auto count = static_cast<cl_uint>(sources.size());

        std::vector<const char*> strings;
        std::vector<size_t> sizes;
        strings.reserve(count);
        sizes.reserve(count);
        for (auto&& src : sources) {
            strings.push_back(src.data());
            sizes.push_back(src.size());
        }

        auto program = clCreateProgramWithSource(
                m_ctx,
                count,
                strings.data(),
                sizes.data(),
                &ecode
        );

        if (RPY_UNLIKELY(program == nullptr)) { RPY_HANDLE_OCL_ERROR(ecode); }

        m_programs.push_back(program);

        cl_uint num_kernels;
        ecode = clGetProgramInfo(
                program,
                CL_PROGRAM_NUM_KERNELS,
                sizeof(cl_uint),
                &num_kernels,
                nullptr
        );

        if (RPY_UNLIKELY(ecode != CL_SUCCESS)) { RPY_HANDLE_OCL_ERROR(ecode); }

        std::vector<cl_kernel> kernels(num_kernels);
        ecode = clCreateKernelsInProgram(
                program,
                num_kernels,
                kernels.data(),
                nullptr
        );

        if (RPY_UNLIKELY(ecode != CL_SUCCESS)) { RPY_HANDLE_OCL_ERROR(ecode); }

        for (auto&& kernel : kernels) {
            DeviceHandle::register_kernel(make_kernel(kernel, true));
        }
    }
}

OCLDeviceHandle::~OCLDeviceHandle()
{
    guard_type access(get_lock());
    cl_int ecode;
    ecode = clReleaseCommandQueue(m_default_queue);
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);

    m_device_id = 0;
    m_default_queue = nullptr;

    while (!m_programs.empty()) {
        ecode = clReleaseProgram(m_programs.back());
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
        m_programs.pop_back();
    }

    clReleaseContext(m_ctx);
    clReleaseDevice(m_device);
}
DeviceInfo OCLDeviceHandle::info() const noexcept
{
    return {DeviceType::OpenCL, m_device_id};
}
optional<fs::path> OCLDeviceHandle::runtime_library() const noexcept
{
    return {};
}
Buffer OCLDeviceHandle::raw_alloc(dimn_t count, dimn_t alignment) const
{
    cl_int ecode;
    auto new_mem
            = clCreateBuffer(m_ctx, CL_MEM_READ_WRITE, count, nullptr, &ecode);

    if (RPY_UNLIKELY(new_mem == nullptr)) { RPY_HANDLE_OCL_ERROR(ecode); }
    return make_buffer(new_mem);
}
void OCLDeviceHandle::raw_free(void* pointer, dimn_t size) const {}
optional<Kernel> OCLDeviceHandle::get_kernel(const string& name) const noexcept
{
    const guard_type access(get_lock());
    auto found = DeviceHandle::get_kernel(name);
    if (found) { return found; }

    const auto& config = get_config();

    fs::path fname(name);
    fname.replace_extension(".cl");
    for (auto&& dir : config.kernel_source_search_dirs()) {
        auto path = dir / fname;

        if (exists(path)) {
            std::ifstream ifs(path);
            if (ifs.is_open()) {
                try {
                    string source;
                    ifs >> source;
                    return {};
                } catch (...) {
                    return {};
                }
            }
        }
    }

    return {};
}

bool OCLDeviceHandle::cl_supports_version(cl_version version) const
{
    cl_platform_id plat = nullptr;
    auto ecode = clGetDeviceInfo(
            m_device,
            CL_DEVICE_PLATFORM,
            sizeof(plat),
            &plat,
            nullptr
    );
    if (ecode != CL_SUCCESS) { return false; }

    RPY_DBG_ASSERT(plat != nullptr);

    auto str_vers
            = cl::string_info(clGetPlatformInfo, plat, CL_PLATFORM_VERSION);

    /*
     * str_vers should be of the form:
     *
     *  OpenCL<space><major_version.minor_version><space><other-stuff>
     *
     *  so the offset to the beginning of the version string is 7. The current
     *  options are 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, or 3.0, so the version string
     *  is of length 3.
     */
    constexpr size_t offset = sizeof("OpenCL"); // Length + 1 for null byte

    string_view actual_version(str_vers.data() + offset, 3);

    cl_version num_version = 0;
    if (actual_version == "3.0") {
        num_version = 300;
    } else if (actual_version == "2.2") {
        num_version = 220;
    } else if (actual_version == "2.1") {
        num_version = 210;
    } else if (actual_version == "2.0") {
        num_version = 200;
    } else if (actual_version == "1.2") {
        num_version = 120;
    } else if (actual_version == "1.1") {
        num_version = 110;
    } else if (actual_version == "1.0") {
        num_version = 100;
    } else {
        RPY_THROW(std::runtime_error, "Invalid version string");
    };

    return num_version >= version;
}

cl_program OCLDeviceHandle::get_header_program(
        const string& name,
        const string& source
) const
{
    guard_type access(get_lock());

    auto program = m_header_cache[name];

    if (program == nullptr) {
        cl_int ecode = CL_SUCCESS;
        const char* c_source = source.c_str();
        size_t source_size = source.size();
        program = clCreateProgramWithSource(
                m_ctx,
                1,
                &c_source,
                &source_size,
                &ecode
        );

        if (program == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }
    }

    return program;
}

cl_program
OCLDeviceHandle::compile_program(const ExtensionSourceAndOptions& args) const
{
    if (args.sources.empty()) {
        RPY_THROW(std::runtime_error, "Sources cannot be empty");
    }
    guard_type access(get_lock());

    std::vector<const char*> source_ptrs;
    std::vector<size_t> source_sizes;
    source_ptrs.reserve(args.sources.size());
    source_sizes.reserve(args.sources.size());
    for (auto&& src : args.sources) {
        source_ptrs.push_back(src.c_str());
        source_sizes.push_back(src.size());
    }

    cl_int ecode = CL_SUCCESS;
    auto program = clCreateProgramWithSource(
            m_ctx,
            source_ptrs.size(),
            source_ptrs.data(),
            source_sizes.data(),
            &ecode
    );

    if (program == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

    std::vector<cl_program> header_programs;
    header_programs.reserve(args.header_name_and_source.size());

    source_ptrs.clear();
    source_ptrs.reserve(args.header_name_and_source.size());
    for (auto&& header : args.header_name_and_source) {
        header_programs.push_back(
                get_header_program(header.first, header.second)
        );
        source_ptrs.push_back(header.first.c_str());
    }

    if (cl_supports_version(120)) {
        ecode = clCompileProgram(
                program,
                1,
                &m_device,
                args.compile_options.c_str(),
                header_programs.size(),
                header_programs.data(),
                source_ptrs.data(),
                nullptr,
                nullptr
        );

        if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

        auto final_program = clLinkProgram(
                m_ctx,
                1,
                &m_device,
                args.link_options.c_str(),
                1,
                &program,
                nullptr,
                nullptr,
                &ecode
        );

        if (final_program == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

        m_programs.push_back(final_program);
        program = final_program;

    } else {

        ecode = clBuildProgram(
                program,
                1,
                &m_device,
                args.compile_options.c_str(),
                nullptr,
                nullptr
        );
        if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

        m_programs.push_back(program);
    }

    return program;
}

optional<Kernel>
OCLDeviceHandle::compile_kernel_from_str(const ExtensionSourceAndOptions& args
) const
{
    auto program = compile_program(args);

    cl_kernel kernel;
    auto ecode = clCreateKernelsInProgram(program, 1, &kernel, nullptr);
    if (RPY_UNLIKELY(ecode != CL_SUCCESS)) { RPY_HANDLE_OCL_ERROR(ecode); }
    return DeviceHandle::register_kernel(make_kernel(kernel, true));
}
void OCLDeviceHandle::compile_kernels_from_src(
        const ExtensionSourceAndOptions& args
) const
{
    auto program = compile_program(args);

    cl_uint num_kernels;
    auto ecode = clGetProgramInfo(
            program,
            CL_PROGRAM_NUM_KERNELS,
            sizeof(cl_uint),
            &num_kernels,
            nullptr
    );
    if (RPY_UNLIKELY(ecode != CL_SUCCESS)) { RPY_HANDLE_OCL_ERROR(ecode); }

    boost::container::small_vector<cl_kernel, 1> kernels(num_kernels);

    ecode = clCreateKernelsInProgram(
            program,
            num_kernels,
            kernels.data(),
            nullptr
    );
    if (RPY_UNLIKELY(ecode != CL_SUCCESS)) { RPY_HANDLE_OCL_ERROR(ecode); }

    for (auto&& kernel : kernels) {
        DeviceHandle::register_kernel(make_kernel(kernel, true));
    }
}
Event OCLDeviceHandle::new_event() const
{
    cl_int ecode;
    auto event = clCreateUserEvent(m_ctx, &ecode);
    if (event != nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

    return make_event(event);
}

Queue OCLDeviceHandle::new_queue() const
{
    cl_int ecode;
    auto queue = clCreateCommandQueueWithProperties(
            m_ctx,
            m_device,
            nullptr,
            &ecode
    );
    if (queue == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }
    return make_queue(queue);
}
Queue OCLDeviceHandle::get_default_queue() const
{
    return make_queue(m_default_queue, false);
}

optional<boost::uuids::uuid> OCLDeviceHandle::uuid() const noexcept
{
    static_assert(
            CL_UUID_SIZE_KHR == sizeof(boost::uuids::uuid::data),
            "UUID should be 16 bytes"
    );
    boost::uuids::uuid uuid;
    auto ecode = clGetDeviceInfo(
            m_device,
            CL_DEVICE_UUID_KHR,
            sizeof(uuid.data),
            &uuid.data,
            nullptr
    );
    if (ecode != CL_SUCCESS) { return {}; }
    return uuid;
}

optional<PCIBusInfo> OCLDeviceHandle::pci_bus_info() const noexcept
{
    static_assert(
            sizeof(cl_uint) == sizeof(uint32_t),
            "bus info items should be 32 bit integers"
    );
    cl_device_pci_bus_info_khr bus_info;
    auto ecode = clGetDeviceInfo(
            m_device,
            CL_DEVICE_PCI_BUS_INFO_KHR,
            sizeof(cl_device_pci_bus_info_khr),
            &bus_info,
            nullptr
    );

    if (ecode != CL_SUCCESS) { return {}; }

    return {
            {bus_info.pci_bus,
             bus_info.pci_device,
             bus_info.pci_domain,
             bus_info.pci_function}
    };
}
DeviceCategory OCLDeviceHandle::category() const noexcept
{
    cl_device_type dtype;
    auto ecode = clGetDeviceInfo(
            m_device,
            CL_DEVICE_TYPE,
            sizeof(dtype),
            &dtype,
            nullptr
    );
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    if (ecode == CL_SUCCESS) {
        switch (dtype) {
            case CL_DEVICE_TYPE_CPU: return DeviceCategory::CPU;
            case CL_DEVICE_TYPE_GPU: return DeviceCategory::GPU;
            case CL_DEVICE_TYPE_ACCELERATOR: return DeviceCategory::AIP;
            case CL_DEVICE_TYPE_CUSTOM: return DeviceCategory::Other;
            default: break;
        }
    }

    return DeviceHandle::category();
}
bool OCLDeviceHandle::has_compiler() const noexcept
{
    cl_bool compiler_available = 0;
    auto ecode = clGetDeviceInfo(
            m_device,
            CL_DEVICE_COMPILER_AVAILABLE,
            sizeof(compiler_available),
            &compiler_available,
            nullptr
    );
    RPY_CHECK(ecode == CL_SUCCESS);

    return compiler_available != 0;
}
DeviceType OCLDeviceHandle::type() const noexcept
{
    return DeviceType::OpenCL;
}
