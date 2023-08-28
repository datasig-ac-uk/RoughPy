//
// Created by sam on 25/08/23.
//

#ifndef ROUGHPY_OPEN_CL_DEVICE_H
#define ROUGHPY_OPEN_CL_DEVICE_H

#include <roughpy/device/core.h>
#include <roughpy/device/device_handle.h>
#include <roughpy/device/kernel.h>

#include "open_cl_runtime_library.h"


namespace rpy {
namespace device {

class OpenCLDevice : public DeviceHandle
{
    OpenCLRuntimeLibrary* p_runtime;


public:
    optional<fs::path> runtime_library() const noexcept override;
    void* raw_allocate(dimn_t size, dimn_t alignment) const override;
    void raw_dealloc(void* d_raw_pointer, dimn_t size) const override;
    void copy_to_device(void* d_dst_raw, const void* h_src_raw, dimn_t count)
            const override;
    void copy_from_device(void* h_dst_raw, const void* d_src_raw, dimn_t count)
            const override;
    Kernel* get_kernel(string_view name) const override;
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_OPEN_CL_DEVICE_H
