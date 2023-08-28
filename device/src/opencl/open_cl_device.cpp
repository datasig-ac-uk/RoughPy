//
// Created by sam on 25/08/23.
//

#include "open_cl_device.h"
std::optional<std::filesystem::path>
rpy::device::OpenCLDevice::runtime_library() const noexcept
{
    return p_runtime->location();
}
void* rpy::device::OpenCLDevice::raw_allocate(
        rpy::dimn_t size, rpy::dimn_t alignment
) const
{
    return nullptr;
}
void rpy::device::OpenCLDevice::raw_dealloc(
        void* d_raw_pointer, rpy::dimn_t size
) const
{}
void rpy::device::OpenCLDevice::copy_to_device(
        void* d_dst_raw, const void* h_src_raw, rpy::dimn_t count
) const
{}
void rpy::device::OpenCLDevice::copy_from_device(
        void* h_dst_raw, const void* d_src_raw, rpy::dimn_t count
) const
{}
rpy::device::Kernel* rpy::device::OpenCLDevice::get_kernel(std::string_view name
) const
{
    return nullptr;
}
