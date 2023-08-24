//
// Created by user on 23/08/23.
//

#include "open_cl_device.h"

#include <roughpy/core/macros.h>

using namespace rpy;
using namespace rpy::platform;








OpenCLDevice::OpenCLDevice(DeviceType type, int32_t device_id)
    : DeviceHandle(type, device_id)
{

}
void* OpenCLDevice::raw_allocate(dimn_t size, dimn_t alignment) const
{
    return DeviceHandle::raw_allocate(size, alignment);
}
void OpenCLDevice::raw_dealloc(void* raw_pointer, dimn_t size) const
{
    DeviceHandle::raw_dealloc(raw_pointer, size);
}
void OpenCLDevice::copy_to_device(
        void* d_dst_raw, const void* h_src_raw, dimn_t count
) const
{
    DeviceHandle::copy_to_device(d_dst_raw, h_src_raw, count);
}
void OpenCLDevice::copy_from_device(
        void* h_dst_raw, const void* d_src_raw, dimn_t count
) const
{
    DeviceHandle::copy_from_device(h_dst_raw, d_src_raw, count);
}
