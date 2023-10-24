//
// Created by sam on 23/10/23.
//

#include "cuda_buffer.h"

#include "cuda_device.h"

using namespace rpy;
using namespace rpy::devices;

BufferMode CUDABuffer::mode() const { return BufferInterface::mode(); }
dimn_t CUDABuffer::size() const { return BufferInterface::size(); }
void* CUDABuffer::ptr() { return BufferInterface::ptr(); }
CUDABuffer::~CUDABuffer() {}
DeviceType CUDABuffer::type() const noexcept { return InterfaceBase::type(); }
dimn_t CUDABuffer::ref_count() const noexcept
{
    return InterfaceBase::ref_count();
}
std::unique_ptr<devices::dtl::InterfaceBase> CUDABuffer::clone() const
{
    return InterfaceBase::clone();
}
Device CUDABuffer::device() const noexcept { return m_device; }