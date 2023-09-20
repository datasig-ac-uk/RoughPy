//
// Created by sam on 20/09/23.
//

#include "cpu_device.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/alloc.h>

#include "opencl/open_cl_device.h"

#include "cpu_buffer.h"

using namespace rpy;
using namespace rpy::device;

boost::intrusive_ptr<const CPUDevice> CPUDevice::get() noexcept
{
    static const CPUDevice device;
    return &device;
}

const CPUDevice::AllocationMetadata& CPUDevice::get_metadata(void* ptr) const
{
    std::lock_guard<std::recursive_mutex> access(m_lock);
    auto found = m_alloc_md.find(ptr);
    RPY_CHECK(found != m_alloc_md.end());
    return found->second;
}

CPUDevice::~CPUDevice() {}
DeviceInfo CPUDevice::info() const noexcept { return {DeviceType::CPU, 0}; }
optional<fs::path> CPUDevice::runtime_library() const noexcept
{
    return {};
}
Buffer CPUDevice::raw_alloc(dimn_t count, dimn_t alignment) const
{
    std::lock_guard<std::recursive_mutex> access(m_lock);
    auto entry = m_alloc_md.insert(
            {aligned_alloc(alignment, count),
            { alignment, count }}
            );

    RPY_CHECK(entry.second);
    return Buffer{ cpu::buffer_interface(), entry.first->first };
}
void CPUDevice::raw_free(Buffer buffer) const
{
    RPY_CHECK(buffer.interface() == cpu::buffer_interface());
    std::lock_guard<std::recursive_mutex> access(m_lock);
    auto* ptr = buffer.content();
    auto found = m_alloc_md.find(ptr);
    RPY_CHECK(found != m_alloc_md.end());

    m_alloc_md.erase(found);
    aligned_free(ptr);
}

