//
// Created by sam on 3/30/24.
//

#include "devices/type.h"


#include <roughpy/platform/alloc.h>

#include "devices/device_handle.h"

using namespace rpy;
using namespace rpy::devices;


Type::Type(string id, string name, TypeInfo info)
    : m_id(std::move(id)), m_name(std::move(name)), m_info(info)
{}

Type::~Type() = default;

Buffer Type::allocate(Device device, dimn_t count) const
{
    return device->alloc(m_info, count);
}

void* Type::allocate_single() const
{
    return aligned_alloc(m_info.alignment, m_info.bytes);
}

void Type::free_single(void* ptr) const { aligned_free(ptr); }

bool Type::supports_device(const Device& device) const noexcept
{
    return true;
}
