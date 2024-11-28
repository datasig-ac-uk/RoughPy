//
// Created by sam on 21/11/24.
//

#include "host_device.h"

#include <memory>

#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/check.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/alloc.h"

#include "roughpy/generics/type.h"

#include "roughpy/device/host_address_memory.h"

using namespace rpy;
using namespace rpy::device;

void HostDeviceHandle::inc_ref() const noexcept
{
    // Do nothing
}
bool HostDeviceHandle::dec_ref() const noexcept
{
    // Do nothing and never destroy this object.
    return false;
}
intptr_t HostDeviceHandle::ref_count() const noexcept
{
    return 1;
}
bool HostDeviceHandle::is_host() const noexcept
{
    return true;
}
bool HostDeviceHandle::supports_type(const generics::Type& type) const noexcept
{
    return true;
}
Rc<Memory> HostDeviceHandle::allocate_memory(
        const generics::Type& type,
        size_t size,
        size_t alignment
) const
{
    if (alignment <= alignof(std::max_align_t)) {
        alignment = alignof(std::max_align_t);
    }
    const auto bytes = size * generics::size_of(type);

    mem::ScopedSafePtr<> safe_holder(alignment, bytes);

    auto result = std::make_unique<HostAddressMemory>(
        type,
        *this,
        safe_holder.data(),
        size,
        bytes,
        MemoryMode::ReadWrite
        );

    // The construction is done, we can hand over management to result.
    safe_holder.reset();

    // Initialize with "zero"
    type.copy_or_fill(result->data(), nullptr, size, true);

    // Make this a Rc instead of a unique ptr.
    return Rc<Memory>(result.release());
}
void HostDeviceHandle::copy_memory(Memory& dst, const Memory& src) const
{
    RPY_CHECK_EQ(dst.type(), src.type());

    const auto* src_device = &src.device();
    const auto* dst_device = &dst.device();

    if (RPY_UNLIKELY(dst_device != this)) {
        dst_device->copy_memory(dst, src);
    }

    if (RPY_UNLIKELY(src_device != this)) {
        src_device->copy_memory(dst, src);
    }

    auto* dst_ptr = dst.data();
    const auto* src_ptr = src.data();
    const auto size = src.size();

    RPY_CHECK_GE(dst.size(), size);

    dst.type().copy_or_fill(dst_ptr, src_ptr, size, false);
}
void HostDeviceHandle::destroy_memory(Memory& memory) const
{
    RPY_CHECK_EQ(&memory.device(), this);

    auto* real_memory = dynamic_cast<HostAddressMemory*>(&memory);
    RPY_CHECK_NE(real_memory, nullptr);

    const auto size = real_memory->size();

    real_memory->type().destroy_range(real_memory->data(), size);
    mem::aligned_free(real_memory->release(), size);
}