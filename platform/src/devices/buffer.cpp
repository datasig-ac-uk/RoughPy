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

#include "devices/buffer.h"

#include <roughpy/core/check.h>
#include <roughpy/core/debug_assertion.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/core/smart_ptr.h>

#include "devices/memory_view.h"
#include "devices/device_handle.h"
#include "devices/device_object_base.h"
#include "devices/event.h"

using namespace rpy;
using namespace rpy::devices;

namespace rpy { namespace devices { namespace dtl {

template class RPY_DLL_EXPORT ObjectBase<BufferInterface, Buffer>;
}}}

BufferMode Buffer::mode() const {
    if (!impl()){
        return BufferMode::Read;
    }
    return impl()->mode();
}

dimn_t Buffer::size() const {
    if (!impl()) {
        return 0;
    }
    return impl()->size();
}

static inline bool check_device_compatibility(Buffer& dst, const Device& device)
{
    if (dst.is_null() || !device) { return true; }

    RPY_CHECK(dst.device() == device);

    return true;
}


void Buffer::to_device(Buffer& dst, const Device& device)
{
    if (impl() && check_device_compatibility(dst, device)) {
        auto queue = device->get_default_queue();
        impl()->to_device(dst, device, queue).wait();
    }
}
Event Buffer::to_device(Buffer& dst, const Device& device, Queue& queue)
{
    if (impl() && check_device_compatibility(dst, device)) {
        return impl()->to_device(dst, device, queue);
    }
    return {};
}

MemoryView Buffer::map(BufferMode map_mode, dimn_t size, dimn_t offset)
{
    if (!impl() || size == 0) {
        return {*this, nullptr, 0, BufferMode::Read};
    }
    void* mapped = impl()->map(map_mode, size, offset);
    RPY_DBG_ASSERT(mapped != nullptr);
    return MemoryView(*this, mapped, size, BufferMode::Read);
}

void Buffer::unmap(MemoryView& view) noexcept {
    RPY_DBG_ASSERT(impl() != nullptr);
    return impl()->unmap(view.raw_ptr());
}
