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

#ifndef ROUGHPY_DEVICE_EVENT_H_
#define ROUGHPY_DEVICE_EVENT_H_

#include "core.h"

#include "device_object_base.h"
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace devices {

class ROUGHPY_PLATFORM_EXPORT EventInterface : public dtl::InterfaceBase
{
public:
    using object_t = Event;

    virtual void wait();

    RPY_NO_DISCARD virtual EventStatus status() const;

    RPY_NO_DISCARD virtual bool is_user() const noexcept;

    virtual void set_status(EventStatus status);
};

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RoughPy_Platform_EXPORTS
namespace dtl {
extern template class ObjectBase<EventInterface, Event>;
}
#  else
namespace dtl {
template class RPY_DLL_IMPORT ObjectBase<EventInterface, Event>;
}
#  endif
#else
namespace dtl {
extern template class ROUGHPY_PLATFORM_EXPORT ObjectBase<EventInterface, Event>;
}
#endif

class ROUGHPY_PLATFORM_EXPORT Event
    : public dtl::ObjectBase<EventInterface, Event>
{
    using base_t = dtl::ObjectBase<EventInterface, Event>;

public:
    using base_t::base_t;

    void wait();

    RPY_NO_DISCARD
    EventStatus status() const;


    RPY_NO_DISCARD bool is_user() const noexcept;

    void set_status(EventStatus status);


};


template <typename T>
class Future : public Event {
    T m_value;

public:

    Future(Event&& event, T&& value)
        : Event(std::move(event)), m_value(std::move(value))
    {}

    T& value() {
        wait();
        return m_value;
    }

};


}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_EVENT_H_
