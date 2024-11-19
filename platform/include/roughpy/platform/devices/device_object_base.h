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

#ifndef ROUGHPY_DEVICE_DEVICE_OBJECT_BASE_H_
#define ROUGHPY_DEVICE_DEVICE_OBJECT_BASE_H_

#include <atomic>
#include <memory>

#include "roughpy/core/check.h"
#include <roughpy/core/debug_assertion.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "core.h"

namespace rpy {
namespace devices {

template <typename Interface>
inline typename Interface::object_t steal_cast(Interface* iface) noexcept;

template <typename Interface>
inline typename Interface::object_t clone_cast(Interface* iface) noexcept;

namespace dtl {

class ROUGHPY_PLATFORM_EXPORT InterfaceBase
{
public:
    using reference_count_type = dimn_t;

    virtual ~InterfaceBase();

    RPY_NO_DISCARD virtual DeviceType type() const noexcept;
    RPY_NO_DISCARD virtual dimn_t ref_count() const noexcept;
    RPY_NO_DISCARD virtual std::unique_ptr<InterfaceBase> clone() const;
    RPY_NO_DISCARD virtual Device device() const noexcept;

    RPY_NO_DISCARD virtual void* ptr() noexcept;
    RPY_NO_DISCARD virtual const void* ptr() const noexcept;

    virtual reference_count_type inc_ref() noexcept;
    virtual reference_count_type dec_ref() noexcept;
};

template <typename Interface>
class RefCountBase : public Interface
{
    static_assert(
            is_base_of_v<InterfaceBase, Interface>,
            "Interface must be derived from InterfaceBase"
    );

    using atomic_t = std::atomic_size_t;
    atomic_t m_ref_count;

public:
    using Interface::Interface;
    using reference_count_type = dimn_t;

    reference_count_type inc_ref() noexcept override;
    reference_count_type dec_ref() noexcept override;
    reference_count_type ref_count() const noexcept override;
};

template <typename Interface, typename Derived>
class ObjectBase
{
    static_assert(
            is_base_of_v<InterfaceBase, Interface>,
            "Interface must be derived from InterfaceBase"
    );

    friend class rpy::devices::DeviceHandle;
    friend class InterfaceBase;
    friend Interface;

    using interface_type = Interface;

    Interface* p_impl = nullptr;

    friend typename Interface::object_t
    rpy::devices::steal_cast<Interface>(Interface*) noexcept;

    friend typename Interface::object_t
    rpy::devices::clone_cast<Interface>(Interface*) noexcept;

    struct steal_t {
    };


protected:
    RPY_NO_DISCARD Interface* impl() noexcept { return p_impl; }
    RPY_NO_DISCARD const Interface* impl() const noexcept { return p_impl; }

public:
    using reference_count_type = typename InterfaceBase::reference_count_type;

    ObjectBase() = default;

    ObjectBase(const ObjectBase& other);
    ObjectBase(ObjectBase&& other) noexcept;

    explicit ObjectBase(Interface* iface) noexcept : p_impl(iface)
    {
        if (p_impl) { p_impl->inc_ref(); }
    }

    explicit ObjectBase(Interface* iface, steal_t) noexcept : p_impl(iface) {}


    ~ObjectBase();

    ObjectBase& operator=(const ObjectBase& other);
    ObjectBase& operator=(ObjectBase&& other) noexcept;

    RPY_NO_DISCARD DeviceType type() const noexcept;
    RPY_NO_DISCARD bool is_null() const noexcept { return !p_impl; }
    RPY_NO_DISCARD reference_count_type ref_count() const noexcept;
    RPY_NO_DISCARD Derived clone() const;
    RPY_NO_DISCARD Device device() const noexcept;

    RPY_NO_DISCARD void* ptr() noexcept;
    RPY_NO_DISCARD const void* ptr() const noexcept;

    RPY_NO_DISCARD Interface& get() noexcept
    {
        RPY_DBG_ASSERT(!!p_impl);
        return *p_impl;
    }
    RPY_NO_DISCARD const Interface& get() const noexcept
    {
        RPY_DBG_ASSERT(!!p_impl);
        return *p_impl;
    }
};

template <typename Interface>
InterfaceBase::reference_count_type RefCountBase<Interface>::inc_ref() noexcept
{
    return m_ref_count.fetch_add(1, std::memory_order_relaxed);
}
template <typename Interface>
InterfaceBase::reference_count_type RefCountBase<Interface>::dec_ref() noexcept
{
    RPY_DBG_ASSERT(m_ref_count.load(std::memory_order_acquire) > 0);
    return m_ref_count.fetch_sub(1, std::memory_order_acq_rel);
}
template <typename Interface>
InterfaceBase::reference_count_type
RefCountBase<Interface>::ref_count() const noexcept
{
    return m_ref_count.load(std::memory_order_relaxed);
}
template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>::ObjectBase(const ObjectBase& other)
    : p_impl(other.p_impl)
{
    if (p_impl) { p_impl->inc_ref(); }
}
template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>::ObjectBase(ObjectBase&& other) noexcept
    : p_impl(other.p_impl)
{
    other.p_impl = nullptr;
}
template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>::~ObjectBase()
{
    RPY_DBG_ASSERT(!p_impl || p_impl->ref_count() > 0);
    if (p_impl && p_impl->dec_ref() == 1) { delete p_impl; }
    p_impl = nullptr;
}

template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>&
ObjectBase<Interface, Derived>::operator=(const ObjectBase& other)
{
    if (&other != this) {
        this->~ObjectBase();
        if (other.p_impl) {
            p_impl = other.p_impl;
            p_impl->inc_ref();
        }
    }
    return *this;
}
template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>&
ObjectBase<Interface, Derived>::operator=(ObjectBase&& other) noexcept
{
    if (&other != this) {
        this->~ObjectBase();
        p_impl = other.p_impl;
        other.p_impl = nullptr;
    }
    return *this;
}
template <typename Interface, typename Derived>
DeviceType ObjectBase<Interface, Derived>::type() const noexcept
{
    if (p_impl) { return p_impl->type(); }
    return DeviceType::CPU;
}

template <typename Interface, typename Derived>
typename ObjectBase<Interface, Derived>::reference_count_type
ObjectBase<Interface, Derived>::ref_count() const noexcept
{
    if (p_impl) { return p_impl->ref_count(); }
    return 0;
}

template <typename Interface, typename Derived>
Derived ObjectBase<Interface, Derived>::clone() const
{
    RPY_CHECK(p_impl != nullptr);
    return Derived(p_impl);
}

template <typename Interface, typename Derived>
Device devices::dtl::ObjectBase<Interface, Derived>::device() const noexcept
{
    if (p_impl) { return p_impl->device(); }
    return nullptr;
}
template <typename Interface, typename Derived>
void* ObjectBase<Interface, Derived>::ptr() noexcept
{
    if (p_impl) { return p_impl->ptr(); }
    return nullptr;
}
template <typename Interface, typename Derived>
const void* ObjectBase<Interface, Derived>::ptr() const noexcept
{
    if (p_impl) { return p_impl->ptr(); }
    return nullptr;
}

}// namespace dtl

template <typename Interface>
inline typename Interface::object_t steal_cast(Interface* iface) noexcept
{
    using object_t = typename Interface::object_t;
    return object_t(
            iface,
            typename dtl::ObjectBase<Interface, object_t>::steal_t()
    );
}

template <typename Interface>
inline typename Interface::object_t clone_cast(Interface* iface) noexcept
{
    using object_t = typename Interface::object_t;
    return object_t(iface);
}
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_DEVICE_OBJECT_BASE_H_
