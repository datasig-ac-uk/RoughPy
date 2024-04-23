// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_DEVICE_OBJECT_BASE_H_
#define ROUGHPY_DEVICE_DEVICE_OBJECT_BASE_H_

#include "core.h"
#include "roughpy/core/smart_ptr.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <atomic>
#include <memory>

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
    RPY_NO_DISCARD void* operator new(std::size_t count);

    void operator delete(void* ptr, std::size_t count);

    virtual ~InterfaceBase();

    RPY_NO_DISCARD virtual bool is_host() const noexcept;
    RPY_NO_DISCARD virtual DeviceType type() const noexcept;
    RPY_NO_DISCARD virtual dimn_t ref_count() const noexcept;
    RPY_NO_DISCARD virtual Device device() const noexcept;

    RPY_NO_DISCARD virtual void* ptr() noexcept;
    RPY_NO_DISCARD virtual const void* ptr() const noexcept;

protected:
    virtual rc_count_t inc_ref() const noexcept;
    virtual rc_count_t dec_ref() const noexcept;

    friend void intrusive_ptr_add_ref(const InterfaceBase* p) noexcept
    {
        auto used = p->inc_ref();
        (void) used;
    }

    friend void intrusive_ptr_release(const InterfaceBase* p) noexcept
    {
        if (p->dec_ref() == 0) { delete p; }
    }
};

template <typename Interface>
class RefCountBase : public Interface
{
    static_assert(
            is_base_of_v<InterfaceBase, Interface>,
            "Interface must be derived from InterfaceBase"
    );

    using RcPolicy = boost::thread_safe_counter;
    using atomic_t = std::atomic_size_t;
    mutable typename RcPolicy::type m_ref_count;

public:
    template <typename... Args>
    explicit RefCountBase(Args&&... args)
        : Interface(std::forward<Args>(args)...),
          m_ref_count(0)
    {}

    rc_count_t inc_ref() const noexcept override;
    rc_count_t dec_ref() const noexcept override;
    rc_count_t ref_count() const noexcept override;
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

    Rc<Interface> p_impl = nullptr;

    friend typename Interface::object_t
    rpy::devices::steal_cast<Interface>(Interface*) noexcept;

    friend typename Interface::object_t
    rpy::devices::clone_cast<Interface>(Interface*) noexcept;

    struct steal_t {
    };

protected:
    RPY_NO_DISCARD Interface* impl() noexcept { return p_impl.get(); }
    RPY_NO_DISCARD const Interface* impl() const noexcept
    {
        return p_impl.get();
    }

public:
    ObjectBase() = default;

    ObjectBase(const ObjectBase& other);
    ObjectBase(ObjectBase&& other) noexcept;

    explicit ObjectBase(Interface* iface) noexcept : p_impl(iface) {}

    explicit ObjectBase(Rc<Interface> iface) noexcept : p_impl(std::move(iface))
    {}

    explicit ObjectBase(Interface* iface, steal_t) noexcept
        : p_impl(iface, false)
    {}

    ObjectBase& operator=(const ObjectBase& other);
    ObjectBase& operator=(ObjectBase&& other) noexcept;

    RPY_NO_DISCARD bool is_host() const noexcept;
    RPY_NO_DISCARD DeviceType type() const noexcept;
    RPY_NO_DISCARD bool is_null() const noexcept { return !p_impl; }
    RPY_NO_DISCARD rc_count_t ref_count() const noexcept;
    RPY_NO_DISCARD Derived clone() const;
    RPY_NO_DISCARD Device device() const noexcept;

    RPY_NO_DISCARD void* ptr() noexcept;
    RPY_NO_DISCARD const void* ptr() const noexcept;

    RPY_NO_DISCARD Interface& get() noexcept
    {
        RPY_DBG_ASSERT(static_cast<bool>(p_impl));
        return *p_impl;
    }
    RPY_NO_DISCARD const Interface& get() const noexcept
    {
        RPY_DBG_ASSERT(static_cast<bool>(p_impl));
        return *p_impl;
    }

    bool operator==(const ObjectBase& other) const noexcept
    {
        return p_impl == other.p_impl;
    }
};

template <typename Interface>
rc_count_t RefCountBase<Interface>::inc_ref() const noexcept
{
    // return m_ref_count.fetch_add(1, std::memory_order_relaxed);
    RcPolicy::increment(m_ref_count);
    return 0;
}

template <typename Interface>
rc_count_t RefCountBase<Interface>::dec_ref() const noexcept
{
    // RPY_DBG_ASSERT(m_ref_count.load(std::memory_order_acquire) > 0);
    // return m_ref_count.fetch_sub(1, std::memory_order_acq_rel);
    return static_cast<rc_count_t>(RcPolicy::decrement(m_ref_count));
}
template <typename Interface>
rc_count_t RefCountBase<Interface>::ref_count() const noexcept
{
    // return m_ref_count.load(std::memory_order_relaxed);
    return RcPolicy::load(m_ref_count);
}
template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>::ObjectBase(const ObjectBase& other)
    : p_impl(other.p_impl)
{}
template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>::ObjectBase(ObjectBase&& other) noexcept
    : p_impl(std::move(other.p_impl))
{}
template <typename Interface, typename Derived>
bool ObjectBase<Interface, Derived>::is_host() const noexcept
{
    return p_impl == nullptr || p_impl->is_host();
}

template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>&
ObjectBase<Interface, Derived>::operator=(const ObjectBase& other)
{
    RPY_DBG_ASSERT(p_impl == nullptr || p_impl->ref_count() > 0);
    RPY_DBG_ASSERT(other.p_impl == nullptr || other.p_impl->ref_count() > 0);
    if (&other != this) {
        this->~ObjectBase();
        if (other.p_impl != nullptr) { p_impl = other.p_impl; }
    }
    return *this;
}
template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>&
ObjectBase<Interface, Derived>::operator=(ObjectBase&& other) noexcept
{
    RPY_DBG_ASSERT(p_impl == nullptr || p_impl->ref_count() > 0);
    RPY_DBG_ASSERT(other.p_impl == nullptr || other.p_impl->ref_count() > 0);
    if (&other != this) {
        this->~ObjectBase();
        std::swap(p_impl, other.p_impl);
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
rc_count_t ObjectBase<Interface, Derived>::ref_count() const noexcept
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

template <typename Interface>
inline typename Interface::object_t clone_cast(const Interface* iface) noexcept
{
    using object_t = typename Interface::object_t;
    // const_cast is safe because this just creates a new strong reference to
    // the interface object.
    return object_t(const_cast<Interface*>(iface));
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_DEVICE_OBJECT_BASE_H_
