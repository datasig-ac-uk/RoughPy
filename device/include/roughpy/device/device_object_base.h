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

#include "core.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <memory>

namespace rpy {
namespace devices {
namespace dtl {

class RPY_EXPORT InterfaceBase
{
public:
    virtual ~InterfaceBase();

    RPY_NO_DISCARD virtual DeviceType type() const noexcept;
    RPY_NO_DISCARD virtual dimn_t ref_count() const noexcept;
    RPY_NO_DISCARD virtual std::unique_ptr<InterfaceBase> clone() const;
    RPY_NO_DISCARD virtual Device device() const noexcept;

    RPY_NO_DISCARD virtual void* ptr() noexcept;
    RPY_NO_DISCARD virtual const void* ptr() const noexcept;

};

template <typename Interface, typename Derived>
class ObjectBase
{
    static_assert(
            is_base_of<InterfaceBase, Interface>::value,
            "Interface must be derived from InterfaceBase"
    );

    friend class rpy::devices::DeviceHandle;
    friend class InterfaceBase;

    using interface_type = Interface;

    static std::unique_ptr<Interface>
    downcast(std::unique_ptr<InterfaceBase>&& base) noexcept
    {
        return std::unique_ptr<Interface>(
                reinterpret_cast<Interface*>(base.release())
        );
    }

protected:
    std::unique_ptr<Interface> p_impl;

private:
    explicit ObjectBase(std::unique_ptr<InterfaceBase>&& base)
        : p_impl(downcast(std::move(base)))
    {}

public:
    ObjectBase() = default;

    ObjectBase(const ObjectBase& other);
    ObjectBase(ObjectBase&& other) noexcept = default;

    template <
            typename IFace,
            typename = enable_if_t<is_base_of<Interface, IFace>::value>>
    explicit ObjectBase(std::unique_ptr<IFace>&& base) : p_impl(std::move(base))
    {}

    explicit ObjectBase(std::unique_ptr<Interface>&& base)
        : p_impl(std::move(base))
    {}

    ObjectBase& operator=(const ObjectBase& other);
    ObjectBase& operator=(ObjectBase&& other) noexcept = default;

    RPY_NO_DISCARD DeviceType type() const noexcept;
    RPY_NO_DISCARD bool is_null() const noexcept { return !p_impl; }
    RPY_NO_DISCARD dimn_t ref_count() const noexcept;
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

template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>::ObjectBase(const ObjectBase& other)
    : p_impl(nullptr)
{
    if (other.p_impl) {
        p_impl = std::move(downcast(std::move(other.p_impl->clone())));
    }
}
template <typename Interface, typename Derived>
ObjectBase<Interface, Derived>&
ObjectBase<Interface, Derived>::operator=(const ObjectBase& other)
{
    if (&other != this) {
        this->~ObjectBase();
        if (other.p_impl) {
            p_impl = std::move(downcast(std::move(other.p_impl->clone())));
        } else {
            p_impl = nullptr;
        }
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
dimn_t ObjectBase<Interface, Derived>::ref_count() const noexcept
{
    if (p_impl) { return p_impl->ref_count(); }
    return 1;
}

template <typename Interface, typename Derived>
Derived ObjectBase<Interface, Derived>::clone() const
{
    RPY_CHECK(p_impl);
    return Derived(p_impl->clone());
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

template <typename I>
RPY_NO_DISCARD
constexpr enable_if_t<
        is_base_of<InterfaceBase, remove_reference_t<I>>::value,
        add_lvalue_reference_t<I>>
device_cast(copy_cv_ref_t<InterfaceBase, remove_reference_t<I>> interface)
        noexcept
{
    return static_cast<add_lvalue_reference_t<I>>(interface);
}

}// namespace dtl
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_DEVICE_OBJECT_BASE_H_
