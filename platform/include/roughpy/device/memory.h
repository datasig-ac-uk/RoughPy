//
// Created by sam on 20/11/24.
//

#ifndef ROUGHPY_DEVICE_MEMORY_H
#define ROUGHPY_DEVICE_MEMORY_H

#include <atomic>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/reference_counting.h"
#include "roughpy/platform/roughpy_platform_export.h"

#include "roughpy/generics/type_ptr.h"
#include "roughpy/generics/type.h"

namespace rpy::device {

// Forward declaration
class DeviceHandle;

enum class MemoryMode : uint8_t
{
    ReadOnly,
    WriteOnly,
    ReadWrite
};

class ConstMemoryView;
class MutableMemoryView;


class ROUGHPY_PLATFORM_EXPORT Memory
    : public mem::PolymorphicRefCounted
{
    friend class ConstMemoryView;
    friend class MutableMemoryView;




public:

    ~Memory() override;

    RPY_NO_DISCARD virtual size_t size() const noexcept = 0;
    RPY_NO_DISCARD virtual size_t bytes() const noexcept;
    RPY_NO_DISCARD virtual const generics::Type& type() const noexcept = 0;
    RPY_NO_DISCARD virtual MemoryMode mode() const noexcept;

    virtual const void* data() const;
    virtual void* data();

    RPY_NO_DISCARD
    virtual const DeviceHandle& device() const noexcept = 0;
    RPY_NO_DISCARD virtual bool is_null() const noexcept;
    RPY_NO_DISCARD virtual bool empty() const noexcept;

    RPY_NO_DISCARD Rc<Memory> to(const DeviceHandle& device) const;
    RPY_NO_DISCARD Rc<Memory> to_host() const;


    RPY_NO_DISCARD
    virtual MutableMemoryView map_memory(
        size_t offset,
        size_t npos
        ) = 0;

    RPY_NO_DISCARD
    virtual ConstMemoryView map_const_memory(
        size_t offset,
        size_t npos
        ) const = 0;

protected:
    virtual void unmap(const ConstMemoryView& child) const = 0;

    virtual void unmap(const MutableMemoryView& child) = 0;

};



class ConstMemoryView
{
    Rc<const Memory> p_owner;
    const void* p_data;
    size_t m_size;

public:

    ConstMemoryView(Rc<const Memory> owner, const void* data, size_t size)
        : p_owner(std::move(owner)),
          p_data(data),
          m_size(size)
    {}

    ~ConstMemoryView()
    {
        if (p_data) {
            p_owner->unmap(*this);
        }
    }

    template <typename T=void>
    const T* data() const
    {
        if constexpr (is_void_v<T>) {
            return p_data;
        } else {
            RPY_CHECK_EQ(p_owner->type().type_info(), typeid(T));
            return std::launder(static_cast<const T*>(p_data));
        }
    }

    size_t size() const noexcept { return m_size; }

    template <typename T>
    span<const T> as_span() const
    {
        return make_span(data<T>(), size());
    }

};

class MutableMemoryView
{
    Rc<Memory> p_owner;
    void* p_data;
    size_t m_size;

public:

    MutableMemoryView(Rc<Memory> owner, void* data, size_t size)
        : p_owner(std::move(owner)),
          p_data(data),
          m_size(size)
    {}

    ~MutableMemoryView()
    {
        if (p_data) {
            p_owner->unmap(*this);
        }
    }

    RPY_NO_DISCARD
    // ReSharper disable once CppNonExplicitConversionOperator
    operator ConstMemoryView() const noexcept
    {
        return { p_owner, p_data, m_size };
    }


    template <typename T=void>
    T* data()
    {
        if constexpr (is_void_v<T>) {
            return p_data;
        } else {
            RPY_CHECK_EQ(p_owner->type().type_info(), typeid(T));
            return std::launder(static_cast<T*>(p_data));
        }
    }

    size_t size() const noexcept { return m_size; }

    template <typename T>
    span<T> as_span()
    {
        return make_span(data<T>(), size());
    }
};

}// namespace rpy::device

#endif// ROUGHPY_DEVICE_MEMORY_H
