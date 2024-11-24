//
// Created by sammorley on 20/11/24.
//

#ifndef ROUGHPY_PLATFORM_REFERENCE_COUNTING_H
#define ROUGHPY_PLATFORM_REFERENCE_COUNTING_H


#include <atomic>

#include "roughpy/core/macros.h"
#include "roughpy/core/smart_ptr.h"
#include "roughpy/core/traits.h"

#include "alloc.h"
#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy {
namespace mem {

class PolymorphicRefCounted;


template <typename T>
enable_if_t<is_base_of_v<PolymorphicRefCounted, T>>
intrusive_ptr_add_ref(const T* ptr) noexcept;

template <typename T>
enable_if_t<is_base_of_v<PolymorphicRefCounted, T>>
intrusive_ptr_release(const T* ptr) noexcept;


/**
 * @class PolymorphicRefCounted
 * @brief A base class implementing reference counting with polymorphism
 * support.
 *
 * The PolymorphicRefCounted class provides a mechanism to manage the lifetime
 * of objects through reference counting. It supports polymorphism, allowing
 * derived classes to be correctly managed even when accessed through base class
 * pointers.
 *
 * Reference counting is a memory management technique where each object has a
 * count of the number of references to it. When the reference count drops to
 * zero, the object is automatically deleted. This class ensures that the
 * reference count is accurately maintained and provides mechanisms to increment
 * and decrement the count safely.
 *
 * The class is intended to be inherited by other classes that require
 * reference-counted lifecycle management.
 *
 * @note Objects of this class should not be instantiated directly. Use derived
 * classes instead.
 *
 */
class ROUGHPY_PLATFORM_EXPORT PolymorphicRefCounted : public SmallObjectBase {

public:

    virtual ~PolymorphicRefCounted();

protected:
    virtual void inc_ref() const noexcept = 0;

    RPY_NO_DISCARD
    virtual bool dec_ref() const noexcept = 0;

public:
    RPY_NO_DISCARD
    virtual intptr_t ref_count() const noexcept = 0;

    template <typename T>
    friend enable_if_t<is_base_of_v<PolymorphicRefCounted, T>>
    intrusive_ptr_add_ref(const T* ptr) noexcept
    {
        static_cast<const PolymorphicRefCounted*>(ptr)->inc_ref();
    }

    template <typename T>
    friend enable_if_t<is_base_of_v<PolymorphicRefCounted, T>>
    intrusive_ptr_release(const T* ptr) noexcept
    {
        if (const auto* base_ptr
            = static_cast<const PolymorphicRefCounted*>(ptr);
            base_ptr->dec_ref()) {
            delete base_ptr;
        }
    }

};


/**
 * @brief A template class for reference counting mechanisms.
 *
 * The RefCountedMiddle class provides an implementation of reference counting
 * in conjunction with a base class that must inherit from
 * PolymorphicRefCounted. It manages the reference count for objects, ensuring
 * proper object lifetime management. The class uses atomic operations to
 * maintain thread safety.
 *
 * This class should be used as a base class for objects that need reference
 * counting. It inherits from a specified base class and extends it with
 * reference counting capabilities.
 *
 * @tparam Base The base class that must inherit from PolymorphicRefCounted.
 *
 * @note The Base class must be derived from PolymorphicRefCounted.
 */
template <typename Base=PolymorphicRefCounted>
class RefCountedMiddle : public Base
{
    static_assert(
            is_base_of_v<PolymorphicRefCounted, Base>,
            "Base class must be derived from PolymorphicRefCounted"
    );

    mutable std::atomic_intptr_t m_ref_count;

public:
    template <typename... Args>
    explicit RefCountedMiddle(Args&&... args)
        : Base(std::forward<Args>(args)...),
          m_ref_count(0)
    {}

protected:
    void inc_ref() const noexcept override;

    RPY_NO_DISCARD bool dec_ref() const noexcept override;

public:
    RPY_NO_DISCARD intptr_t ref_count() const noexcept override;
};

template <typename Base>
void RefCountedMiddle<Base>::inc_ref() const noexcept
{
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
}
template <typename Base>
bool RefCountedMiddle<Base>::dec_ref() const noexcept
{
    return m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1;
}
template <typename Base>
intptr_t RefCountedMiddle<Base>::ref_count() const noexcept
{
    return m_ref_count.load(std::memory_order_acquire);
}


};

using mem::intrusive_ptr_add_ref;
using mem::intrusive_ptr_release;

}


#endif //ROUGHPY_PLATFORM_REFERENCE_COUNTING_H
