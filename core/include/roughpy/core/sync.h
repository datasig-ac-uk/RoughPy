//
// Created by sam on 5/17/24.
//

#ifndef ROUGHPY_CORE_SYNC_H
#define ROUGHPY_CORE_SYNC_H

#include <mutex>
#include <thread>

#include "traits.h"

namespace rpy {

using std::defer_lock;

using std::lock;

using Mutex = std::mutex;
using RecursiveMutex = std::recursive_mutex;

template <typename M>
using LockGuard = std::lock_guard<M>;

template <typename M>
using UniqueLock = std::unique_lock<M>;

namespace dtl {

/**
 * @brief This class provides a guarded reference to an object. The reference
 * can only be accessed when locked by a mutex.
 *
 * @tparam _T The type of the object being guarded.
 * @tparam _M The type of the mutex used for synchronization.
 */
template <typename _T, typename _M>
class GuardedRef
{
    using reference = add_lvalue_reference_t<_T>;
    using pointer = add_pointer_t<_T>;
    UniqueLock<_M> m_lk;
    reference m_value;

public:
    GuardedRef(reference& value, _M& mutex) : m_lk(mutex), m_value(value) {}

    GuardedRef(const GuardedRef&) = delete;
    GuardedRef& operator=(const GuardedRef&) = delete;

    GuardedRef(GuardedRef& other) noexcept = default;
    GuardedRef& operator=(GuardedRef&&) noexcept = default;

    // ReSharper disable CppNonExplicitConversionOperator
    operator reference() noexcept
    {
        RPY_DBG_ASSERT(m_lk.owns_lock());
        return m_value;
    }
    // ReSharper restore CppNonExplicitConversionOperator

    pointer operator->() noexcept
    {
        RPY_DBG_ASSERT(m_lk.owns_lock());
        return std::addressof(m_value);
    }
};

}// namespace dtl

/**
 * @brief A thread-safe container that holds a value and provides synchronized
 * access to it.
 *
 * GuardedValue is a class template that provides synchronized access to a value
 * held internally. It uses a mutex to ensure thread-safety and allows multiple
 * reader threads and exclusive writer threads.
 *
 * @tparam _T The type of the value to be stored.
 * @tparam _M The type of the mutex to be used for synchronization.
 *
 * @note This class template requires the synchronization to be managed
 * externally using the specified mutex type. @note The value_type and
 * mutex_type aliases provide convenient access to the template parameters.
 */
template <typename _T, typename _M = Mutex>
class GuardedValue
{
    mutable _M m_mutex;

public:
    using value_type = remove_cv_ref_t<_T>;
    using mutex_type = _M;
    using reference = dtl::GuardedRef<value_type, _M>;
    using const_reference = dtl::GuardedRef<const value_type, _M>;

    using pointer = reference;
    using const_pointer = const_reference;

private:
    value_type m_value;

public:
    GuardedValue() = default;

    GuardedValue(const GuardedValue& other) : m_value(other.value()) {}

    GuardedValue(GuardedValue&& other) noexcept
        : m_value(std::move(other.m_value))
    {}

    explicit GuardedValue(const value_type& value) : m_value(value) {}

    explicit GuardedValue(value_type&& value) : m_value(std::move(value)) {}

    template <typename... Args>
    explicit GuardedValue(Args&&... args) : m_value(std::forward<Args>(args)...)
    {}

    template <typename _OM>
    explicit GuardedValue(const GuardedValue<_T, _OM>& other)
    {
        m_value = other.value();
    }

    template <typename _OM>
    explicit GuardedValue(GuardedValue<_T, _OM>&& other) noexcept
        : m_value(other.take())
    {}

    GuardedValue& operator=(const GuardedValue& other)
    {
        if (&other != this) {
            UniqueLock<_M> left_lk(m_mutex, defer_lock);
            UniqueLock<_M> right_lk(other.m_mutex, defer_lock);
            lock(left_lk, right_lk);

            m_value = other.m_value;
        }
        return *this;
    }

    GuardedValue& operator=(_T& value)
    {
        {
            UniqueLock<_M> lk(m_mutex);
            m_value = value;
        }
        return *this;
    }

    template <typename _OM>
    void swap(GuardedValue<_T, _OM>& other)
    {
        if (&other == this) { return; }
        UniqueLock<_M> left_lk(m_mutex, defer_lock);
        UniqueLock<_OM> right_lk(other.mutex(), defer_lock);
        lock(left_lk, right_lk);
        std::swap(m_value, other.m_value);
    }

    void swap(_T& other)
    {
        LockGuard<_M> lk(m_mutex);
        std::swap(m_value, other);
    }

    mutex_type& mutex() const { return m_mutex; }

    value_type value() const
    {
        UniqueLock<_M> lk(m_mutex);
        return m_value;
    }

    value_type take() noexcept { return m_value; }

    reference operator*() { return reference(m_value, m_mutex); }

    const_reference operator*() const
    {
        return const_reference(m_value, m_mutex);
    }

    pointer operator->() { return pointer(m_value, m_mutex); }
    const_pointer operator->() const { return const_pointer(m_value, m_mutex); }
};

}// namespace rpy

#endif// ROUGHPY_CORE_SYNC_H
