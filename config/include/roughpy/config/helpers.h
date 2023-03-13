#ifndef ROUGHPY_CONFIG_POINTER_HELPERS_H_
#define ROUGHPY_CONFIG_POINTER_HELPERS_H_

#include "implementation_types.h"
#include "traits.h"
#include "macros.h"

#include <cstring>
#include <iterator>

namespace rpy {


/**
 * @brief Cast the bit value from a value of type From to a value
 * of type To.
 *
 * We're using the same conditions on From and To as the Abseil library
 * if we have to define our own version using memcpy.
 */
#if defined(__cpp_lib_bit_cast) && __cpp_Lib_bit_cast >= 201806L
using std::bit_cast;
#else
template <typename To, typename From>
traits::enable_if_t<
    sizeof(To) == sizeof(From)
    && traits::is_trivially_copyable<From>::value
    && traits::is_trivially_copyable<To>::value
    && traits::is_default_constructible<To>::value
    , To>
bit_cast(From from) {
    To to;
    memcpy(static_cast<void*>(std::addressof(to)),
           static_cast<const void*>(std::addressof(from)),
           sizeof(To));
    return to;
}
#endif



/**
 * @brief
 * @tparam T
 */
template <typename T>
class MaybeOwned {
    enum State {
        IsOwned,
        IsBorrowed
    };

    T* p_data;
    State m_state;

public:

    constexpr MaybeOwned(nullptr_t) : p_data(nullptr), m_state(IsOwned) {}
    constexpr MaybeOwned(T* ptr) : p_data(ptr), m_state(IsBorrowed) {}

    ~MaybeOwned() {
        if (m_state == IsOwned) {
            delete[] p_data;
        }
    }

    constexpr MaybeOwned& operator=(T* ptr) {
        p_data = ptr;
        m_state = IsOwned;
        return *this;
    }

    operator T* () const noexcept { return p_data; }

    operator bool() const noexcept { return p_data != nullptr; }

};



template <typename T>
class Slice {
    T* p_data = nullptr;
    std::size_t m_size = nullptr;


public:

    template <typename Container, typename=traits::enable_if_t<
        traits::is_same<
            typename Container::value_type, T>::value
        >>
    Slice(Container& container)
        : p_data(container.data()), m_size(container.size())
    {}

    template <std::size_t N>
    Slice(T (&array)[N])
        : p_data(array), m_size(N)
    {}

    Slice(T* ptr, std::size_t N)
        : p_data(ptr), m_size(N)
    {}

    template <typename I>
    traits::enable_if_t<
        traits::is_integral<I>::value,
        const T&>
    operator[](I i) noexcept {
        assert(0 <= i && i < m_size);
        return p_data[i];
    }

    template <typename I>
    traits::enable_if_t<
        traits::is_integral<I>::value,
        T&>
    operator[](I i) const noexcept {
        assert(0 <= i && i < m_size);
        return p_data[i];
    }

    std::size_t size() const noexcept {
        return m_size;
    }

    T* begin() noexcept { return p_data; }
    T* end() noexcept { return p_data + m_size; }
    const T* begin() const { return p_data; }
    const T* end() const { return p_data + m_size; }

    std::reverse_iterator<T*> rbegin() noexcept
    { return std::reverse_iterator<T*>(p_data + m_size); }
    std::reverse_iterator<T*> rend() noexcept
    { return std::reverse_iterator<T*>(p_data); }
    std::reverse_iterator<const T*> rbegin() const noexcept
    { return std::reverse_iterator<const T*>(p_data + m_size); }
    std::reverse_iterator<const T*> rend() const noexcept
    { return std::reverse_iterator<const T*>(p_data); }


};





}

#endif // ROUGHPY_CONFIG_POINTER_HELPERS_H_
