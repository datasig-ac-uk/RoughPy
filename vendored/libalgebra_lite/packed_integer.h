//
// Created by user on 13/08/22.
//

#ifndef LIBALGEBRA_LITE_PACKED_INTEGER_H
#define LIBALGEBRA_LITE_PACKED_INTEGER_H

#include <limits>
#include <type_traits>
#include <ostream>

namespace lal {
namespace dtl {


template <typename PacktedInt, typename RefType>
class packed_integer_ref;



template<typename Int, typename Packed>
class packed_integer
{
    static_assert(sizeof(Int) > sizeof(Packed),
            "Integral type must contain more bits than packed type");

    Int m_data;

    static constexpr int remaining_bits = 8*(sizeof(Int)-sizeof(Packed));
    static constexpr Int integral_mask = (Int(1) << remaining_bits) - Int(1);
    static constexpr Int packed_mask = ~integral_mask;

    friend class packed_integer_ref<packed_integer, Int>;
    friend class packed_integer_ref<packed_integer, const Int>;
    friend class packed_integer_ref<packed_integer, Packed>;
    friend class packed_integer_ref<packed_integer, const Packed>;


    using integer_ref = packed_integer_ref<packed_integer, Int>;
    using const_integer_ref = packed_integer_ref<packed_integer, const Int>;
    using packed_ref = packed_integer_ref<packed_integer, Packed>;
    using const_packed_ref = packed_integer_ref<packed_integer, const Packed>;

#ifdef LAL_TESTING
    friend struct packed_integer_access;
#endif

public:

    using integral_type = Int;
    using packed_type = Packed;

    explicit constexpr packed_integer(Int val) : m_data(val & integral_mask)
    {}

    explicit constexpr packed_integer(Packed packed)
        : m_data(static_cast<Int>(packed)<<remaining_bits)
    {}

    constexpr packed_integer(Packed packed, Int val)
        : m_data((static_cast<Int>(packed)<<remaining_bits) + (val & integral_mask))
    {}

    constexpr explicit operator Int() const noexcept
    {
        return m_data & integral_mask;
    }
    constexpr explicit operator Packed() const noexcept
    {
        return (m_data & packed_mask) >> remaining_bits;
    }

    operator integer_ref() noexcept // NOLINT(google-explicit-constructor)
    {
        return integer_ref(m_data);
    }

    operator packed_ref() noexcept // NOLINT(google-explicit-constructor)
    {
        return packed_ref(m_data);
    }

    constexpr bool operator==(const packed_integer& arg) const noexcept
    { return m_data == arg.m_data; }
    constexpr bool operator!=(const packed_integer& arg) const noexcept
    { return m_data != arg.m_data; }
    constexpr bool operator<(const packed_integer& arg) const noexcept
    { return m_data < arg.m_data; }
    constexpr bool operator<=(const packed_integer& arg) const noexcept
    { return m_data <= arg.m_data; }
    constexpr bool operator>(const packed_integer& arg) const noexcept
    { return m_data > arg.m_data; }
    constexpr bool operator>=(const packed_integer& arg) const noexcept
    { return m_data >= arg.m_data; }

    friend std::size_t hash_value(const packed_integer& value) noexcept {
        std::hash<Int> hasher;
        return hasher(value.m_data);
    }

};


template <typename Int, typename Packed>
std::ostream& operator<<(std::ostream& os, const packed_integer<Int, Packed>& arg) noexcept
{
    return os << Packed(arg) << Int(arg);
}


template <typename S, typename T>
struct copy_constness
{
    using type = T;
};

template <typename S, typename T>
struct copy_constness<const S, T>
{
    using type = const T;
};

template <typename S, typename T>
using copy_constness_t = typename copy_constness<S, T>::type;


#define LAL_ASSERT_NOT_CONST   \
    static_assert(!std::is_const<RefType>::value, \
        "Reference type must not be const")

template <typename PackedInteger, typename RefType>
class packed_integer_ref {

    using main_ref_t = copy_constness_t<RefType, typename PackedInteger::integral_type>&;
    main_ref_t m_ref;
    RefType m_dummy;

    static constexpr RefType make_dummy(main_ref_t ref) noexcept
    {
        return (std::is_same<
                std::remove_cv_t<RefType>,
                typename PackedInteger::packed_type
        >::value)
               ? RefType((ref & PackedInteger::packed_mask) >> PackedInteger::remaining_bits)
               : RefType(ref & PackedInteger::integral_mask);

    }

    bool updated_referenced(std::add_lvalue_reference_t<std::add_const_t<RefType>> arg)
    {
        using trait = std::is_same<std::remove_cv_t<RefType>,
                typename PackedInteger::packed_type>;
        auto old = make_dummy(m_ref);
        auto change = arg - old;
        m_ref += ((trait::value) ? (change << PackedInteger::remaining_bits)
                                 : change);
        return change != 0;
    }

public:

    explicit packed_integer_ref(main_ref_t data)
            :m_ref(data), m_dummy(make_dummy(data)) { }

    operator std::add_lvalue_reference_t<std::add_const_t<RefType>>() const noexcept
    {
        return m_dummy;
    }

    packed_integer_ref& operator=(std::add_lvalue_reference_t<std::add_const_t<RefType>> arg) noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(arg)) {
            m_dummy = arg;
        }
        return *this;
    }

    packed_integer_ref& operator=(std::add_rvalue_reference_t<std::remove_cv_t<RefType>> arg) noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(arg)) {
            m_dummy = arg;
        }
        return *this;
    }

    template <typename Int>
    packed_integer_ref& operator=(Int arg) noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(RefType(arg))) {
            m_dummy = arg;
        }
        return *this;
    }

    packed_integer_ref& operator++() noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(m_dummy+1)) {
            ++m_dummy;
        }
        return *this;
    }

    packed_integer_ref& operator--() noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(m_dummy-1)) {
            --m_dummy;
        }
        return *this;
    }

    template <typename Int>
    packed_integer_ref& operator+=(Int arg) noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(m_dummy+arg)) {
            m_dummy += arg;
        }
        return *this;
    }

    template <typename Int>
    packed_integer_ref& operator-=(Int arg) noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(m_dummy-arg)) {
            m_dummy -= arg;
        }
        return *this;
    }

    template <typename Int>
    packed_integer_ref& operator*=(Int arg) noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(m_dummy*arg)) {
            m_dummy *= arg;
        }
        return *this;
    }


    template <typename Int>
    packed_integer_ref& operator/=(Int arg) noexcept
    {
        LAL_ASSERT_NOT_CONST;
        if (updated_referenced(m_dummy/arg)) {
            m_dummy /= arg;
        }
        return *this;
    }

};

#undef LAL_ASSERT_NOT_CONST


} // namespace dtl
} // namespace alg


#endif //LIBALGEBRA_LITE_PACKED_INTEGER_H
