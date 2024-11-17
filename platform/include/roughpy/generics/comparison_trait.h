//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_COMPARISON_TRAIT_H
#define ROUGHPY_GENERICS_COMPARISON_TRAIT_H

#include "type_ptr.h"


#include "roughpy/platform/roughpy_platform_export.h"

#include "builtin_trait.h"
#include "roughpy/core/debug_assertion.h"
#include "type.h"

namespace rpy::generics {

class ConstRef;


class ROUGHPY_PLATFORM_EXPORT ComparisonTrait : public BuiltinTrait
{
    const Type* p_type;
protected:

    constexpr explicit ComparisonTrait(const Type* type) : BuiltinTrait
    (my_id), p_type(type)
    {}

public:

    static constexpr BuiltinTraitID my_id = BuiltinTraitID::Comparison;

    virtual ~ComparisonTrait();

    RPY_NO_DISCARD
    virtual bool has_comparison(ComparisonType comp) const noexcept = 0;

    RPY_NO_DISCARD
    virtual bool unsafe_compare_equal(const void* lhs, const void* rhs) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool unsafe_compare_less(const void* lhs, const void* rhs) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool unsafe_compare_less_equal(const void* lhs, const void* rhs) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool unsafe_compare_greater(const void* lhs, const void* rhs) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool unsafe_compare_greater_equal(const void* lhs, const void* rhs) const noexcept = 0;

    RPY_NO_DISCARD
    bool compare_equal(ConstRef lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    bool compare_less(ConstRef lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    bool compare_less_equal(ConstRef lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    bool compare_less_greater(ConstRef lhs, ConstRef rhs) const;

    RPY_NO_DISCARD
    bool compare_less_greater_equal(ConstRef lhs, ConstRef rhs) const;

};



template <typename T>
class ComparisonTraitImpl : public ComparisonTrait
{
public:

    explicit ComparisonTraitImpl(const Type* type);

    RPY_NO_DISCARD bool has_comparison(ComparisonType comp) const noexcept override;

    RPY_NO_DISCARD
    bool unsafe_compare_equal(const void* lhs, const void* rhs) const noexcept override;
    RPY_NO_DISCARD
    bool unsafe_compare_less(const void* lhs, const void* rhs) const noexcept override;
    RPY_NO_DISCARD
    bool unsafe_compare_less_equal(const void* lhs, const void* rhs) const noexcept override;
    RPY_NO_DISCARD
    bool unsafe_compare_greater(const void* lhs, const void* rhs) const noexcept override;
    RPY_NO_DISCARD
    bool unsafe_compare_greater_equal(const void* lhs, const void* rhs) const noexcept override;
};

namespace dtl {

template <typename T>
using equals_test_t
        = decltype(std::declval<const T&>() == std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_equal_test_v = false;

template <typename T>
inline constexpr bool has_equal_test_v<T, void_t<equals_test_t<T>>> = true;

template <typename T>
using less_test_t = decltype(std::declval<const T&>() < std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_less_test_v = false;

template <typename T>
inline constexpr bool has_less_test_v<T, void_t<less_test_t<T>>> = true;

template <typename T>
using less_equal_test_t
        = decltype(std::declval<const T&>() <= std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_less_equal_test_v = false;

template <typename T>
inline constexpr bool has_less_equal_test_v<T, void_t<less_equal_test_t<T>>>
        = true;

template <typename T>
using greater_test_t
        = decltype(std::declval<const T&>() > std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_greater_test_v = false;

template <typename T>
inline constexpr bool has_greater_test_v<T, void_t<greater_test_t<T>>> = true;

template <typename T>
using greater_equal_test_t
        = decltype(std::declval<const T&>() >= std::declval<const T&>());

template <typename T, typename = void>
inline constexpr bool has_greater_equal_test_v = false;

template <typename T>
inline constexpr bool
        has_greater_equal_test_v<T, void_t<greater_equal_test_t<T>>>
        = true;

}// namespace dtl

template <typename T>
ComparisonTraitImpl<T>::ComparisonTraitImpl(const Type* type)
    : ComparisonTrait(type)
{
    RPY_DBG_ASSERT_EQ(type->type_info(), typeid(T));
}

template <typename T>
bool ComparisonTraitImpl<T>::has_comparison(ComparisonType comp) const noexcept
{
    switch (comp) {
        case ComparisonType::Equal:
            return dtl::has_equal_test_v<T>;
        case ComparisonType::Less:
            return dtl::has_less_test_v<T>;
        case ComparisonType::LessEqual:
            return dtl::has_less_equal_test_v<T> || (dtl::has_less_test_v<T> && dtl::has_equal_test_v<T>);
        case ComparisonType::Greater:
            return dtl::has_greater_test_v<T>;
        case ComparisonType::GreaterEqual:
            return dtl::has_greater_equal_test_v<T> || (dtl::has_greater_test_v<T> && dtl::has_equal_test_v<T>);
    }
    RPY_UNREACHABLE_RETURN();
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::Equal));
    return *static_cast<const T*>(lhs) == *static_cast<const T*>(rhs);
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_less(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::Less));
    return *static_cast<const T*>(lhs) < *static_cast<const T*>(rhs);
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_less_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::LessEqual));
    if constexpr (dtl::has_less_equal_test_v<T>) {
        return *static_cast<const T*>(lhs) <= *static_cast<const T*>(rhs);
    } else {
        return (*static_cast<const T*>(lhs) < *static_cast<const T*>(rhs))
             || (*static_cast<const T*>(lhs) == *static_cast<const T*>(rhs));
    }
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_greater(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::Greater));
    return *static_cast<const T*>(lhs) > *static_cast<const T*>(rhs);
}
template <typename T>
bool ComparisonTraitImpl<T>::unsafe_compare_greater_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT(has_comparison(ComparisonType::GreaterEqual));
    if constexpr (dtl::has_greater_equal_test_v<T>) {
        return *static_cast<const T*>(lhs) >= *static_cast<const T*>(rhs);
    } else {
        return (*static_cast<const T*>(lhs) > *static_cast<const T*>(rhs))
             || (*static_cast<const T*>(lhs) == *static_cast<const T*>(rhs));
    }
}

}


#endif //ROUGHPY_GENERICS_COMPARISON_TRAIT_H
