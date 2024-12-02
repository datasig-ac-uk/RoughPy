//
// Created by sammorley on 02/12/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_CONVERSION_IMPL_H
#define ROUGHPY_GENERICS_INTERNAL_CONVERSION_IMPL_H


#include <memory>

#include "roughpy/core/hash.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/meta.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/generics/conversion_trait.h"
#include "roughpy/generics/type_ptr.h"
#include "roughpy/generics/type.h"


namespace rpy::generics::conv {



/**
 * @brief Represents the result of a type conversion operation.
 *
 * ConversionResult is an enumeration used to indicate the outcome of
 * a conversion process, which can either be successful, inexact, or failed.
 */
enum class ConversionResult
{
    Success,
    Inexact,
    Failed
};


template <typename From, typename To, typename=void>
struct ConversionHelper
{
    using from_ptr = const From*;
    using to_ptr = To*;

    static constexpr bool is_possible = false;

    // Set to true if the conversion is guaranteed to be exact
    static constexpr bool is_always_exact = false;

    // The conversion implementation. Should only check for an inexact
    // conversion if the ensure_exact flag is set (and if the is_always_exact is
    // false).
    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact);

};

/**
 * @brief Specialized implementation of the ConversionTrait for specific types.
 *
 * The ConversionTraitImpl class provides concrete functionality for type
 * conversion between specified source and destination types. It includes
 * methods for checking if the conversion is exact and performing the conversion
 * without safety checks. It ensures that the conversion between the specified
 * types is possible at compile time.
 *
 * @tparam From The source type for the conversion.
 * @tparam To The destination type for the conversion.
 */
template <typename From, typename To, typename SFINEA=void>
class ConversionTraitImpl : public ConversionTrait
{
    using helper = ConversionHelper<From, To>;
    static_assert(helper::is_possible, "From must be convertible to To");

public:
    ConversionTraitImpl(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type)) {}

    bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

template <typename From, typename To, typename SFINEA>
bool ConversionTraitImpl<From, To, SFINEA>::is_exact() const noexcept
{
    return helper::is_always_exact;
}

template <typename From, typename To, typename SFINEA>
void ConversionTraitImpl<From, To, SFINEA>::unsafe_convert(void* dst,
    const void* src,
    bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* dst_ptr = static_cast<typename helper::to_ptr>(dst);
    auto* src_ptr = static_cast<typename helper::from_ptr>(src);

    auto result = helper::convert(dst_ptr, src_ptr, exact);
    if (result == ConversionResult::Failed) {
        throw std::runtime_error("conversion failed");
    }
    if (exact && ConversionResult::Inexact == result) {
        RPY_THROW(std::runtime_error,
            "conversion was required to be precise but was not");
    }
}


/**
 * @brief This class is responsible for creating type conversion objects.
 *
 * The ConversionFactory class provides a factory mechanism to generate instances
 * of type conversion objects. It manages the creation process and simplifies
 * the instantiation of converters for various type transformations by delegating
 * the responsibility of object creation to the factory class.
 */
class ConversionFactory
{
public:
    virtual ~ConversionFactory();

    RPY_NO_DISCARD
    virtual std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const = 0;
};

template <typename From, typename To>
class ConversionFactoryImpl : public ConversionFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename From, typename To>
std::unique_ptr<const ConversionTrait> ConversionFactoryImpl<From, To>::make(
    TypePtr from_type,
    TypePtr to_type) const
{
    return std::make_unique<const ConversionTraitImpl<From, To>>(
            std::move(from_type),
            std::move(to_type)
    );
}


template <typename From, typename Map, typename To, typename... Ts>
void build_conversion_from_table(Map& map, meta::TypeList<To, Ts...> list)
{
    using helper = ConversionHelper<From, To>;
    ignore_unused(list);
    if constexpr (!is_same_v<From, To> && helper::is_possible) {
        Hash<string_view> hasher;
        map.emplace(
            hasher(type_id_of<To>),
            std::make_unique<ConversionFactoryImpl<From, To>>()
        );
    }
    if constexpr (sizeof...(Ts) > 0) {
        build_conversion_from_table<From>(map, meta::TypeList<Ts...>{});
    }
}



template <typename To, typename Map, typename From, typename... Ts>
void build_conversion_to_table(Map& map, meta::TypeList<From, Ts...> list)
{
    ignore_unused(list);
    using helper = ConversionHelper<From, To>;
    if constexpr (!is_same_v<From, To> && helper::is_possible) {
        Hash<string_view> hasher;
        map.emplace(
            hasher(type_id_of<From>),
            std::make_unique<ConversionFactoryImpl<From, To>>()
        );
    }
    if constexpr (sizeof...(Ts) > 0) {
        build_conversion_from_table<To>(map, meta::TypeList<Ts...>{});
    }
}






} // namespace rpy::generics::conv


#endif //ROUGHPY_GENERICS_INTERNAL_CONVERSION_IMPL_H
