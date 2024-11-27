//
// Created by sam on 27/11/24.
//

#include "polynomial_type.h"

#include <algorithm>
#include <typeinfo>


#include <roughpy/core/construct_inplace.h>
#include <roughpy/core/debug_assertion.h>

#include <roughpy/platform/alloc.h>

#include "indeterminate.h"
#include "monomial.h"
#include "polynomial.h"


using namespace rpy;
using namespace rpy::generics;

void PolynomialType::inc_ref() const noexcept
{
    // do nothing
}

bool PolynomialType::dec_ref() const noexcept
{
    // Do nothing
    return false;
}

intptr_t PolynomialType::ref_count() const noexcept { return 1; }

const std::type_info& PolynomialType::type_info() const noexcept
{
    return typeid(Polynomial);
}

BasicProperties PolynomialType::basic_properties() const noexcept
{
    return {false, false, false, false, false, false, false, false, true,
            false, false};
}

size_t PolynomialType::object_size() const noexcept
{
    return sizeof(Polynomial);
}

string_view PolynomialType::name() const noexcept { return "Polynomial"; }

string_view PolynomialType::id() const noexcept { return "poly"; }

void* PolynomialType::allocate_object() const
{
    auto* ptr = static_cast<Polynomial*>(mem::small_object_alloc(
        sizeof(Polynomial)));
    if (ptr == nullptr) { throw std::bad_alloc(); }
    construct_inplace(ptr);

    return ptr;
}

void PolynomialType::free_object(void* ptr) const
{
    auto* poly = static_cast<Polynomial*>(ptr);
    poly->~Polynomial();
    mem::small_object_free(poly, sizeof(Polynomial));
}

bool PolynomialType::parse_from_string(void* data,
                                       string_view str) const noexcept
{
    return Type::parse_from_string(data, str);
}

void PolynomialType::copy_or_move(void* dst,
                                  const void* src,
                                  size_t count,
                                  bool move) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    auto* dst_poly = static_cast<Polynomial*>(dst);
    const auto* src_poly = static_cast<const Polynomial*>(src);

    if (src_poly == nullptr) {
        for (size_t i = 0; i < count; ++i) { dst_poly[i] = Polynomial(); }
    } else if (move) {
        auto* modifiable_src = const_cast<Polynomial*>(src_poly);
        for (size_t i = 0; i < count; ++i) {
            dst_poly[i] = std::move(modifiable_src[i]);
        }
    } else { std::copy_n(src_poly, count, dst_poly); }
}

void PolynomialType::destroy_range(void* data, size_t count) const
{
    RPY_DBG_ASSERT_NE(data, nullptr);
    auto* poly = static_cast<Polynomial*>(data);
    std::destroy_n(poly, count);
}

std::unique_ptr<const ConversionTrait> PolynomialType::
convert_to(const Type& type) const noexcept { return Type::convert_to(type); }

std::unique_ptr<const ConversionTrait> PolynomialType::
convert_from(const Type& type) const noexcept
{
    return Type::convert_from(type);
}

const BuiltinTrait* PolynomialType::
get_builtin_trait(BuiltinTraitID id) const noexcept
{
    return Type::get_builtin_trait(id);
}

const std::ostream& PolynomialType::display(std::ostream& os,
                                            const void* value) const
{
    if (value == nullptr) { return os << "{ }"; }
    const auto* poly = static_cast<const Polynomial*>(value);
    poly_print(os, *poly);
    return os;
}

hash_t PolynomialType::hash_of(const void* value) const noexcept
{
    if (value == nullptr) { return 0; }
    return hash_value(*static_cast<const Polynomial*>(value));
}

TypePtr PolynomialType::get() noexcept
{
    static PolynomialType type;
    return &type;
}