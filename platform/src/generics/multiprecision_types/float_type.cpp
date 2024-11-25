//
// Created by sammorley on 25/11/24.
//

#include "float_type.h"


#include <gmp.h>
#include <mpfr.h>

using namespace rpy;
using namespace rpy::generics;

MPFloatType::MPFloatType(int precision)
    : m_precision(precision) {}

const std::type_info& MPFloatType::type_info() const noexcept
{
    return typeid(mpfr_t);
}

BasicProperties MPFloatType::basic_properties() const noexcept
{
    return {
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false
    };
}

size_t MPFloatType::object_size() const noexcept { return sizeof(mpfr_ptr); }

void* MPFloatType::allocate_object() const
{
    auto* new_float = static_cast<mpfr_ptr>(mem::aligned_alloc(
        alignof(mpfr_t),
        sizeof(mpfr_t)));

    if (new_float == nullptr) { throw std::bad_alloc(); }
    mpfr_init2(new_float, m_precision);
    mpfr_set_zero(new_float, 1);
    return new_float;
}

void MPFloatType::free_object(void* ptr) const
{
    RPY_DBG_ASSERT_NE(ptr, nullptr);
    mpfr_clear(static_cast<mpfr_ptr>(ptr));
    mem::aligned_free(ptr);
}

bool MPFloatType::parse_from_string(void* data, string_view str) const noexcept
{
    RPY_DBG_ASSERT_NE(data, nullptr);
    string tmp(str);
    return mpfr_set_str(static_cast<mpfr_ptr>(data), tmp.c_str(), 10, MPFR_RNDN)
            != -1;
}

void MPFloatType::copy_or_move(void* dst,
                               const void* src,
                               size_t count,
                               bool move) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);

    if (RPY_UNLIKELY(count == 0)) { return; }

    auto* dst_ptr = static_cast<mpfr_ptr>(dst);
    if (src == nullptr) {
        for (size_t i = 0; i < count; ++i) { mpfr_set_zero(++dst_ptr, 1); }
    } else if (move) {
        auto* src_ptr = static_cast<mpfr_ptr>(const_cast<void*>(src));
        for (size_t i = 0; i < count; ++i, ++src_ptr, ++dst_ptr) {
            mpfr_swap(dst_ptr, src_ptr);
            mpfr_clear(src_ptr);
        }
    } else {
        const auto* src_ptr = static_cast<mpfr_srcptr>(src);
        for (size_t i = 0; i < count; ++i, ++src_ptr, ++dst_ptr) {
            mpfr_set(dst_ptr, src_ptr, MPFR_RNDN);
        }
    }

}

void MPFloatType::destroy_range(void* data, size_t count) const {}

std::unique_ptr<const ConversionTrait> MPFloatType::
convert_to(const Type& type) const noexcept
{
    return RefCountedMiddle<Type>::convert_to(type);
}

std::unique_ptr<const ConversionTrait> MPFloatType::
convert_from(const Type& type) const noexcept
{
    return RefCountedMiddle<Type>::convert_from(type);
}

const BuiltinTrait* MPFloatType::
get_builtin_trait(BuiltinTraitID id) const noexcept
{
    return RefCountedMiddle<Type>::get_builtin_trait(id);
}

const std::ostream& MPFloatType::display(std::ostream& os,
                                         const void* value) const {}

hash_t MPFloatType::hash_of(const void* value) const noexcept
{
    return RefCountedMiddle<Type>::hash_of(value);
}