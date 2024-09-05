//
// Created by sam on 4/26/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H
#define ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H

#include "algorithms.h"

#include <roughpy/core/errors.h>
#include <roughpy/core/hash.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/host_device.h>
#include <roughpy/devices/type.h>

#include <cmath>

namespace rpy {
namespace devices {

namespace dtl {
template <typename T>
struct IDAndNameOfFType;

}// namespace dtl

namespace math_fn_impls {

using namespace std;

template <typename T>
void abs_fn(void* out, const void* in)
{
    if constexpr (is_signed_v<T>) {
        *static_cast<T*>(out) = abs(*static_cast<const T*>(in));
    } else {
        *static_cast<T*>(out) = *static_cast<const T*>(in);
    }
}
template <typename T>
void real_fn(void* out, const void* in)
{
    *static_cast<T*>(out) = *static_cast<const T*>(in);
}
template <typename T>
void pow_fn(void* out, const void* in, unsigned power) noexcept
{
    *static_cast<T*>(out) = pow(*static_cast<const T*>(in), power);
}

template <typename T>
void sqrt_fn(void* out, const void* in)
{
    *static_cast<T*>(out) = sqrt(*static_cast<const T*>(in));
}
template <typename T>
void exp_fn(void* out, const void* in)
{
    *static_cast<T*>(out) = exp(*static_cast<const T*>(in));
}
template <typename T>
void log_fn(void* out, const void* in)
{
    *static_cast<T*>(out) = log(*static_cast<const T*>(in));
}

}// namespace math_fn_impls

/**
 * @class FundamentalType
 * @brief A class that represents a fundamental data type.
 *
 * The FundamentalType class is a subclass of the Type class and represents a
 * fundamental data type. It provides methods to get the unique ID and name of
 * the type, as well as the type information and traits.
 */
template <typename T>
class RPY_LOCAL FundamentalType : public Type
{

    static hash_t hash_function(const void* value_ptr)
    {
        Hash<T> hash;
        return hash(*static_cast<const T*>(value_ptr));
    }

public:
    FundamentalType(string_view id, string_view name)
        : Type(id, name, devices::type_info<T>(), devices::traits_of<T>())
    {
        // #ifndef RPY_NO_RTTI
        //         register_type(typeid(T), this);
        // #endif
        const auto device = get_host_device();
        device->register_algorithm_drivers<HostDriversImpl, T, T>(this, this);

        auto& num_traits = setup_num_traits();

        num_traits.rational_type = this;
        num_traits.real_type = this;
        num_traits.imag_type = nullptr;

        num_traits.abs = math_fn_impls::abs_fn<T>;
        num_traits.real = math_fn_impls::real_fn<T>;

        if RPY_IF_CONSTEXPR (is_floating_point_v<T>) {
            num_traits.pow = math_fn_impls::pow_fn<T>;
            num_traits.sqrt = math_fn_impls::sqrt_fn<T>;
            num_traits.exp = math_fn_impls::exp_fn<T>;
            num_traits.log = math_fn_impls::log_fn<T>;
        }

        set_hash_fn(hash_function);
    }

    /**
     * @brief Returns a pointer to the static instance of FundamentalType<T>
     * with the specified type T.
     *
     * @tparam T The type for which the FundamentalType instance is returned.
     *
     * @return A pointer to the static instance of FundamentalType<T>.
     *
     * @note The returned pointer is valid throughout the program execution and
     * should not be deleted.
     * @note The returned pointer is guaranteed to be non-null.
     * @note The returned pointer may be used to access the ID and name of the
     * FundamentalType<T>.
     */
    RPY_NO_DISCARD static const FundamentalType* get() noexcept;

    void display(std::ostream& os, const void* ptr) const override;

    RPY_NO_DISCARD ConstReference zero() const override;
    RPY_NO_DISCARD ConstReference one() const override;
    RPY_NO_DISCARD ConstReference mone() const override;

    void copy(void* dst, const void* src, dimn_t count) const override;
    void move(void* dst, void* src, dimn_t count) const override;
};

template <typename T>
const FundamentalType<T>* FundamentalType<T>::get() noexcept
{
    using IDName = dtl::IDAndNameOfFType<T>;
    static const Rc<const FundamentalType> type(
            new FundamentalType(IDName::id, IDName::name)
    );
    return &*type;
}

template <typename T>
void FundamentalType<T>::display(std::ostream& os, const void* ptr) const
{
    os << *static_cast<const T*>(ptr);
}

template <typename T>
ConstReference FundamentalType<T>::zero() const
{
    static constexpr T zero{};
    return ConstReference{this, &zero};
}
template <typename T>
ConstReference FundamentalType<T>::one() const
{
    static constexpr T one{1};
    return ConstReference{this, &one};
}
template <typename T>
ConstReference FundamentalType<T>::mone() const
{
    if constexpr (is_signed_v<T>) {
        static constexpr T mone{-1};
        return ConstReference{this, &mone};
    } else {
        return Type::mone();
    }
}

template <typename ThisT>
void register_all_supports()
{
    const auto* tp = FundamentalType<ThisT>::get();
    dtl::register_type_support<ThisT>(tp, dtl::FundamentalTypesList());
}

template <typename T>
void FundamentalType<T>::copy(void* dst, const void* src, dimn_t count) const
{
    auto* optr = static_cast<T*>(dst);
    const auto* sprt = static_cast<const T*>(src);
    for (dimn_t i = 0; i < count; ++i) { optr[i] = sprt[i]; }
}
template <typename T>
void FundamentalType<T>::move(void* dst, void* src, dimn_t count) const
{
    auto* optr = static_cast<T*>(dst);
    auto* sptr = static_cast<T*>(src);
    for (dimn_t i = 0; i < count; ++i) { optr[i] = std::move(sptr[i]); }
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H
