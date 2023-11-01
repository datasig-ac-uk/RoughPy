

// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_interface.h>
#include <roughpy/scalars/scalar_type.h>

#include "scalar/arithmetic.h"
#include <roughpy/device/types.h>

#include <stdexcept>

using namespace rpy;
using namespace rpy::scalars;

Scalar::Scalar(Scalar::type_pointer type) : integer_for_convenience(0)
{
    if (!type.is_null()) {
        dtl::ScalarContentType mode;
        if (type.is_pointer()) {
            mode = dtl::content_type_of(type);
            p_type_and_content_type = type_pointer(type.get_pointer(), mode);
        } else {
            auto info = type.get_type_info();
            mode = dtl::content_type_of(info);
            if (mode == dtl::ScalarContentType::OwnedPointer) {
                p_type_and_content_type
                        = type_pointer(scalar_type_of(info), mode);
            } else {
                p_type_and_content_type = type_pointer(info, mode);
            }
        }

        if (mode == dtl::ScalarContentType::OwnedPointer) {
            opaque_pointer = type->allocate_single();
        }
    }
}

Scalar::Scalar(const ScalarType* type)
    : p_type_and_content_type(type, dtl::content_type_of(type->type_info())),
      integer_for_convenience(0)
{
    if (p_type_and_content_type.get_enumeration()
        == dtl::ScalarContentType::OwnedPointer) {
        opaque_pointer = type->allocate_single();
    }
}
Scalar::Scalar(devices::TypeInfo info)
    : p_type_and_content_type(info, dtl::content_type_of(info)),
      integer_for_convenience(0)
{
    auto mode = p_type_and_content_type.get_enumeration();
    if (mode == dtl::ScalarContentType::OwnedPointer) {
        p_type_and_content_type = type_pointer(scalar_type_of(info), mode);
        opaque_pointer = p_type_and_content_type->allocate_single();
    }
}
Scalar::Scalar() : p_type_and_content_type(), integer_for_convenience(0) {}

Scalar::Scalar(const Scalar& other) {}
Scalar::Scalar(Scalar&& other) noexcept {}

Scalar::~Scalar()
{
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::OwnedPointer:
            RPY_CHECK(p_type_and_content_type.is_pointer());
            p_type_and_content_type->free_single(opaque_pointer);
            break;
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface: interface.~unique_ptr();
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstTrivialBytes:
        case dtl::ScalarContentType::ConstOpaquePointer: break;
    }
};

Scalar& Scalar::operator=(const Scalar& other)
{
    if (&other != this) {
        if (this->fast_is_zero()) {
            p_type_and_content_type = other.p_type_and_content_type;

            switch (p_type_and_content_type.get_enumeration()) {
                case dtl::ScalarContentType::TrivialBytes:
                case dtl::ScalarContentType::ConstTrivialBytes:
                    std::memcpy(
                            trivial_bytes,
                            other.trivial_bytes,
                            sizeof(interface_pointer_t)
                    );
                    break;
                case dtl::ScalarContentType::OpaquePointer:
                case dtl::ScalarContentType::ConstOpaquePointer:
                    opaque_pointer = other.opaque_pointer;
                    break;
                case dtl::ScalarContentType::OwnedPointer:
                    // TODO: implement copy for owned pointers
                    break;
                case dtl::ScalarContentType::Interface:
                case dtl::ScalarContentType::OwnedInterface:
                    // Copying of interface pointers is disallowed.
                    RPY_THROW(
                            std::runtime_error,
                            "copying of interface pointers "
                            "is not allowed"
                    );
            }
        } else {
            dtl::scalar_convert_copy(
                    mut_pointer(),
                    p_type_and_content_type.get_type_info(),
                    other
            );
        }
    }
    return *this;
}

Scalar& Scalar::operator=(Scalar&& other) noexcept
{
    if (&other != this) {
        this->~Scalar();
        p_type_and_content_type = other.p_type_and_content_type;
        switch (p_type_and_content_type.get_enumeration()) {
            case dtl::ScalarContentType::TrivialBytes:
            case dtl::ScalarContentType::ConstTrivialBytes:
                std::memcpy(
                        trivial_bytes,
                        other.trivial_bytes,
                        sizeof(interface_pointer_t)
                );
                break;
            case dtl::ScalarContentType::OpaquePointer:
            case dtl::ScalarContentType::ConstOpaquePointer:
            case dtl::ScalarContentType::OwnedPointer:
                opaque_pointer = other.opaque_pointer;
                other.opaque_pointer = nullptr;
                break;
            case dtl::ScalarContentType::Interface:
            case dtl::ScalarContentType::OwnedInterface:
                interface = std::move(other.interface);
                other.interface = nullptr;
                break;
        }
    }
    return *this;
}

static inline bool is_pointer_zero(
        const void* ptr,
        const PackedScalarTypePointer<scalars::dtl::ScalarContentType>& p_type
) noexcept
{
    using namespace rpy::devices;
    if (ptr == nullptr) { return true; }

    // const ScalarType* type_ptr = nullptr;

    auto info = (p_type.is_pointer()) ? p_type->type_info()
                                      : p_type.get_type_info();

    switch (info.code) {
        case devices::TypeCode::Int:
            switch (info.bytes) {
                case 1: return *reinterpret_cast<const int8_t*>(ptr) == 0;
                case 2: return *reinterpret_cast<const int16_t*>(ptr) == 0;
                case 4: return *reinterpret_cast<const int32_t*>(ptr) == 0;
                case 8: return *reinterpret_cast<const int64_t*>(ptr) == 0;
            }
            break;
        case devices::TypeCode::UInt:
            switch (info.bytes) {
                case 1: return *reinterpret_cast<const uint8_t*>(ptr) == 0;
                case 2: return *reinterpret_cast<const uint16_t*>(ptr) == 0;
                case 4: return *reinterpret_cast<const uint32_t*>(ptr) == 0;
                case 8: return *reinterpret_cast<const uint64_t*>(ptr) == 0;
            }
            break;
        case devices::TypeCode::Float:
            switch (info.bytes) {
                case 2: return *reinterpret_cast<const half*>(ptr) == 0;
                case 4: return *reinterpret_cast<const float*>(ptr) == 0;
                case 8: return *reinterpret_cast<const double*>(ptr) == 0;
            }
            break;
        case devices::TypeCode::OpaqueHandle: break;
        case devices::TypeCode::BFloat:
            if (info.bytes == 2) {
                return *reinterpret_cast<const bfloat16*>(ptr) == 0;
            }
            break;
        case devices::TypeCode::Complex:
            switch (info.bytes) {
                case 2:
                    return *reinterpret_cast<const half_complex*>(ptr)
                            == half_complex();
                case 4:
                    return *reinterpret_cast<const float_complex*>(ptr)
                            == float_complex();
                case 8:
                    return *reinterpret_cast<const double_complex*>(ptr)
                            == double_complex();
            }
            break;
        case devices::TypeCode::Bool: break;
        case devices::TypeCode::Rational:
        case devices::TypeCode::ArbitraryPrecision: break;
        case devices::TypeCode::ArbitraryPrecisionUInt: break;
        case devices::TypeCode::ArbitraryPrecisionFloat: break;
        case devices::TypeCode::ArbitraryPrecisionComplex: break;
        case devices::TypeCode::ArbitraryPrecisionRational:
            return *reinterpret_cast<const rational_scalar_type*>(ptr) == 0;
        case devices::TypeCode::APRationalPolynomial:
            return reinterpret_cast<const rational_poly_scalar*>(ptr)->empty();
    }

    // Anything else, just return true because we don't support it anyway
    return true;
}

bool Scalar::is_zero() const noexcept
{
    if (fast_is_zero()) { return true; }

    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::ConstTrivialBytes:
            return bit_cast<uintptr_t>(trivial_bytes) == 0;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstOpaquePointer:
        case dtl::ScalarContentType::OwnedPointer:
            return is_pointer_zero(opaque_pointer, p_type_and_content_type);
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface:
            return is_pointer_zero(
                    interface->pointer(),
                    p_type_and_content_type
            );
    }
}

bool Scalar::is_reference() const noexcept
{
    if (fast_is_zero()) { return false; }
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::ConstTrivialBytes:
        case dtl::ScalarContentType::OwnedPointer: return false;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstOpaquePointer:
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface: return true;
    }
}

bool Scalar::is_const() const noexcept
{
    if (fast_is_zero()) { return true; }
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes: return false;
        case dtl::ScalarContentType::ConstTrivialBytes: return true;
        case dtl::ScalarContentType::OpaquePointer: return false;
        case dtl::ScalarContentType::ConstOpaquePointer: return true;
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface: return false;
        case dtl::ScalarContentType::OwnedPointer: return false;
    }
}

const void* Scalar::pointer() const noexcept
{
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::ConstTrivialBytes: return &trivial_bytes;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstOpaquePointer:
        case dtl::ScalarContentType::OwnedPointer: return opaque_pointer;
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface:
            return interface->pointer();
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}
void* Scalar::mut_pointer()
{
    if (fast_is_zero()) {
        RPY_THROW(
                std::runtime_error,
                "cannot get mutable pointer to constant"
                " value zero"
        );
    }
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes: return &trivial_bytes;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::OwnedPointer: return opaque_pointer;
        case dtl::ScalarContentType::ConstTrivialBytes:
        case dtl::ScalarContentType::ConstOpaquePointer:
            RPY_THROW(
                    std::runtime_error,
                    "cannot get mutable pointer to constant value"
            );
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface:
            RPY_THROW(
                    std::runtime_error,
                    "cannot get mutable pointer to special scalar references"
            );
    }

    RPY_UNREACHABLE_RETURN(nullptr);
}
optional<const ScalarType*> Scalar::type() const noexcept
{
    if (p_type_and_content_type.is_pointer()) {
        return p_type_and_content_type.get_pointer();
    }

    auto info = p_type_and_content_type.get_type_info();
    // TODO: Get type from type info?

    return {};
}
devices::TypeInfo Scalar::type_info() const noexcept
{
    if (p_type_and_content_type.is_pointer()) {
        return p_type_and_content_type->type_info();
    }
    return p_type_and_content_type.get_type_info();
}

static void print_trivial_bytes(
        std::ostream& os,
        const byte* bytes,
        PackedScalarTypePointer<scalars::dtl::ScalarContentType> p_type
)
{}

static void print_opaque_pointer(
        std::ostream& os,
        const void* opaque,
        PackedScalarTypePointer<scalars::dtl::ScalarContentType> p_type
)
{}

std::ostream& rpy::scalars::operator<<(std::ostream& os, const Scalar& value)
{
    if (value.fast_is_zero()) {
        os << 0;
        return os;
    }

    switch (value.p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::ConstTrivialBytes:
            print_trivial_bytes(
                    os,
                    value.trivial_bytes,
                    value.p_type_and_content_type
            );
            break;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstOpaquePointer:
        case dtl::ScalarContentType::OwnedPointer:
            print_opaque_pointer(
                    os,
                    value.opaque_pointer,
                    value.p_type_and_content_type
            );
            break;
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface:
            value.interface->print(os);
            break;
    }

    return os;
}
namespace {
//
// template <typename I, typename J>
// inline enable_if_t<is_integral<I>::value && is_integral<J>::value, bool>
// eq_impl(const I& lhs, const J& rhs) noexcept
//{
//    if constexpr (is_signed<I>::value) {
//        if constexpr (is_signed<J>::value) {
//            if constexpr (sizeof(I) < sizeof(J)) {
//                return static_cast<J>(lhs) == rhs;
//            } else {
//                return lhs == static_cast<I>(rhs);
//            }
//        } else {
//            if constexpr (sizeof(I) < sizeof(J)) {
//                return lhs >= 0 && static_cast<J>(lhs) == rhs;
//            } else {
//                using uI = make_unsigned_t<I>;
//                return lhs >= 0 && static_cast<uI>(lhs) ==
//                static_cast<uI>(rhs);
//            }
//        }
//    } else {
//        return eq_impl(rhs, lhs);
//    }
//}
//
// template <typename F, typename G>
// inline enable_if_t<
//        is_floating_point<F>::value && is_floating_point<G>::value,
//        bool>
// eq_impl(const F& lhs, const G& rhs) noexcept
//{
//    if constexpr (sizeof(F) < sizeof(G)) {
//        return static_cast<G>(lhs) == rhs;
//    } else {
//        return lhs == static_cast<F>(rhs);
//    }
//}
//
// template <typename I, typename F>
// inline enable_if_t<is_integral<I>::value && is_floating_point<F>::value,
// bool> eq_impl(const I& lhs, const F& rhs) noexcept
//{
//    constexpr auto mantissa_bits = std::numeric_limits<F>::digits;
//    if constexpr (CHAR_BIT * sizeof(I) <= mantissa_bits) {
//        return static_cast<F>(lhs) == rhs;
//    } else {
//        // Loss of precision here, maybe consider changing this later.
//        return static_cast<F>(lhs) == rhs;
//    }
//}
//
//
// template <typename I>
// inline enable_if_t<!is_same<I, devices::rational_poly_scalar>::value, bool>
// eq_impl(const I& lhs, const devices::rational_scalar_type& rhs) noexcept
//{
//    return static_cast<devices::rational_scalar_type>(lhs) == rhs;
//}
//
// template <typename I>
// inline enable_if_t<!is_same<I, devices::rational_poly_scalar>::value, bool>
// eq_impl(const devices::rational_scalar_type& lhs, const I& rhs) noexcept
//{
//    return lhs == static_cast<devices::rational_scalar_type>(rhs);
//}
//
// template <typename T>
// inline enable_if_t<!is_same<T, devices::rational_poly_scalar>::value, bool>
// eq_impl(const T& lhs, const devices::rational_poly_scalar& rhs) noexcept
//{
//    return static_cast<devices::rational_poly_scalar>(lhs) == rhs;
//}
// template <typename T>
// inline enable_if_t<!is_same<T, devices::rational_poly_scalar>::value, bool>
// eq_impl(const devices::rational_poly_scalar& lhs, const T& rhs) noexcept
//{
//    return lhs == static_cast<devices::rational_poly_scalar>(rhs);
//}
//
// template <typename T>
// inline bool eq_impl(const T& lhs, const T& rhs) noexcept
//{
//    return lhs == rhs;
//}
//
// template <typename T>
// inline bool eq_impl2(const T& lhs, const Scalar& rhs) noexcept
//{
//    auto info = rhs.type_info();
//    switch (info.code) {
//        case devices::TypeCode::Int:
//            switch (info.bytes) {
//                case 1: return eq_impl(lhs, rhs.as_type<int8_t>());
//                case 2: return eq_impl(lhs, rhs.as_type<int16_t>());
//                case 4: return eq_impl(lhs, rhs.as_type<int32_t>());
//                case 8: return eq_impl(lhs, rhs.as_type<int64_t>());
//            }
//            break;
//        case devices::TypeCode::UInt:
//            switch (info.bytes) {
//                case 1: return eq_impl(lhs, rhs.as_type<uint8_t>());
//                case 2: return eq_impl(lhs, rhs.as_type<uint16_t>());
//                case 4: return eq_impl(lhs, rhs.as_type<uint32_t>());
//                case 8: return eq_impl(lhs, rhs.as_type<uint64_t>());
//            }
//            break;
//        case devices::TypeCode::Float:
//            switch (info.bytes) {
//                case 2:
//                    return eq_impl(lhs, float(rhs.as_type<devices::half>()));
//                case 4: return eq_impl(lhs, rhs.as_type<float>());
//                case 8: return eq_impl(lhs, rhs.as_type<double>());
//            }
//            break;
//        case devices::TypeCode::OpaqueHandle: break;
//        case devices::TypeCode::BFloat:
//            if (info.bytes == 2) {
//                return eq_impl(lhs, float(rhs.as_type<devices::bfloat16>()));
//            }
//            break;
//        case devices::TypeCode::Complex: break;
//        case devices::TypeCode::Bool: break;
//        case devices::TypeCode::Rational: break;
//        case devices::TypeCode::ArbitraryPrecision: break;
//        case devices::TypeCode::ArbitraryPrecisionUInt: break;
//        case devices::TypeCode::ArbitraryPrecisionFloat: break;
//        case devices::TypeCode::ArbitraryPrecisionComplex: break;
//        case devices::TypeCode::ArbitraryPrecisionRational:
//            return eq_impl(lhs, rhs.as_type<devices::rational_scalar_type>());
//        case devices::TypeCode::APRationalPolynomial:
//            return eq_impl(lhs, rhs.as_type<devices::rational_poly_scalar>());
//    }
//    return false;
//}
//
}// namespace

bool Scalar::operator==(const Scalar& other) const
{
    if (fast_is_zero()) { return other.is_zero(); }
    if (other.fast_is_zero()) { return is_zero(); }

    return false;
}
