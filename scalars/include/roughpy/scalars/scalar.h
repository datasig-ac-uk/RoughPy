// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_SCALARS_SCALAR_H_
#define ROUGHPY_SCALARS_SCALAR_H_

#include "scalar_interface.h"
#include "scalar_type.h"
#include "scalars_fwd.h"
#include <roughpy/devices/kernel_arg.h>

#include <roughpy/core/alloc.h>
#include <roughpy/core/container/vector.h>
#include <roughpy/core/helpers.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include <cereal/types/vector.hpp>

namespace rpy {
namespace scalars {

namespace dtl {

/**
 * @brief Discriminator for internal storage models of the Scalar type.
 */
enum class ScalarContentType : uint8_t
{
    /// Data held in internal array of bytes
    TrivialBytes = 0,
    /// Data is a pointer to an external value
    OpaquePointer = 1,
    /// Data is const
    IsConst = 2,
    // This one exists just to keep the switch statements happy.
    // In reality, all TrivialBytes scalars are "const" in the sense that
    // mutating them shouldn't have external effects.
    /// Data is trivial bytes and const
    ConstTrivialBytes = IsConst,
    /// Data is a const pointer to an external value
    ConstOpaquePointer = OpaquePointer | IsConst,
    // We leave open the possibility of non-owning interface pointers in the
    // future, but for now all interface pointers are owned.
    /// Data is a pointer to a (mutable) polymorphic interface type
    Interface = 4,
    /// Data is a pointer to an external value that is owned by the scalar
    OwnedPointer = 5,
    /// Data is a pointer to a (mutable) polymorphic interface type that is
    /// owned by the Scalar
    OwnedInterface = 6
};

/**
 * @brief Convert a scalar and copy it to the destination.
 *
 * @param dst Pointer to the destination memory space.
 * @param dst_type The storage model of the destination scalar.
 * @param src The source scalar to be converted and copied.
 *
 * @return True if the conversion and copy was successful, false otherwise.
 *
 * @note This function internally determines the type of the source scalar and
 * performs the conversion to the destination scalar type. The converted scalar
 * is then copied to the destination memory space pointed by `dst`. The function
 * returns true if the conversion and copy were successful. Otherwise, it
 * returns false.
 *
 * @warning The caller of this function is responsible for allocating memory for
 * the destination scalar and ensuring that it has enough memory to hold the
 * converted scalar.
 */
ROUGHPY_SCALARS_EXPORT bool scalar_convert_copy(
        void* dst,
        PackedScalarType dst_type,
        const Scalar& src
) noexcept;

/**
 * @brief Converts and copies scalar values from the source buffer to the
 * destination buffer.
 *
 * @param dst The pointer to the destination buffer where the converted scalar
 * values will be copied to.
 * @param dst_type The internal storage model of the destination buffer scalar
 * type.
 * @param src The pointer to the source buffer where the scalar values are
 * located.
 * @param src_type The internal storage model of the source buffer scalar type.
 * @param count The number of scalar values to convert and copy. It defaults
 * to 1.
 *
 * @return Returns true if the conversion and copying operation is successful;
 * otherwise, returns false.
 */
ROUGHPY_SCALARS_EXPORT
bool scalar_convert_copy(
        void* dst,
        PackedScalarType dst_type,
        const void* src,
        PackedScalarType src_type,
        dimn_t count = 1
) noexcept;

/**
 * @brief Determines the content type of a given TypeInfo.
 *
 * This function determines the content type of a given TypeInfo. The content
 * type indicates how the data is stored internally.
 *
 * @param info The TypeInfo for which the content type is to be determined.
 *
 * @return The content type of the TypeInfo.
 *         - TrivialBytes: If the TypeInfo represents a scalar of Int, UInt,
 * Float, or BFloat type with size less than or equal to sizeof(uintptr_t).
 *         - OwnedPointer: If the TypeInfo represents a scalar of Int, UInt,
 * Float, or BFloat type with size greater than sizeof(uintptr_t).
 *         - OpaquePointer: If the TypeInfo represents a scalar of OpaqueHandle
 * type.
 *         - TrivialBytes: If the TypeInfo represents a scalar of Complex type
 * with size less than or equal to sizeof(uintptr_t).
 *         - OwnedPointer: If the TypeInfo represents a scalar of Complex type
 * with size greater than sizeof(uintptr_t).
 *         - OwnedPointer: If the TypeInfo represents a scalar of
 * ArbitraryPrecisionRational or APRationalPolynomial type.
 *         - OpaquePointer: If none of the above conditions are met.
 */
inline ScalarContentType content_type_of(devices::TypeInfo info) noexcept
{
    switch (info.code) {
        case devices::TypeCode::Int:
        case devices::TypeCode::UInt:
        case devices::TypeCode::Float:
        case devices::TypeCode::BFloat:
            if (info.bytes <= sizeof(uintptr_t)) {
                return ScalarContentType::TrivialBytes;
            } else {
                return ScalarContentType::OwnedPointer;
            }
        case devices::TypeCode::OpaqueHandle: break;
        case devices::TypeCode::Complex:
            if (2 * info.bytes <= sizeof(uintptr_t)) {
                return ScalarContentType::TrivialBytes;
            } else {
                return ScalarContentType::OwnedPointer;
            }
        case devices::TypeCode::Bool: break;
        case devices::TypeCode::Rational: break;
        case devices::TypeCode::ArbitraryPrecision: break;
        case devices::TypeCode::ArbitraryPrecisionUInt: break;
        case devices::TypeCode::ArbitraryPrecisionFloat: break;
        case devices::TypeCode::ArbitraryPrecisionComplex: break;
        case devices::TypeCode::ArbitraryPrecisionRational:
        case devices::TypeCode::APRationalPolynomial:
            return ScalarContentType::OwnedPointer;
    }
    return ScalarContentType::OpaquePointer;
}

inline ScalarContentType
content_type_of(PackedScalarTypePointer<ScalarContentType> ptype) noexcept
{
    return content_type_of(ptype.get_type_info());
}

template <typename T>
struct can_be_scalar
    : conditional_t<!is_base_of_v<Scalar, T>, std::true_type, std::false_type> {
};

template <typename T>
struct can_be_scalar<std::unique_ptr<T>> : std::false_type {
};

}// namespace dtl

class Scalar : public devices::Value
{
public:
    using Value::Value;

    RPY_NO_DISCARD Scalar reciprocal() const
    {
        Scalar result(type(), 1);
        result /= *this;
        return result;
    }
};

/**
 * @class Scalar
 * @brief A class representing a scalar value with different storage models.
 *
 * The Scalar class provides a flexible representation for scalar values with
 * different storage models. It supports various types of scalar values,
 * including trivial bytes, opaque pointers, owned pointers, and interfaces. The
 * class is designed to be efficient and lightweight, while still providing
 * flexibility and ease of use.
 *
 */
// class ROUGHPY_SCALARS_EXPORT Scalar
// {
//     using interface_pointer_t = ScalarInterface*;
//     //    static_assert(sizeof(interface_pointer_t) == sizeof(uintptr_t),
//     ""); using type_pointer =
//     PackedScalarTypePointer<dtl::ScalarContentType>; type_pointer
//     p_type_and_content_type;
//
//     union
//     {
//         uintptr_t integer_for_convenience;
//         byte trivial_bytes[sizeof(uintptr_t)];
//         interface_pointer_t interface_ptr;
//         void* opaque_pointer;
//     };
//
//     void allocate_data();
//
//     void copy_from_opaque_pointer(devices::TypeInfo info, const void* src);
//
// protected:
//     type_pointer packed_type() const noexcept
//     {
//         return p_type_and_content_type;
//     }
//
// public:
//     Scalar();
//
//     explicit Scalar(PackedScalarType type);
//     explicit Scalar(type_pointer type);
//     explicit Scalar(const ScalarType* type);
//     explicit Scalar(devices::TypeInfo info);
//
//     //    template <
//     //            typename T,
//     //            typename = enable_if_t<
//     //                    !is_reference<T>::value &&
//     //                    is_standard_layout<T>::value
//     //                    && is_trivially_copyable<T>::value
//     //                    && is_trivially_destructible<T>::value
//     //                    && (sizeof(T) <= sizeof(void*))>>
//     //    explicit Scalar(T value)
//     //        : p_type_and_content_type(
//     //                devices::type_info<T>(),
//     //                dtl::ScalarContentType::TrivialBytes
//     //        ),
//     //          integer_for_convenience(0)
//     //    {
//     //        std::memcpy(&trivial_bytes, &value, sizeof(value));
//     //    }
//
//     template <typename T, typename =
//     enable_if_t<dtl::can_be_scalar<T>::value>> explicit Scalar(const T&
//     value)
//         : p_type_and_content_type(
//                   devices::type_info<remove_cv_t<T>>(),
//                   dtl::ScalarContentType::TrivialBytes
//           ),
//           integer_for_convenience(0)
//     {
//         if constexpr (is_standard_layout_v<T> && is_trivially_copyable_v<T>
//         && is_trivially_destructible_v<T> && sizeof(T) <= sizeof(void*)) {
//             std::memcpy(trivial_bytes, &value, sizeof(T));
//         } else {
//             allocate_data();
//             construct_inplace(static_cast<T*>(opaque_pointer), value);
//         }
//         RPY_DBG_ASSERT(!p_type_and_content_type.is_null());
//     }
//
//     template <
//             typename T,
//             typename = enable_if_t<
//                     !is_pointer_v<T> && is_standard_layout_v<T>
//                     && is_trivially_copyable_v<T>
//                     && is_trivially_destructible_v<T>>>
//     explicit Scalar(const ScalarType* type, T&& value)
//         : p_type_and_content_type(type,
//         dtl::ScalarContentType::TrivialBytes),
//           integer_for_convenience(0)
//     {
//         dtl::scalar_convert_copy(
//                 trivial_bytes,
//                 type_info(),
//                 &value,
//                 devices::type_info<remove_cv_ref_t<T>>()
//         );
//     }
//
//     template <
//             typename T,
//             enable_if_t<
//                     !is_pointer_v<T>
//                             && (!is_standard_layout_v<T>
//                                 || !is_trivially_copyable_v<T>
//                                 || !is_trivially_destructible_v<T>),
//                     int>
//             = 0>
//     explicit Scalar(const ScalarType* type, T&& value)
//         : p_type_and_content_type(type,
//         dtl::ScalarContentType::OwnedPointer),
//           integer_for_convenience(0)
//     {
//         allocate_data();
//         auto this_info = type_info();
//         auto value_info = devices::type_info<remove_cv_ref_t<T>>();
//         if (this_info == value_info) {
//             construct_inplace(
//                     static_cast<remove_cv_ref_t<T>*>(opaque_pointer),
//                     std::forward<T>(value)
//             );
//         } else {
//             dtl::scalar_convert_copy(
//                     opaque_pointer,
//                     this_info,
//                     &value,
//                     value_info,
//                     1
//             );
//         }
//     }
//
//     explicit Scalar(PackedScalarType info, void* ptr);
//     explicit Scalar(PackedScalarType info, const void* ptr);
//
//     template <
//             typename I,
//             typename = enable_if_t<is_base_of_v<ScalarInterface, I>>>
//     explicit Scalar(std::unique_ptr<I>&& iface)
//         : p_type_and_content_type(
//                   iface->type(),
//                   dtl::ScalarContentType::OwnedInterface
//           ),
//           interface_ptr(std::move(iface).release())
//     {}
//
//     explicit Scalar(PackedScalarType info, int64_t num, int64_t denom);
//
//     Scalar(const Scalar& other);
//     Scalar(Scalar&& other) noexcept;
//
//     ~Scalar();
//
//     /**
//      * @brief Assigns a value to the Scalar object.
//      *
//      * This method assigns the given value to the Scalar object. The assigned
//      * value must be of the same type as the Scalar object. If the value is
//      of a
//      * different type, a conversion will be performed using the
//      * `scalar_convert_copy` function.
//      *
//      * @param value The value to assign to the Scalar object.
//      *
//      */
//     template <typename T>
//     void assign(const T& value)
//     {
//         dtl::scalar_convert_copy(
//                 mut_pointer(),
//                 type_info(),
//                 &value,
//                 devices::type_info<T>()
//         );
//     }
//
//     /*
//      * ********* WARNING **********
//      * The assignment has different semantics than usual. Rather than simply
//      * changing the whole of *this to match the incoming value, the incoming
//      * value is converted to the current type of *this;
//      *
//      * The move/copy assignment operator have the usual semantics if they are
//      * empty values, to aid the compiler when the return value is a Scalar.
//      *
//      */
//     template <typename T>
//     enable_if_t<!is_base_of_v<Scalar, T>, Scalar&> operator=(const T& value
//     )
//     {
//         if (p_type_and_content_type.is_null()) {
//             construct_inplace(this, value);
//         } else {
//             switch (p_type_and_content_type.get_enumeration()) {
//                 case dtl::ScalarContentType::TrivialBytes:
//                 case dtl::ScalarContentType::ConstTrivialBytes:
//                     // We're actually not paying attention to "const" in
//                     trivial
//                     // bytes, since the value is owned by the Scalar.
//                     if (!dtl::scalar_convert_copy(
//                                 trivial_bytes,
//                                 type_info(),
//                                 &value,
//                                 devices::type_info<remove_cv_t<T>>()
//                         )) {
//                         RPY_THROW(std::runtime_error, "assignment failed");
//                     }
//                     break;
//                 case dtl::ScalarContentType::OpaquePointer:
//                 case dtl::ScalarContentType ::OwnedPointer:
//                     if (!dtl::scalar_convert_copy(
//                                 opaque_pointer,
//                                 type_info_from(p_type_and_content_type),
//                                 &value,
//                                 devices::type_info<remove_cv_t<T>>()
//                         )) {
//                         RPY_THROW(std::runtime_error, "assignment failed");
//                     }
//                     break;
//                 case dtl::ScalarContentType::ConstOpaquePointer:
//                     RPY_THROW(
//                             std::runtime_error,
//                             "attempting to write to a const value"
//                     );
//                 case dtl::ScalarContentType::Interface:
//                 case dtl::ScalarContentType::OwnedInterface:
//                     interface_ptr->set_value(
//                             Scalar(devices::type_info<remove_cv_t<T>>(),
//                             &value)
//                     );
//                     break;
//             }
//         }
//         return *this;
//     }
//
//     Scalar& operator=(const Scalar& other);
//     Scalar& operator=(Scalar&& other) noexcept;
//
//     /**
//      * @brief Perform a quick test to see if this is zero.
//      * @return true if this is trivially equivalent to zero, false otherwise.
//      */
//     bool fast_is_zero() const noexcept
//     {
//         return p_type_and_content_type.is_null()
//                 || integer_for_convenience == 0;
//     }
//
//     /**
//      * @brief Fully check if this is zero.
//      * @return true if this is equivalent to zero, false otherwise.
//      */
//     bool is_zero() const noexcept;
//
//     /**
//      * @brief Check if the underlying data is a local value.
//      * @return true if the underlying data is local, and false otherwise.
//      *
//      * We say that the underlying data is local if either the value has a
//      * trivial type and is held inline inside the Scalar, or if it is a
//      * pointer to a value allocated by this.
//      */
//     bool is_reference() const noexcept;
//
//     /**
//      * @brief Check if the internal data is immutable.
//      * @return true if the underlying data is immutable, and false if it can
//      * be safely mutated in-place.
//      */
//     bool is_const() const noexcept;
//
//     /**
//      * @brief Get an immutable pointer to the internal data representation.
//      * @return const pointer to the underlying data.
//      *
//      * This method cannot fail, since all Scalar values should contain some
//      * value. This pointer should never be null, but it might not have
//      * semantic meaning.
//      */
//     const void* pointer() const noexcept;
//
//     /**
//      * @brief Get a mutable pointer to the internal data representation.
//      * @return mutable pointer to the underlying data
//      *
//      * This method will throw a runtime_error if the underlying data is not
//      * able to be mutated in-place.
//      */
//     void* mut_pointer();
//
//     /**
//      * @brief Determines whether the Scalar object is mutable or not.
//      *
//      * @return    true if the Scalar object is mutable (non-const), false
//      * otherwise.
//      *
//      * @note The mutability of a Scalar object is determined by the value of
//      its
//      * internal storage model, namely the ScalarContentType. If the
//      * ScalarContentType is set to IsConst, then the object is considered
//      * immutable (const). If the ScalarContentType is not set to IsConst,
//      then
//      * the object is considered mutable (non-const). This method returns true
//      * for mutable objects and false for immutable objects.
//      *
//      * @note This method does not modify the internal state of the Scalar
//      * object.
//      *
//      * @see ScalarContentType
//      */
//     bool is_mutable() const noexcept;
//
//     /**
//      * @brief Get a pointer to the scalar type representing this value.
//      *
//      * @return Packed scalar type object
//      */
//     PackedScalarType type() const noexcept
//     {
//         return static_cast<PackedScalarType>(p_type_and_content_type);
//     }
//
//     /**
//      * @brief Get the type info associated with this.
//      * @return type info of this.
//      */
//     devices::TypeInfo type_info() const noexcept;
//
//     /**
//      * @brief Changes the type of the Scalar object.
//      *
//      * This method allows changing the type of the Scalar object to a new
//      type
//      * specified by the provided new_type_info. The method first creates a
//      * temporary Scalar object, tmp, with the new_type_info. Then, it
//      converts
//      * the content of the original Scalar object to the new type using
//      * dtl::scalar_convert_copy method. After the conversion, it destroys the
//      * original Scalar object using the destructor and constructs a new
//      Scalar
//      * object of the new type in its place.
//      *
//      * @param new_type_info The PackedScalarType representing the new type
//      * information.
//      *
//      */
//     void change_type(PackedScalarType new_type_info)
//     {
//         Scalar tmp(new_type_info);
//         dtl::scalar_convert_copy(tmp.mut_pointer(), new_type_info, *this);
//         this->~Scalar();
//         construct_inplace(this, std::move(tmp));
//     }
//
//     friend ROUGHPY_SCALARS_EXPORT std::ostream&
//     operator<<(std::ostream& os, const Scalar& value);
//
//     /**
//      * @brief Cast the scalar to a concrete type T.
//      * @tparam T Type to cast
//      * @return const reference to type
//      *
//      * No type checking is done here, this should only be called if *this is
//      * known to hold a value of type T. Any other usage results in undefined
//      * behaviour.
//      */
//     template <typename T>
//     const T& as_type() const noexcept;
//
//     bool operator==(const Scalar& other) const;
//
//     bool operator!=(const Scalar& other) const { return !(operator==(other));
//     }
//
//     Scalar operator-() const;
//
//     Scalar& operator+=(const Scalar& other);
//     Scalar& operator-=(const Scalar& other);
//     Scalar& operator*=(const Scalar& other);
//     Scalar& operator/=(const Scalar& other);
//
//     /**
//      * @brief Returns the reciprocal value of the Scalar.
//      *
//      * This method calculates the reciprocal value of the Scalar and returns
//      it.
//      * The reciprocal of a number is defined as 1 divided by the number.
//      *
//      * @note This method does not modify the original Scalar value.
//      *
//      * @return The reciprocal value of the Scalar.
//      */
//     RPY_NO_DISCARD Scalar reciprocal() const;
//
//     RPY_SERIAL_SAVE_FN();
//     RPY_SERIAL_LOAD_FN();
//
// private:
//     containers::Vec<byte> to_raw_bytes() const;
//     void from_raw_bytes(PackedScalarType info, Slice<byte> bytes);
// };
//
// ROUGHPY_SCALARS_EXPORT
// std::ostream& operator<<(std::ostream&, const Scalar&);
//
// RPY_SERIAL_LOAD_FN_IMPL(Scalar)
// {
//     devices::TypeInfo type_info;
//     // RPY_SERIAL_SERIALIZE_VAL(type_info);
//     containers::Vec<byte> raw_bytes;
//     RPY_SERIAL_SERIALIZE_VAL(raw_bytes);
//     // from_raw_bytes(type_info, raw_bytes);
// }
//
// RPY_SERIAL_SAVE_FN_IMPL(Scalar)
// {
//     // RPY_SERIAL_SERIALIZE_NVP("type_info", type_info());
//     RPY_SERIAL_SERIALIZE_NVP("raw_bytes", to_raw_bytes());
// }
// //
// // #ifdef RPY_COMPILING_SCALARS
// // RPY_SERIAL_EXTERN_SAVE_CLS_BUILD(Scalar)
// // RPY_SERIAL_EXTERN_LOAD_CLS_BUILD(Scalar)
// // #else
// // RPY_SERIAL_EXTERN_SAVE_CLS_IMP(Scalar)
// // RPY_SERIAL_EXTERN_LOAD_CLS_IMP(Scalar)
// // #endif
//
// template <typename T>
// const T& Scalar::as_type() const noexcept
// {
//     switch (p_type_and_content_type.get_enumeration()) {
//         case dtl::ScalarContentType::TrivialBytes:
//         case dtl::ScalarContentType::ConstTrivialBytes:
//             return *reinterpret_cast<const T*>(trivial_bytes);
//         case dtl::ScalarContentType::ConstOpaquePointer:
//         case dtl::ScalarContentType::OpaquePointer:
//         case dtl::ScalarContentType::OwnedPointer:
//             return *static_cast<const T*>(opaque_pointer);
//         case dtl::ScalarContentType::Interface:
//         case dtl::ScalarContentType::OwnedInterface:
//             return *static_cast<const T*>(interface_ptr->pointer());
//     }
//     RPY_UNREACHABLE_RETURN(*((const T*) &trivial_bytes));
// }
//
// inline Scalar operator+(const Scalar& lhs, const Scalar& rhs)
// {
//     Scalar result(lhs);
//     RPY_DBG_ASSERT(result.is_mutable());
//     result += rhs;
//     return result;
// }
// inline Scalar operator-(const Scalar& lhs, const Scalar& rhs)
// {
//     Scalar result(lhs);
//     RPY_DBG_ASSERT(result.is_mutable());
//     result -= rhs;
//     return result;
// }
// inline Scalar operator*(const Scalar& lhs, const Scalar& rhs)
// {
//     Scalar result(lhs);
//     RPY_DBG_ASSERT(result.is_mutable());
//     result *= rhs;
//     return result;
// }
// inline Scalar operator/(const Scalar& lhs, const Scalar& rhs)
// {
//     Scalar result(lhs);
//     RPY_DBG_ASSERT(result.is_mutable());
//     result /= rhs;
//     return result;
// }
//
// template <typename T>
// RPY_NO_DISCARD T scalar_cast(const Scalar& value)
// {
//     T result{};
//
//     // If the scalar is trivial zero, no need to do anything special.
//     if (!value.fast_is_zero()) {
//         const auto target_info = devices::type_info<T>();
//
//         if (!dtl::scalar_convert_copy(&result, target_info, value)) {
//             // Conversion failed for now just throw an exception.
//             RPY_THROW(std::runtime_error, "unable to cast");
//         }
//     }
//     return result;
// }
//
// RPY_NO_DISCARD inline devices::KernelArgument to_kernel_arg(const Scalar& arg
// ) noexcept
// {
//     return {arg.type_info(), arg.pointer()};
// }
//
// RPY_NO_DISCARD inline devices::KernelArgument to_kernel_arg(Scalar& arg
// ) noexcept
// {
//     return {arg.type_info(),
//             arg.is_mutable() ? arg.mut_pointer() : arg.pointer()};
// }

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_H_
