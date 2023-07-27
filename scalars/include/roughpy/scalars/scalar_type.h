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

#ifndef ROUGHPY_SCALARS_SCALAR_TYPE_H_
#define ROUGHPY_SCALARS_SCALAR_TYPE_H_

#include "scalars_fwd.h"

#include <roughpy/core/alloc.h>
#include <roughpy/core/slice.h>

#include <functional>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace rpy {
namespace scalars {

namespace dtl {
template <typename T>
struct type_id_of_impl;

}

template <typename T>
inline const string& type_id_of() noexcept;

struct RingCharacteristics {
    unsigned is_field : 1;
    unsigned is_ordered : 1;
    unsigned has_sqrt : 1;
};

class RPY_EXPORT ScalarType
{
    ScalarTypeInfo m_info;
    RingCharacteristics m_characteristics RPY_UNUSED_VAR;

protected:
    explicit ScalarType(ScalarTypeInfo info) : m_info(std::move(info)) {}

public:
    template <typename T>
    RPY_NO_DISCARD inline static const ScalarType* of();

    template <typename T>
    RPY_NO_DISCARD inline static const ScalarType*
    of(const ScalarDeviceInfo& device);

    /*
     * ScalarTypes objects should be unique for each configuration,
     * and should only ever be accessed via a pointer. The deleted
     * copy and move constructors reflect this fact.
     */

    ScalarType(const ScalarType&) = delete;
    ScalarType(ScalarType&&) noexcept = delete;
    ScalarType& operator=(const ScalarType&) = delete;
    ScalarType& operator=(ScalarType&&) noexcept = delete;

    virtual ~ScalarType() = default;

    /**
     * @brief Get the most appropriate scalar type for type id
     * @param id Id to query
     * @return const pointer to appropriate scalar type
     */
    RPY_NO_DISCARD static const ScalarType* for_id(const string& id);

    /**
     * @brief Get the most appropriate scalar type for type info
     * @param details Basic description of scalar type
     * @param device Type and ID of the device for scalars.
     * @return const pointer to appropriate scalar type
     */
    RPY_NO_DISCARD static const ScalarType* from_type_details(
            const BasicScalarInfo& details, const ScalarDeviceInfo& device
    );

    /**
     * @brief Get the unique internal ID string for this type
     * @return const reference to the ID string.
     */
    RPY_NO_DISCARD const string& id() const noexcept { return m_info.id; }

    /**
     * @brief Get the extended scalar type information for this type.
     * @return ScalarTypeInfo struct containing the information.
     */
    RPY_NO_DISCARD const ScalarTypeInfo& info() const noexcept
    {
        return m_info;
    }

    /**
     * @brief Get the size of a single scalar in bytes
     * @return number of bytes.
     */
    RPY_NO_DISCARD int itemsize() const noexcept { return m_info.n_bytes; }

    /**
     * @brief Get the rational type associated with this scalar type
     * @return pointer to the rational type
     */
    RPY_NO_DISCARD virtual const ScalarType* rational_type() const noexcept;

    /**
     * @brief Get the type of this scalar situated on host (CPU)
     * @return pointer to the host type
     */
    RPY_NO_DISCARD virtual const ScalarType* host_type() const noexcept;

    /**
     * @brief Create a new scalar from numerator and denominator
     * @param numerator Integer numerator of result
     * @param denominator Integer denominator of result
     * @return new Scalar with value numerator/denominator (in the appropriate
     * type)
     */
    RPY_NO_DISCARD virtual Scalar
    from(long long numerator, long long denominator) const;

    /**
     * @brief Allocate new scalars in memory
     * @param count Number of scalars to allocate space
     * @return ScalarPointer pointing to the newly allocated raw memory.
     */
    RPY_NO_DISCARD virtual ScalarPointer allocate(std::size_t count) const = 0;

    /**
     * @brief Free a previously allocated block of memory
     * @param pointer ScalarPointer pointing to the beginning of the allocated
     * block
     * @param count Number of scalars to be freed.
     */
    virtual void free(ScalarPointer pointer, std::size_t count) const = 0;

    /**
     * @brief Swap the values at two scalar locations
     * @param lhs Pointer to left hand scalar
     * @param rhs Pointer to right hand scalar
     */
    virtual void swap(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    virtual void
    convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count) const
            = 0;

    /**
     * @brief Copy count scalars from in to out, converting as necessary
     * @param out raw pointer to output
     * @param in raw pointer to input
     * @param count number of scalars to copy
     * @param info BasicScalarInfo information about the input scalar type
     */
    virtual void convert_copy(
            void* out, const void* in, std::size_t count, BasicScalarInfo info
    ) const = 0;

    /**
     * @brief Copy count scalars from in to out, converting as necessary
     * @param out raw pointer to destination
     * @param in ScalarPointer to source data
     * @param count number of scalars to copy
     */
    virtual void
    convert_copy(void* out, ScalarPointer in, std::size_t count) const
            = 0;

    /**
     * @brief Copy count scalars from in to out, converting as necessary
     * @param out ScalarPointer to destination data
     * @param in raw pointer to source data
     * @param count number of scalars to copy
     * @param id ID of scalar type for source data
     */
    virtual void convert_copy(
            ScalarPointer out, const void* in, std::size_t count,
            const string& id
    ) const = 0;

    /**
     * @brief
     * @param out
     * @param in
     * @param count
     * @param id
     */
    virtual void convert_fill(
            ScalarPointer out, ScalarPointer in, dimn_t count, const string& id
    ) const;

    /**
     * @brief Parse a string into this scalar type
     * @param str string to be parsed
     * @return new Scalar containing parsed value
     */
    RPY_NO_DISCARD virtual Scalar parse(string_view str) const;

    /**
     * @brief Get the scalar whose value is one
     * @return new Scalar object
     */
    RPY_NO_DISCARD virtual Scalar one() const;

    /**
     * @brief Get the scalar whose value is minus one
     * @return new Scalar object
     */
    RPY_NO_DISCARD virtual Scalar mone() const;

    /**
     * @brief Get the scalar whose value is zero
     * @return new Scalar object
     */
    RPY_NO_DISCARD virtual Scalar zero() const;

    /**
     * @brief Get the closest scalar_t value to a given scalar
     * @param arg pointer to value
     * @return scalar_t value close to arg in value
     */
    RPY_NO_DISCARD virtual scalar_t to_scalar_t(ScalarPointer arg) const = 0;

    /**
     * @brief assign the rational value numerator/denominator to the target
     * scalar
     * @param target ScalarPointer to old value
     * @param numerator numerator of rational
     * @param denominator denominator of rational
     */
    virtual void
    assign(ScalarPointer target, long long numerator, long long denominator
    ) const = 0;

    /**
     * @brief Create a copy of a scalar value
     * @param source ScalarPointer to source value
     * @return new Scalar object
     */
    RPY_NO_DISCARD virtual Scalar copy(ScalarPointer source) const;

    /**
     * @brief Get the scalar whose value is minus given value
     * @param arg ScalarPointer to source value
     * @return new Scalar whose value is -(*arg)
     */
    RPY_NO_DISCARD virtual Scalar uminus(ScalarPointer arg) const = 0;

    /**
     * @brief Add one scalar value to another
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     * @return new Scalar with sum of the two values
     */
    RPY_NO_DISCARD virtual Scalar
    add(ScalarPointer lhs, ScalarPointer rhs) const;

    /**
     * @brief Subract one scalar value to another
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     * @return new Scalar with difference of the two values
     */
    RPY_NO_DISCARD virtual Scalar
    sub(ScalarPointer lhs, ScalarPointer rhs) const;

    /**
     * @brief Multiply two scalar values
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     * @return new Scalar with product of the two values
     */
    RPY_NO_DISCARD virtual Scalar
    mul(ScalarPointer lhs, ScalarPointer rhs) const;

    /**
     * @brief Divide one scalar value by another
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     * @return new scalar with left value divided by right value
     */
    RPY_NO_DISCARD virtual Scalar
    div(ScalarPointer lhs, ScalarPointer rhs) const;

    /**
     * @brief Add right value to left inplace
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     */
    virtual void add_inplace(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    /**
     * @brief Subtract right value from left value inplace
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     */
    virtual void sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    /**
     * @brief Multiply left value by right value inplace
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     */
    virtual void mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    /**
     * @brief Divide left value by right value inplace
     * @param lhs ScalarPointer to left value
     * @param rhs ScalarPointer to right value
     */
    virtual void div_inplace(ScalarPointer lhs, ScalarPointer rhs) const = 0;

    /**
     * @brief Compute the sum of two scalar arrays componentwise, with optional
     * mask
     * @param dst Destination to write result
     * @param lhs Left hand side input,
     * @param rhs Right hand side input
     * @param count Number of elements to add
     * @param mask Mask to apply, use nullptr for no mask. Otherwise, this
     * must must be a pointer to at least ceil(count / 64) uint64_ts where each
     * bit denotes a flag, set 1 for perform operation on corresponding elements
     * and 0 for not operation.
     */
    virtual void add_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const = 0;

    virtual void sub_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const = 0;

    virtual void mul_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const = 0;

    virtual void div_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const = 0;

    /**
     * @brief Test if the given value is equal to zero
     * @param arg argument to test
     * @return bool, true if the scalar pointer points to zero
     */
    RPY_NO_DISCARD virtual bool is_zero(ScalarPointer arg) const;

    /**
     * @brief Test if two scalars are equal
     * @param lhs pointer to left value
     * @param rhs pointer to right value
     * @return bool, true if left == right
     */
    RPY_NO_DISCARD virtual bool
    are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept
            = 0;

    /**
     * @brief Print the value of a ScalarPointer to an output stream
     * @param arg ScalarPointer to value
     * @param os output stream to print
     */
    virtual void print(ScalarPointer arg, std::ostream& os) const;

    /**
     * @brief Get a new random number generator for this scalar type
     * @param bit_generator Source of randomness used for generating random
     * numbers
     * @param seed Seed bits (as a slice/array) of uint64_t (regardless of bit
     * generator's seed type).
     * @return Pointer to new RandomGenerator instance.
     */
    RPY_NO_DISCARD virtual std::unique_ptr<RandomGenerator>
    get_rng(const string& bit_generator = "", Slice<uint64_t> seed = {}) const;

    /**
     * @brief Get a new instance of a blas interface
     * @return
     */
    RPY_NO_DISCARD virtual std::unique_ptr<BlasInterface> get_blas() const;

    /**
     * @brief Produce a stream of raw bytes after any pointer resolution.
     * @param ptr Input pointer
     * @param count Number of elements to process
     * @return Vector of bytes (char)
     */
    RPY_NO_DISCARD virtual std::vector<byte>
    to_raw_bytes(const ScalarPointer& ptr, dimn_t count) const = 0;

    /**
     * @brief Read raw bytes into a scalar array.
     * @param raw_bytes
     * @return
     */
    RPY_NO_DISCARD virtual ScalarPointer
    from_raw_bytes(Slice<byte> raw_bytes, dimn_t count) const
            = 0;
};

inline bool operator==(const ScalarType& lhs, const ScalarType& rhs) noexcept
{
    return std::addressof(lhs) == std::addressof(rhs);
}
inline bool operator!=(const ScalarType& lhs, const ScalarType& rhs) noexcept
{
    return std::addressof(lhs) != std::addressof(rhs);
}

// Implementation of type getting mechanics

namespace dtl {

#define ROUGHPY_MAKE_TYPE_ID_OF(TYPE, NAME)                                    \
    template <>                                                                \
    struct RPY_EXPORT type_id_of_impl<TYPE> {                                  \
        static const string& get_id() noexcept;                                \
    }

ROUGHPY_MAKE_TYPE_ID_OF(char, "i8");

ROUGHPY_MAKE_TYPE_ID_OF(unsigned char, "u8");

ROUGHPY_MAKE_TYPE_ID_OF(short, "i16");

ROUGHPY_MAKE_TYPE_ID_OF(unsigned short, "u16");

ROUGHPY_MAKE_TYPE_ID_OF(int, "i32");

ROUGHPY_MAKE_TYPE_ID_OF(unsigned int, "u32");

ROUGHPY_MAKE_TYPE_ID_OF(long long, "i64");

ROUGHPY_MAKE_TYPE_ID_OF(unsigned long long, "u64");

ROUGHPY_MAKE_TYPE_ID_OF(signed_size_type_marker, "isize");

ROUGHPY_MAKE_TYPE_ID_OF(unsigned_size_type_marker, "usize");

ROUGHPY_MAKE_TYPE_ID_OF(half, "f16");

ROUGHPY_MAKE_TYPE_ID_OF(bfloat16, "bf16");

ROUGHPY_MAKE_TYPE_ID_OF(float, "f32");

ROUGHPY_MAKE_TYPE_ID_OF(double, "f64");

ROUGHPY_MAKE_TYPE_ID_OF(rational_scalar_type, "Rational");

ROUGHPY_MAKE_TYPE_ID_OF(rational_poly_scalar, "RationalPoly");

// Long is silly. On Win64 it is 32 bits (because, Microsoft) on Unix, it is 64
// bits
template <>
struct type_id_of_impl<long>
    : public std::conditional_t<
              (sizeof(long) == sizeof(int)), type_id_of_impl<int>,
              type_id_of_impl<long long>> {
};

#undef ROUGHPY_MAKE_TYPE_ID_OF

template <typename ScalarImpl>
struct RPY_EXPORT scalar_type_holder {
    static const ScalarType* get_type() noexcept { return nullptr; }
};

// template <typename ScalarImpl>
// struct scalar_type_holder<ScalarImpl &> : scalar_type_holder<ScalarImpl> {};
//
// template <typename ScalarImpl>
// struct scalar_type_holder<const ScalarImpl &> :
// scalar_type_holder<ScalarImpl> {};

template <>
struct RPY_EXPORT scalar_type_holder<float> {
    static const ScalarType* get_type() noexcept;
};

template <>
struct RPY_EXPORT scalar_type_holder<double> {
    static const ScalarType* get_type() noexcept;
};

template <>
struct RPY_EXPORT scalar_type_holder<rational_scalar_type> {
    static const ScalarType* get_type() noexcept;
};

template <>
struct RPY_EXPORT scalar_type_holder<rational_poly_scalar> {
    static const ScalarType* get_type() noexcept;
};

template <>
struct RPY_EXPORT scalar_type_holder<half> {
    static const ScalarType* get_type() noexcept;
};

template <>
struct RPY_EXPORT scalar_type_holder<bfloat16> {
    static const ScalarType* get_type() noexcept;
};

}// namespace dtl

template <typename T>
inline const ScalarType* ScalarType::of()
{
    return dtl::scalar_type_holder<remove_cv_ref_t<T>>::get_type();
}

template <typename T>
const ScalarType* ScalarType::of(const ScalarDeviceInfo& device)
{
    return get_type(type_id_of<T>(), device);
}

template <typename T>
inline const string& type_id_of() noexcept
{
    return dtl::type_id_of_impl<T>::get_id();
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_TYPE_H_
