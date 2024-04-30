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

#include <roughpy/devices/core.h>
#include <roughpy/devices/type.h>
#include <roughpy/devices/device_handle.h>
#include "scalars_fwd.h"

#include "packed_scalar_type_ptr.h"

#include <roughpy/core/container/unordered_map.h>

namespace rpy {
namespace scalars {

/**
 * @struct RingCharacteristics
 * @brief Represents the characteristics of a mathematical ring.
 *
 * The RingCharacteristics struct is used to store the properties of a
 * mathematical ring. It contains four boolean flags representing different
 * characteristics of the ring. The flags are used to indicate whether the ring
 * is a field, whether it is ordered, whether it has square root functionality,
 * and whether it deals with complex numbers.
 */
struct RingCharacteristics {
    bool is_field : 1;
    bool is_ordered : 1;
    bool has_sqrt : 1;
    bool is_complex : 1;
};

/**
 * @class ScalarType
 * @brief Represents a scalar type.
 */
class RPY_SCALAR_TYPE_ALIGNMENT ScalarType
{
    const devices::Type* p_type;

protected:
    using lock_type = std::recursive_mutex;
    using guard_type = std::lock_guard<lock_type>;
    using rng_getter = std::unique_ptr<
            RandomGenerator> (*)(const ScalarType*, Slice<seed_int_t>);

private:
    mutable lock_type m_lock;

public:
    devices::Device m_device;
    string_view m_name;
    string_view m_id;
    devices::TypeInfo m_info;
    RingCharacteristics m_characteristics;

    containers::HashMap<string, rng_getter> m_rng_getters;

    explicit ScalarType(
            const devices::Type* type,
            devices::Device device,
            RingCharacteristics characteristics
    );

    virtual ~ScalarType();

    RPY_NO_DISCARD guard_type lock() const noexcept
    {
        return guard_type(m_lock);
    }

    const devices::Type* as_type() const noexcept { return p_type; }

    template <typename T>
    static inline optional<const ScalarType*> of() noexcept
    {
        return scalar_type_of<T>();
    }

    template <typename T>
    static inline optional<const ScalarType*> of(const devices::Device& device)
    {
        auto host = of<T>();
        if (host) { return (*host)->with_device(device); }
        return {};
    }

    /**
     * @brief Returns a pointer to the ScalarType based on the provided
     * TypeInfo.
     *
     * The for_info method is a static member function of the ScalarType class.
     * It takes a reference to a TypeInfo object as input and returns a const
     * pointer to the corresponding ScalarType object based on the information
     * in the TypeInfo object.
     *
     * @param info The TypeInfo object that provides information about the data
     * type.
     * @return A const pointer to the ScalarType object based on the provided
     * TypeInfo.
     * @throws std::runtime_error if the data type is not supported.
     */
    static const ScalarType* for_info(const devices::TypeInfo& info);

    /**
     * @fn static const ScalarType* ScalarType::for_id(string_view id)
     * @brief Retrieves the ScalarType object associated with the given ID.
     *
     * This method returns a pointer to the ScalarType object that corresponds
     * to the provided ID. The ID is passed as a string_view parameter, which
     * allows for efficient access to the characters of the ID without
     * necessarily copying them to a new string object.
     *
     * @param id The ID of the ScalarType to retrieve.
     * @return A pointer to the ScalarType object associated with the given ID,
     * or nullptr if no ScalarType can be found.
     */
    static const ScalarType* for_id(string_view id);

    /**
     * @brief Get the name of this type
     */
    RPY_NO_DISCARD string_view name() const noexcept { return m_name; }

    /**
     * @brief Get the unique internal ID string for this type
     * @return const reference to the ID string.
     */
    RPY_NO_DISCARD string_view id() const noexcept { return m_id; }

    /**
     * @brief Get the underlying device for this type.
     * @return Pointer to device handle
     */
    RPY_NO_DISCARD devices::Device device() const noexcept { return m_device; }

    /**
     * @brief Returns the type information of a device.
     *
     * The type_info method is a const member function that returns the type
     * information of a device. It is used to retrieve the characteristics and
     * properties of a device.
     *
     * @return The type information of the device.
     *
     * @note This method does not throw any exceptions.
     */
    RPY_NO_DISCARD devices::TypeInfo type_info() const noexcept
    {
        return m_info;
    }

    /**
     * @brief Checks if the device is the CPU device.
     *
     * The `is_cpu` method is used to determine if the device is the CPU device.
     * It compares the device with the host device obtained from the `devices`
     * namespace. If the two devices are the same, it returns `true`; otherwise,
     * it returns `false`.
     *
     * @return `true` if the device is the CPU device, `false` otherwise.
     */
    RPY_NO_DISCARD bool is_cpu() const noexcept
    {
        return m_device->is_host();
    }

    /**
     * @brief Get the rational type associated with this scalar type.
     *
     * If the scalar type is a field, then this should always return this.
     */
    RPY_NO_DISCARD virtual const ScalarType* rational_type() const noexcept;

    /**
     * @brief Allocate new scalars in memory
     * @param count Number of scalars to allocate space
     * @return ScalarPointer pointing to the newly allocated raw memory.
     *
     * Note that ScalarArrays are internally reference counted, so will
     * remain valid whilst there is a ScalarArray object with that data.
     */
    RPY_NO_DISCARD virtual ScalarArray allocate(dimn_t count) const;

    /**
     * @brief Allocate single scalar pointer for a Scalar.
     *
     * Only necessary for large scalar types.
     */
    RPY_NO_DISCARD virtual void* allocate_single() const;

    /**
     * @brief Free a previously allocated single scalar value.
     *
     * Only necessary for large scalar types
     */
    virtual void free_single(void* ptr) const;

protected:
    void register_rng_getter(string name, rng_getter getterhalf);

public:
    /**
     * @brief Get a new random number generator for this scalar type
     * @param bit_generator Source of randomness used for generating random
     * numbers
     * @param seed Seed bits (as a slice/array) of uint64_t (regardless of bit
     * generator's seed type).
     * @return Pointer to new RandomGenerator instance.
     */
    RPY_NO_DISCARD std::unique_ptr<RandomGenerator>
    get_rng(const string& bit_generator,
            Slice<seed_int_t> seed = nullptr) const;

    /**
     * @brief Copy the contents of one array into another.
     * @param dst Array to fill with values from src
     * @param src Source array of values.
     */
    virtual void convert_copy(ScalarArray& dst, const ScalarArray& src) const;

    /**
     * @brief Assign a value to all elements in an array
     * @param dst Array to place copies of value
     * @param value Value to place in array.
     */
    virtual void assign(ScalarArray& dst, Scalar value) const;

    // Scalar methods

    /**
     * @brief Get a new scalar type whose underlying device is given.
     *
     * @param device device on which the new scalar type should be based.
     */
    virtual const ScalarType* with_device(const devices::Device& device) const;
};

static_assert(
        alignof(ScalarType) >= min_scalar_type_alignment,
        "ScalarType must have alignment of at least 8 bytes so there are 3 "
        "free bits in the low end of pointers."
);

inline optional<devices::Device> device_from(const PackedScalarType& type
) noexcept
{
    return type.is_pointer() ? type->device() : optional<devices::Device>{};
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_TYPE_H_
