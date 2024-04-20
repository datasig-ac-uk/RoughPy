//
// Created by sam on 3/27/24.
//

#ifndef ROUGPY_ALGEBRA_KEY_SCALAR_ARRAY_H
#define ROUGPY_ALGEBRA_KEY_SCALAR_ARRAY_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/scalars/devices/core.h>
#include <roughpy/scalars/scalar_array.h>

#include "basis_key.h"
#include "key_array.h"

namespace rpy {
namespace algebra {

/**
 * @class KeyScalarArray
 *
 * @brief A class that represents an array of scalar values with associated
 * keys.
 *
 * This class inherits from the `scalars::ScalarArray` class and adds the
 * functionality to store and manipulate associated keys. It provides methods to
 * allocate keys with a given size, allocate scalars with a given size and type,
 * and check if keys are present.
 *
 * @note This class is exported as part of the ROUGHPY_ALGEBRA module.
 */
class ROUGHPY_ALGEBRA_EXPORT KeyScalarArray : public scalars::ScalarArray
{
    KeyArray m_keys;

    using ScalarArray = scalars::ScalarArray;

public:
    KeyScalarArray() = default;

    KeyScalarArray(const KeyScalarArray&) = default;
    KeyScalarArray(KeyScalarArray&&) noexcept = default;

    explicit KeyScalarArray(ScalarArray&& scalars) noexcept
        : ScalarArray(std::move(scalars))
    {}

    KeyScalarArray(ScalarArray scalars, Slice<BasisKey> keys)
        : ScalarArray(std::move(scalars)),
          m_keys(keys)
    {}

    explicit KeyScalarArray(scalars::PackedScalarType type, dimn_t size);

    KeyScalarArray& operator=(const KeyScalarArray&) = default;
    KeyScalarArray& operator=(KeyScalarArray&&) noexcept = default;

    KeyScalarArray& operator=(const ScalarArray& other)
    {
        if (&other != this) {
            static_cast<ScalarArray&>(*this) = other;
            m_keys = KeyArray();
        }
        return *this;
    }
    KeyScalarArray& operator=(ScalarArray&& other) noexcept
    {
        if (&other != this) {
            static_cast<ScalarArray&>(*this) = std::move(other);
            m_keys = KeyArray();
        }

        return *this;
    }

    /**
     * @brief Allocate the keys for the KeyScalarArray with the given size.
     *
     * @param size The size of the keys to allocate.
     *
     * This method allocates the keys for the KeyScalarArray with the given
     * size. The keys are represented by an instance of the KeyArray class.
     */
    void allocate_keys(const dimn_t size) { m_keys = KeyArray(size); }

    /**
     *
     */
    void
    allocate_scalars(dimn_t size, scalars::PackedScalarType type=nullptr)
    {
        if (type.is_null()) { type = this->type(); }
        RPY_CHECK(!type.is_null());
        *this = ScalarArray(type, size);
    }

    RPY_NO_DISCARD bool has_keys() const noexcept  {return !m_keys.empty(); }

    ~KeyScalarArray();

    const KeyArray& keys() const noexcept { return m_keys; }
    KeyArray& keys() noexcept { return m_keys; }

    RPY_SERIAL_LOAD_FN();
    RPY_SERIAL_SAVE_FN();
};


RPY_SERIAL_LOAD_FN_IMPL(KeyScalarArray)
{
    RPY_SERIAL_SERIALIZE_BASE(ScalarArray);
    bool hkeys = false;
    RPY_SERIAL_SERIALIZE_NVP("has_keys", hkeys);
    if (hkeys) {
        std::vector<key_type> tmp_keys;
        RPY_SERIAL_SERIALIZE_NVP("keys", tmp_keys);

        // RPY_CHECK(tmp_keys.size() == size());
        // allocate_keys();
        // std::memcpy(keys(), tmp_keys.data(), size() * sizeof(key_type));
        // std::copy(tmp_keys.begin(), tmp_keys.end(), keys());
    }
}

RPY_SERIAL_SAVE_FN_IMPL(KeyScalarArray)
{
    RPY_SERIAL_SERIALIZE_BASE(ScalarArray);
    bool hkeys = has_keys();
    RPY_SERIAL_SERIALIZE_NVP("has_keys", hkeys);
    if (hkeys) {
        // std::vector<key_type> tmp_keys(p_keys, p_keys + size());
        // RPY_SERIAL_SERIALIZE_NVP("keys", tmp_keys);
    }
}

}// namespace algebra
}// namespace rpy

#endif// ROUGPY_ALGEBRA_KEY_SCALAR_ARRAY_H
