//
// Created by sam on 1/30/24.
//

#ifndef ROUGHPY_ALGEBRA_BASIS_KEY_H
#define ROUGHPY_ALGEBRA_BASIS_KEY_H

#include <roughpy/core/types.h>
#include <roughpy/core/helpers.h>
#include <roughpy/core/traits.h>

#include "algebra_fwd.h"
#include "roughpy_algebra_export.h"

#include <cstddef>
#include <atomic>

namespace rpy {
namespace algebra {


enum class BasisKeyType {
    Simple,
    Ordered,
    Graded,
    WordLike
};


class ROUGHPY_ALGEBRA_EXPORT BasisKeyInterface {
    mutable std::atomic_uint32_t m_ref_count;
public:
    virtual ~BasisKeyInterface();

    virtual string_view key_type() const noexcept = 0;

    virtual BasisPointer basis() const noexcept = 0;

    uint32_t inc_ref() const noexcept;

    uint32_t dec_ref() const noexcept;
};


class BasisKey {
    using data_type = uintptr_t;
    static_assert(sizeof(data_type) == sizeof(void*),
                  "data type must be the same size as a pointer");

    data_type m_data;

    static_assert(alignof(BasisKeyInterface) >= 2,
                  "alignment of pointer must be at least 2");

    static constexpr data_type spare_bits = alignof(BasisKeyInterface);
    static constexpr data_type flag_mask = 1;
    static constexpr data_type index_offset = 1;

    static constexpr data_type valid_pointer_mask = ~(
        (data_type(1) << spare_bits) - 1);


public:

    BasisKey() : m_data(0) {}

    constexpr explicit BasisKey(const BasisKeyInterface* pointer) noexcept
        : m_data(bit_cast<data_type>(pointer))
    {
        RPY_DBG_ASSERT(pointer != nullptr);
        pointer->inc_ref();
    }

    ~BasisKey()
    {
        if (is_valid_pointer() && get_pointer()->dec_ref() == 0) {
            delete const_cast<BasisKeyInterface*>(get_pointer());
        }
    }

    constexpr explicit BasisKey(dimn_t index) noexcept
        : m_data((static_cast<data_type>(index) << index_offset) | flag_mask) {}


    BasisKey& operator=(std::nullptr_t) noexcept
    {
        m_data = 0;
        return *this;
    }

    constexpr operator bool() const noexcept { return m_data != 0; }

    constexpr bool is_index() const noexcept
    {
        return (m_data & flag_mask) == 1;
    }

    constexpr bool is_pointer() const noexcept
    {
        return (m_data & flag_mask) == 0;
    }

    constexpr bool is_valid_pointer() const noexcept
    {
        return (m_data & valid_pointer_mask) != 0;
    }

    constexpr dimn_t get_index() const noexcept
    {
        RPY_DBG_ASSERT(is_index());
        return static_cast<dimn_t>(m_data >> index_offset);
    }

    const BasisKeyInterface* get_pointer() const noexcept
    {
        RPY_DBG_ASSERT(is_valid_pointer());
        return bit_cast<const BasisKeyInterface*>(m_data);
    }


};


}
}


#endif //ROUGHPY_ALGEBRA_BASIS_KEY_H
