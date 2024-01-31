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

namespace rpy {
namespace algebra {


enum class BasisKeyType {
    Simple,
    Ordered,
    Graded,
    WordLike
};


class ROUGHPY_ALGEBRA_EXPORT BasisKeyInterface {
public:
    virtual ~BasisKey();

    virtual BasisPointer basis() const noexcept;

    virtual BasisKeyType type() const noexcept;


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

public:

    BasisKey() : m_data(0) {}

    constexpr explicit BasisKey(const BasisKey* pointer) noexcept
        : m_data(bit_cast<data_type>(pointer))
    {
        RPY_DBG_ASSERT(pointer != nullptr);
    }

    constexpr explicit BasisKey(dimn_t index) noexcept
        : m_data((static_cast<data_type>(index) << index_offset) | flag_mask) {}


    BasisKey& operator=(std::nullptr_t) noexcept
    {
        m_data = 0;
    }

    constexpr operator bool() const noexcept { return m_data == 0; }

    constexpr bool is_index() const noexcept
    {
        return (m_data & flag_mask) == 1;
    }

    constexpr bool is_pointer() const noexcept
    {
        return (m_data & flag_mask) == 0;
    }

    constexpr dimn_t get_index() const noexcept
    {
        RPY_DBG_ASSERT(is_index());
        return static_cast<dimn_t>(m_data >> index_offset);
    }

    constexpr const BasisKey* get_pointer() const noexcept
    {
        RPY_DBG_ASSERT(is_pointer());
        return bit_cast<const BasisKey*>(m_data);
    }


};


}
}


#endif //ROUGHPY_ALGEBRA_BASIS_KEY_H
