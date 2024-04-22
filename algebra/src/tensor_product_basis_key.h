//
// Created by sam on 3/18/24.
//

#ifndef ROUGHPY_TENSOR_PRODUCT_BASIS_KEY_H
#define ROUGHPY_TENSOR_PRODUCT_BASIS_KEY_H

#include "basis_key.h"

#include <roughpy/core/container/vector.h>

namespace rpy {
namespace algebra {

class TensorProductBasisKey : public BasisKeyInterface
{
    containers::Vec<BasisKey> m_keys;

public:
    static constexpr string_view key_name = "tensor_product_key";

    explicit TensorProductBasisKey(Slice<BasisKey> keys) : m_keys(keys) {}

    string_view key_type() const noexcept override;

    decltype(auto) begin() const noexcept { return m_keys.begin(); }
    decltype(auto) end() const noexcept { return m_keys.end(); }
    dimn_t size() const noexcept { return m_keys.size(); }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_TENSOR_PRODUCT_BASIS_KEY_H
