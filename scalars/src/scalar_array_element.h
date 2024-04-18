//
// Created by sam on 3/12/24.
//

#ifndef ROUGHPY_SCALAR_ARRAY_ELEMENT_H
#define ROUGHPY_SCALAR_ARRAY_ELEMENT_H

#include "scalar_interface.h"

#include "devices/buffer.h"
#include "scalar/packed_type.h"

namespace rpy {
namespace scalars {

class ScalarArrayElement : public ScalarInterface
{
    devices::Buffer m_buffer;
    dimn_t m_index;
    PackedScalarType p_type_or_info;

public:
    ScalarArrayElement(
            devices::Buffer& buffer,
            dimn_t index,
            const ScalarType* type = nullptr
    );
    ScalarArrayElement(
            const devices::Buffer& buffer,
            dimn_t index,
            const ScalarType* type = nullptr
    );

    PackedScalarType type() const noexcept override;

    void* mut_pointer();
    const void* pointer() const noexcept override;
    void set_value(const Scalar& value) override;
    void print(std::ostream& os) const override;
    void add_inplace(const Scalar& other) override;
    void sub_inplace(const Scalar& other) override;
    void mul_inplace(const Scalar& other) override;
    void div_inplace(const Scalar& other) override;
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALAR_ARRAY_ELEMENT_H
