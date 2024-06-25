//
// Created by sam on 21/03/24.
//

#ifndef ROUGHPY_MUTABLE_VECTOR_ELEMENT_H
#define ROUGHPY_MUTABLE_VECTOR_ELEMENT_H

#include "basis_key.h"
#include "key_array.h"
#include "vector.h"


namespace rpy {
namespace algebra {

class MutableVectorElement : public scalars::ScalarInterface
{
    Vector* p_vector;
    BasisKey m_key;
    scalars::Scalar m_value;
    optional<dimn_t> m_loc;

    void get_value();

public:
    MutableVectorElement(Vector* vector, BasisKey key)
        : p_vector(vector),
          m_key(std::move(key))
    {}

    scalars::PackedScalarType type() const noexcept override;

    virtual ~MutableVectorElement();
    virtual const void* pointer() const noexcept;
    virtual void set_value(const scalars::Scalar& value);
    virtual void print(std::ostream& os) const;
    virtual void add_inplace(const scalars::Scalar& other);
    virtual void sub_inplace(const scalars::Scalar& other);
    virtual void mul_inplace(const scalars::Scalar& other);
    virtual void div_inplace(const scalars::Scalar& other);
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_MUTABLE_VECTOR_ELEMENT_H
