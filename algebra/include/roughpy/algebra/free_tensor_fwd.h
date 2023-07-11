#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_FWD_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_FWD_H_

#include "algebra_base.h"
#include "algebra_bundle.h"
#include "tensor_basis.h"
#include <roughpy/platform/serialization.h>

namespace rpy {
namespace algebra {

// extern template class AlgebraInterface<FreeTensor, TensorBasis>;

class RPY_EXPORT FreeTensorInterface
    : public AlgebraInterface<FreeTensor, TensorBasis>
{
public:
    using algebra_interface_t = AlgebraInterface<FreeTensor, TensorBasis>;

    using algebra_interface_t::algebra_interface_t;

    RPY_NO_DISCARD
    virtual FreeTensor exp() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensor log() const = 0;
//    RPY_NO_DISCARD
//    virtual FreeTensor inverse() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensor antipode() const = 0;
    virtual void fmexp(const FreeTensor& other) = 0;
};

template <typename, template <typename> class>
class FreeTensorImplementation;

// extern template class AlgebraBase<FreeTensorInterface,
//                                   FreeTensorImplementation>;

class RPY_EXPORT FreeTensor
    : public AlgebraBase<FreeTensorInterface, FreeTensorImplementation>
{
    using base_t = AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;

public:
    using base_t::base_t;

    static constexpr AlgebraType s_alg_type = AlgebraType::FreeTensor;

    RPY_NO_DISCARD
    FreeTensor exp() const;
    RPY_NO_DISCARD
    FreeTensor log() const;
//    RPY_NO_DISCARD
//    FreeTensor inverse() const;
    RPY_NO_DISCARD
    FreeTensor antipode() const;
    FreeTensor& fmexp(const FreeTensor& other);

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(FreeTensor) { RPY_SERIAL_SERIALIZE_BASE(base_t); }

// extern template class BundleInterface<FreeTensorBundle, FreeTensor,
// FreeTensor>;

class RPY_EXPORT FreeTensorBundleInterface
    : public BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor>
{
public:
    using algebra_interface_t
            = BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor>;

    using algebra_interface_t::algebra_interface_t;

    RPY_NO_DISCARD
    virtual FreeTensorBundle exp() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensorBundle log() const = 0;
//    RPY_NO_DISCARD
//    virtual FreeTensorBundle inverse() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensorBundle antipode() const = 0;
    virtual void fmexp(const FreeTensorBundle& other) = 0;
};

template <typename, template <typename> class>
class FreeTensorBundleImplementation;

// extern template class AlgebraBundleBase<FreeTensorBundleInterface,
//                                         FreeTensorBundleImplementation>;

class RPY_EXPORT FreeTensorBundle
    : public AlgebraBundleBase<
              FreeTensorBundleInterface, FreeTensorBundleImplementation>
{
    using base_t = AlgebraBundleBase<
            FreeTensorBundleInterface, FreeTensorBundleImplementation>;

public:
    using base_t::base_t;

    static constexpr AlgebraType s_alg_type = AlgebraType::FreeTensorBundle;

    RPY_NO_DISCARD
    FreeTensorBundle exp() const;
    RPY_NO_DISCARD
    FreeTensorBundle log() const;
//    RPY_NO_DISCARD
//    FreeTensorBundle inverse() const;
    RPY_NO_DISCARD
    FreeTensorBundle antipode() const;
    FreeTensorBundle& fmexp(const FreeTensorBundle& other);

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(FreeTensorBundle)
{
    RPY_SERIAL_SERIALIZE_BASE(base_t);
}

}// namespace algebra
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::FreeTensor, rpy::serial::specialization::member_serialize
)
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::FreeTensorBundle,
        rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_FWD_H_
