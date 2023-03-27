#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_H_

#include "algebra_base.h"
#include "context.h"

#include "tensor_basis.h"

namespace rpy {
namespace algebra {


extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraInterface<FreeTensor, TensorBasis>;

class ROUGHPY_ALGEBRA_EXPORT FreeTensorInterface
    : public AlgebraInterface<FreeTensor, TensorBasis>
{
public:
    using algebra_interface_t = AlgebraInterface<FreeTensor, TensorBasis>;

    using algebra_interface_t::algebra_interface_t;

    virtual FreeTensor exp() const = 0;
    virtual FreeTensor log() const = 0;
    virtual FreeTensor inverse() const = 0;
    virtual FreeTensor antipode() const = 0;
    virtual void fmexp(const FreeTensor& other) = 0;
};

template <typename, template <typename> class>
class FreeTensorImplementation;

extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;

class ROUGHPY_ALGEBRA_EXPORT FreeTensor
    : public AlgebraBase<FreeTensorInterface, FreeTensorImplementation>
{
    using base_t = AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;
public:
    using base_t::base_t;

    static constexpr AlgebraType s_alg_type = AlgebraType::FreeTensor;

    FreeTensor exp() const;
    FreeTensor log() const;
    FreeTensor inverse() const;
    FreeTensor antipode() const;
    FreeTensor& fmexp(const FreeTensor& other);


};

}
}

#endif // ROUGHPY_ALGEBRA_FREE_TENSOR_H_
