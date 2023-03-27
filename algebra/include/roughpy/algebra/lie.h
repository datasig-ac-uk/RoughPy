#ifndef ROUGHPY_ALGEBRA_LIE_H_
#define ROUGHPY_ALGEBRA_LIE_H_

#include "algebra_base.h"

#include "lie_basis.h"

namespace rpy {
namespace algebra {


extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraInterface<Lie, LieBasis>;

using LieInterface = AlgebraInterface<Lie, LieBasis>;

extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraBase<LieInterface>;


class Lie : public AlgebraBase<LieInterface>
{
    using base_t = AlgebraBase<LieInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::Lie;

    using base_t::base_t;
};

}
}
#endif // ROUGHPY_ALGEBRA_LIE_H_
