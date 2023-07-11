#ifndef ROUGHPY_ALGEBRA_LIE_FWD_H_
#define ROUGHPY_ALGEBRA_LIE_FWD_H_

#include "algebra_base.h"
#include "algebra_bundle.h"

#include "lie_basis.h"

namespace rpy {
namespace algebra {

// extern template class AlgebraInterface<Lie, LieBasis>;

class RPY_EXPORT LieInterface : public AlgebraInterface<Lie, LieBasis>
{
    using base_t = AlgebraInterface<Lie, LieBasis>;

public:
    using base_t::base_t;
};

// extern template class AlgebraBase<LieInterface>;

class RPY_EXPORT Lie : public AlgebraBase<LieInterface>
{
    using base_t = AlgebraBase<LieInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::Lie;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

class LieBundle;

// extern template class BundleInterface<LieBundle, Lie, Lie>;

class RPY_EXPORT LieBundleInterface
    : public BundleInterface<LieBundle, Lie, Lie>
{
    using base_t = BundleInterface<LieBundle, Lie, Lie>;

public:
    using base_t::base_t;
};

// extern template class AlgebraBundleBase<LieBundleInterface>;

class RPY_EXPORT LieBundle : public AlgebraBundleBase<LieBundleInterface>
{
    using base_t = AlgebraBundleBase<LieBundleInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::LieBundle;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

}// namespace algebra
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::LieInterface,
        rpy::serial::specialization::member_serialize
);
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::Lie, rpy::serial::specialization::member_serialize
)
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::LieBundleInterface,
        rpy::serial::specialization::member_serialize
);
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::LieBundle, rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_ALGEBRA_LIE_FWD_H_
