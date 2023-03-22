#ifndef ROUGHPY_SCALARS_RANDOM_H_
#define ROUGHPY_SCALARS_RANDOM_H_

#include "owned_scalar_array.h"
#include "roughpy/config/helpers.h"
#include "scalar_type.h"

namespace rpy {
namespace scalars {

class ROUGHPY_SCALARS_EXPORT RandomGenerator {
protected:
    const ScalarType *p_type;

public:
    explicit RandomGenerator(const ScalarType *type)
        : p_type(type) {}

    virtual ~RandomGenerator() = default;

    virtual void set_seed(Slice<uint64_t> seed_data) = 0;

    virtual std::vector<uint64_t> get_seed() const = 0;

    virtual OwnedScalarArray uniform_random_scalar(ScalarArray lower, ScalarArray upper, dimn_t count) const = 0;
    virtual OwnedScalarArray normal_random(Scalar loc, Scalar scale, dimn_t count) const = 0;
};



}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_RANDOM_H_
