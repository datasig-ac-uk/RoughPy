#ifndef ROUGHPY_SCALARS_OWNED_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_OWNED_SCALAR_ARRAY_H_

#include "scalars_fwd.h"

#include "scalar_array.h"

namespace rpy { namespace scalars {

class ROUGHPY_SCALARS_EXPORT OwnedScalarArray : public ScalarArray {
public:
    OwnedScalarArray() = default;

    OwnedScalarArray(const OwnedScalarArray &other);
    OwnedScalarArray(OwnedScalarArray &&other) noexcept;

    explicit OwnedScalarArray(const ScalarType *type);
    OwnedScalarArray(const ScalarType *type, dimn_t size);
    OwnedScalarArray(const Scalar &value, dimn_t count);
    explicit OwnedScalarArray(const ScalarArray &other);

    explicit OwnedScalarArray(const ScalarType *type, const void *data, dimn_t count);

    OwnedScalarArray &operator=(const ScalarArray &other);
    OwnedScalarArray &operator=(OwnedScalarArray &&other) noexcept;

    ~OwnedScalarArray();
};


}}

#endif // ROUGHPY_SCALARS_OWNED_SCALAR_ARRAY_H_
