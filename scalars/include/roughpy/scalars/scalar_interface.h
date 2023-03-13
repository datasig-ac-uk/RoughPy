#ifndef ROUGHPY_SCALARS_SCALAR_INTERFACE_H_
#define ROUGHPY_SCALARS_SCALAR_INTERFACE_H_

#include "scalars_fwd.h"
#include "roughpy_scalars_export.h"


#include <iosfwd>

#include "scalar_pointer.h"


namespace rpy { namespace scalars {

class ROUGHPY_SCALARS_EXPORT ScalarInterface {

    public:
    virtual ~ScalarInterface() = default;

    virtual const ScalarType *type() const noexcept = 0;

    virtual bool is_const() const noexcept = 0;
    virtual bool is_value() const noexcept = 0;
    virtual bool is_zero() const noexcept = 0;

    virtual scalar_t as_scalar() const = 0;
    virtual void assign(ScalarPointer) = 0;
    virtual void assign(const Scalar &other) = 0;
    virtual void assign(const void *data, const std::string &type_id) = 0;

    virtual ScalarPointer to_pointer() = 0;
    virtual ScalarPointer to_pointer() const noexcept = 0;
    virtual Scalar uminus() const;


    virtual void add_inplace(const Scalar &other) = 0;
    virtual void sub_inplace(const Scalar &other) = 0;
    virtual void mul_inplace(const Scalar &other) = 0;
    virtual void div_inplace(const Scalar &other) = 0;

    virtual bool equals(const Scalar &other) const noexcept;

    virtual std::ostream &print(std::ostream &os) const;
};

}}

#endif // ROUGHPY_SCALARS_SCALAR_INTERFACE_H_
