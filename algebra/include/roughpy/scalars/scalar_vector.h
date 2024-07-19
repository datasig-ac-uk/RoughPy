//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALARS_SCALAR_VECTOR_H
#define ROUGHPY_SCALARS_SCALAR_VECTOR_H

#include "scalars_fwd.h"
#include <roughpy/core/smart_ptr.h>
#include <roughpy/platform/alloc.h>

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/core.h>
#include <roughpy/devices/device_handle.h>
#include <roughpy/devices/host_device.h>
#include <roughpy/devices/type.h>

#include "scalar_array.h"

namespace rpy {
namespace scalars {

namespace dtl {

class VectorData;

class ScalarVectorIterator
{
};
}// namespace dtl

class VectorOperation;

class ROUGHPY_SCALARS_EXPORT ScalarVector
{
public:
    using VectorDataPtr = Rc<dtl::VectorData>;

private:
    VectorDataPtr p_base = nullptr;
    VectorDataPtr p_fibre = nullptr;

    friend class MutableVectorElement;
    friend class VectorOperation;

    dtl::VectorData& base_data() noexcept { return *p_base; }
    const dtl::VectorData& base_data() const noexcept { return *p_base; }
    dtl::VectorData& fibre_data() noexcept { return *p_fibre; }
    const dtl::VectorData& fibre_data() const noexcept { return *p_fibre; }

public:
    using iterator = dtl::ScalarVectorIterator;
    using const_iterator = dtl::ScalarVectorIterator;
    using value_type = Scalar;
    using reference = ScalarRef;
    using const_reference = ScalarCRef;

protected:
    void resize_dim(dimn_t new_dim);
    RPY_NO_DISCARD dimn_t buffer_size() const noexcept;

    void set_zero() const noexcept;

    ScalarVector(VectorDataPtr base, VectorDataPtr fibre);

    ScalarArray& mut_scalars() const noexcept;
    const ScalarArray& scalars() const noexcept;

public:
    ScalarVector();
    explicit ScalarVector(TypePtr scalar_type, dimn_t size = 0);

    ~ScalarVector();

    RPY_NO_DISCARD bool fast_is_zero() const noexcept;

    RPY_NO_DISCARD ScalarVector base() const noexcept;
    RPY_NO_DISCARD ScalarVector fibre() const noexcept;

    RPY_NO_DISCARD devices::Device device() const noexcept;
    RPY_NO_DISCARD TypePtr scalar_type() const noexcept;
    RPY_NO_DISCARD dimn_t dimension() const noexcept;
    RPY_NO_DISCARD dimn_t size() const noexcept;
    RPY_NO_DISCARD bool is_zero() const noexcept;

    RPY_NO_DISCARD const_reference get(dimn_t index) const;
    RPY_NO_DISCARD reference get_mut(dimn_t index);

    RPY_NO_DISCARD const_iterator begin() const noexcept;
    RPY_NO_DISCARD const_iterator end() const noexcept;

    template <typename V>
    RPY_NO_DISCARD enable_if_t<is_base_of_v<ScalarVector, V>, V> borrow() const
    {
        return V(p_base, p_fibre);
    }

    template <typename V>
    RPY_NO_DISCARD enable_if_t<is_base_of_v<ScalarVector, V>, V> borrow_mut()
    {
        return V(p_base, p_fibre);
    }

    RPY_NO_DISCARD ScalarVector uminus() const;

    RPY_NO_DISCARD ScalarVector add(const ScalarVector& other) const;

    RPY_NO_DISCARD ScalarVector sub(const ScalarVector& other) const;

    RPY_NO_DISCARD ScalarVector left_smul(const Scalar& scalar) const;
    RPY_NO_DISCARD ScalarVector right_smul(const Scalar& scalar) const;
    RPY_NO_DISCARD ScalarVector sdiv(const Scalar& scalar) const;

    ScalarVector& add_inplace(const ScalarVector& other);
    ScalarVector& sub_inplace(const ScalarVector& other);
    ScalarVector& left_smul_inplace(const Scalar& other);
    ScalarVector& right_smul_inplace(const Scalar& other);
    ScalarVector& sdiv_inplace(const Scalar& other);

    ScalarVector& add_scal_mul(const ScalarVector& other, const Scalar& scalar);
    ScalarVector& sub_scal_mul(const ScalarVector& other, const Scalar& scalar);

    ScalarVector& add_scal_div(const ScalarVector& other, const Scalar& scalar);
    ScalarVector& sub_scal_div(const ScalarVector& other, const Scalar& scalar);

    RPY_NO_DISCARD bool operator==(const ScalarVector& other) const;
    RPY_NO_DISCARD bool operator!=(const ScalarVector& other) const
    {
        return !operator==(other);
    }
};

namespace ops = devices::operators;

class ROUGHPY_SCALARS_EXPORT VectorOperation
{
protected:
    void resize_destination(ScalarVector& arg, dimn_t new_size) const;

public:
    virtual ~VectorOperation();

    virtual dimn_t arity() const noexcept = 0;
};

class ROUGHPY_SCALARS_EXPORT UnaryVectorOperation : public VectorOperation
{

public:
    static constexpr dimn_t my_arity = 1;
    dimn_t arity() const noexcept override { return my_arity; }

    virtual void
    eval(ScalarVector& destination,
         const ScalarVector& source,
         const ops::Operator& op) const;

    virtual void eval_inplace(ScalarVector& arg, const ops::Operator& op) const;
};

class ROUGHPY_SCALARS_EXPORT BinaryVectorOperation : public VectorOperation
{
public:
    static constexpr dimn_t my_arity = 2;

    dimn_t arity() const noexcept override { return my_arity; }
    virtual void
    eval(ScalarVector& destination,
         const ScalarVector& left,
         const ScalarVector& right,
         const ops::Operator& op) const;

    virtual void eval_inplace(
            ScalarVector& left,
            const ScalarVector& right,
            const ops::Operator& op
    ) const;
};

class ROUGHPY_SCALARS_EXPORT TernaryVectorOperation : public VectorOperation
{
public:
    static constexpr dimn_t my_arity = 3;

    dimn_t arity() const noexcept override { return my_arity; }
    virtual void
    eval(ScalarVector& destination,
         const ScalarVector& first,
         const ScalarVector& second,
         const ScalarVector& third,
         const ops::Operator& op) const;

    virtual void eval_inplace(
            ScalarVector& first,
            const ScalarVector& second,
            const ScalarVector& third,
            const ops::Operator& op
    ) const;
};

template <typename Op>
enable_if_t<is_base_of_v<VectorOperation, Op>, const Op&>
op_cast(const VectorOperation& op)
{
    RPY_CHECK(Op::my_arity == op.arity());
    return static_cast<const Op&>(op);
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_VECTOR_H
