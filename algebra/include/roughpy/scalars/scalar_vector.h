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

#include <roughpy/device_support/host_kernel.h>

#include "roughpy/devices/operation.h"
#include "scalar_array.h"

namespace rpy {
namespace scalars {

namespace dtl {

class ScalarVectorIterator
{
};
}// namespace dtl

class VectorOperation;

class ROUGHPY_SCALARS_EXPORT ScalarVector : public RcBase<ScalarVector>,
                                            public platform::SmallObjectBase
{
    ScalarArray m_base_data;
    ScalarArray m_fibre_data;

    friend class MutableVectorElement;
    friend class VectorOperation;

public:
    using iterator = dtl::ScalarVectorIterator;
    using const_iterator = dtl::ScalarVectorIterator;
    using value_type = Scalar;
    using reference = ScalarRef;
    using const_reference = ScalarCRef;

protected:
    void resize_base_dim(dimn_t new_dim);
    void resize_fibre_dim(dimn_t new_dim);

    RPY_NO_DISCARD dimn_t base_buffer_size() const noexcept
    {
        return m_base_data.size();
    };
    RPY_NO_DISCARD dimn_t fibre_buffer_size() const noexcept
    {
        return m_fibre_data.size();
    };

    void set_base_zero();
    void set_fibre_zero();
    void set_zero()
    {
        set_base_zero();
        set_fibre_zero();
    }

    ScalarArray& mut_base_data();
    const ScalarArray& base_data() const noexcept { return m_base_data; };
    ScalarArray& mut_fibre_data();
    const ScalarArray& fibre_data() const noexcept { return m_fibre_data; };

    ScalarVector(ScalarArray base, ScalarArray fibre)
        : m_base_data(std::move(base)),
          m_fibre_data(std::move(fibre)){};

public:
    ScalarVector();
    explicit ScalarVector(TypePtr scalar_type, dimn_t size = 0);

    explicit ScalarVector(ScalarArray&& data)
        : m_base_data(std::move(data)),
          m_fibre_data(m_base_data.type())
    {}

    ~ScalarVector();

    RPY_NO_DISCARD bool fast_is_zero() const noexcept
    {
        return m_base_data.empty();
    }

    RPY_NO_DISCARD ScalarVector base() const noexcept
    {
        return {m_base_data, {}};
    }
    RPY_NO_DISCARD ScalarVector fibre() const noexcept
    {
        return {m_fibre_data, {}};
    }

    RPY_NO_DISCARD devices::Device device() const noexcept
    {
        return m_base_data.device();
    }

    RPY_NO_DISCARD TypePtr scalar_type() const noexcept
    {
        return m_base_data.type();
    };
    RPY_NO_DISCARD dimn_t dimension() const noexcept
    {
        return base_buffer_size();
    };

    RPY_NO_DISCARD dimn_t base_dimension() const noexcept
    {
        return m_base_data.size();
    }
    RPY_NO_DISCARD dimn_t fibre_dimension() const noexcept
    {
        return m_fibre_data.size();
    }

    RPY_NO_DISCARD dimn_t size() const noexcept;
    RPY_NO_DISCARD dimn_t base_size() const noexcept;
    RPY_NO_DISCARD dimn_t fibre_size() const noexcept;

    RPY_NO_DISCARD bool is_zero() const noexcept;

    RPY_NO_DISCARD const_reference get(dimn_t index) const;
    RPY_NO_DISCARD reference get_mut(dimn_t index);

    RPY_NO_DISCARD const_iterator begin() const noexcept;
    RPY_NO_DISCARD const_iterator end() const noexcept;

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

    virtual optional<devices::Kernel> get_kernel(
            devices::Device device,
            string_view base_name,
            Slice<const Type* const> types
    ) const;

    virtual devices::KernelLaunchParams
    get_launch_params(const devices::KernelArguments& args) const noexcept;

    void eval_impl(
            string_view base_name,
            const devices::Kernel& generic_kernel,
            const devices::KernelArguments& kernel_args
    ) const
    {
        auto params = get_launch_params(kernel_args);

        const auto generic_types = kernel_args.get_generic_types();
        devices::EventStatus result;
        if (auto kernel
            = get_kernel(kernel_args.get_device(), base_name, generic_types)) {
            result = kernel->launch_sync(params, kernel_args);
        } else {
            result = generic_kernel.launch_sync(params, kernel_args);
        }
        RPY_CHECK(result == devices::EventStatus::CompletedSuccessfully);
    }

public:
    virtual ~VectorOperation();

    virtual dimn_t arity() const noexcept = 0;
};

class ROUGHPY_SCALARS_EXPORT UnaryVectorOperation : public VectorOperation
{
protected:
    using VectorOperation::VectorOperation;

public:
    static constexpr dimn_t my_arity = 1;
    dimn_t arity() const noexcept override { return my_arity; }

    virtual void
    eval(ScalarVector& destination,
         const ScalarVector& source,
         const ops::Operator& op) const
            = 0;

    virtual void eval_inplace(ScalarVector& arg, const ops::Operator& op) const
            = 0;
};

class ROUGHPY_SCALARS_EXPORT BinaryVectorOperation : public VectorOperation
{
protected:
    using VectorOperation::VectorOperation;

public:
    static constexpr dimn_t my_arity = 2;

    dimn_t arity() const noexcept override { return my_arity; }
    virtual void
    eval(ScalarVector& destination,
         const ScalarVector& left,
         const ScalarVector& right,
         const ops::Operator& op) const
            = 0;

    virtual void eval_inplace(
            ScalarVector& left,
            const ScalarVector& right,
            const ops::Operator& op
    ) const = 0;
};

class ROUGHPY_SCALARS_EXPORT TernaryVectorOperation : public VectorOperation
{
protected:
    using VectorOperation::VectorOperation;

public:
    static constexpr dimn_t my_arity = 3;

    dimn_t arity() const noexcept override { return my_arity; }

    virtual void
    eval(ScalarVector& destination,
         const ScalarVector& first,
         const ScalarVector& second,
         const ScalarVector& third,
         const ops::Operator& op) const
            = 0;

    virtual void eval_inplace(
            ScalarVector& first,
            const ScalarVector& second,
            const ScalarVector& third,
            const ops::Operator& op
    ) const = 0;
};

template <template <typename> class Operation>
class StandardUnaryVectorOperation : public UnaryVectorOperation
{

    template <typename T>
    using KernelType = devices::UnaryHostKernel<Operation, T>;

    template <typename T>
    using InplaceKernelType = devices::UnaryInplaceHostKernel<Operation, T>;

    using GenericKernelType = KernelType<devices::Value>;
    using GenericInplaceKernelType = InplaceKernelType<devices::Value>;

public:
    StandardUnaryVectorOperation();

    void
    eval(ScalarVector& destination,
         const ScalarVector& source,
         const ops::Operator& op) const override;
    void
    eval_inplace(ScalarVector& arg, const ops::Operator& op) const override;
};

template <template <typename> class Operation>
StandardUnaryVectorOperation<Operation>::StandardUnaryVectorOperation()
{
    // TODO: Register the host kernels.
}

template <template <typename> class Operation>
void StandardUnaryVectorOperation<Operation>::eval(
        ScalarVector& destination,
        const ScalarVector& source,
        const ops::Operator& op
) const
{
    resize_destination(destination, source.dimension());

    auto binding = GenericKernelType::new_binding();
    binding->bind(destination);
    binding->bind(source);
    binding->bind(op);

    eval_impl(
            GenericKernelType::get_base_name(),
            *GenericKernelType::get(),
            *binding
    );
}
template <template <typename> class Operation>
void StandardUnaryVectorOperation<Operation>::eval_inplace(
        ScalarVector& arg,
        const ops::Operator& op
) const
{
    // No resizing should be necessary.
    auto binding = GenericInplaceKernelType::new_binding();
    binding->bind(arg);
    binding->bind(op);

    eval_impl(
            GenericInplaceKernelType::get_base_name(),
            *GenericInplaceKernelType::get(),
            *binding
    );
}

template <template <typename> class Operation>
class StandardBinaryVectorOperation : public BinaryVectorOperation
{

    template <typename T>
    using KernelType = devices::BinaryHostKernel<Operation, T>;
    template <typename T>
    using InplaceKernelType = devices::BinaryInplaceHostKernel<Operation, T>;

    using GenericKernelType = KernelType<devices::Value>;
    using GenericInplaceKernelType = InplaceKernelType<devices::Value>;

public:
    StandardBinaryVectorOperation();

    void
    eval(ScalarVector& destination,
         const ScalarVector& left,
         const ScalarVector& right,
         const ops::Operator& op) const override;
    void eval_inplace(
            ScalarVector& left,
            const ScalarVector& right,
            const ops::Operator& op
    ) const override;
};

template <template <typename> class Operation>
StandardBinaryVectorOperation<Operation>::StandardBinaryVectorOperation()
{}
template <template <typename> class Operation>
void StandardBinaryVectorOperation<Operation>::eval(
        ScalarVector& destination,
        const ScalarVector& left,
        const ScalarVector& right,
        const ops::Operator& op
) const
{
    const auto max_dim = std::max(left.dimension(), right.dimension());
    resize_destination(destination, max_dim);

    auto binding = GenericKernelType::new_binding();
    binding->bind(destination);
    binding->bind(left);
    binding->bind(right);
    binding->bind(op);

    eval_impl(
            GenericKernelType::get_base_name(),
            *GenericKernelType::get(),
            *binding
    );
}
template <template <typename> class Operation>
void StandardBinaryVectorOperation<Operation>::eval_inplace(
        ScalarVector& left,
        const ScalarVector& right,
        const ops::Operator& op
) const
{
    const auto max_dim = std::max(left.dimension(), right.dimension());
    resize_destination(left, max_dim);

    auto binding = GenericInplaceKernelType::new_binding();
    binding->bind(left);
    binding->bind(right);
    binding->bind(op);

    eval_impl(
            GenericInplaceKernelType::get_base_name(),
            GenericInplaceKernelType::get(),
            *binding
    );
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_VECTOR_H
