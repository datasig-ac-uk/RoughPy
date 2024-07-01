// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_KERNEL_H_
#define ROUGHPY_DEVICE_KERNEL_H_

#include "core.h"

#include <roughpy/core/container/vector.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

#include "device_object_base.h"
#include "event.h"
#include "kernel_arg.h"
#include "value.h"

namespace rpy {
namespace devices {

/**
 * @class KernelLaunchParams
 * @brief Class representing the launch parameters for a kernel.
 */
class ROUGHPY_DEVICES_EXPORT KernelLaunchParams
{
    Size3 m_work_dims;
    Dim3 m_group_size;
    optional<Dim3> m_offsets;

public:
    explicit KernelLaunchParams(Size3 work_dims)
        : m_work_dims(work_dims),
          m_group_size(),
          m_offsets()
    {}

    explicit KernelLaunchParams(Size3 work_dims, Dim3 group_size)
        : m_work_dims(work_dims),
          m_group_size(group_size),
          m_offsets()
    {}

    RPY_NO_DISCARD bool has_work() const noexcept;

    RPY_NO_DISCARD Size3 total_work_dims() const noexcept;

    RPY_NO_DISCARD dimn_t total_work_size() const noexcept;

    RPY_NO_DISCARD dsize_t num_dims() const noexcept;

    RPY_NO_DISCARD Dim3 num_work_groups() const noexcept;

    RPY_NO_DISCARD Size3 underflow_of_groups() const noexcept;

    RPY_NO_DISCARD Dim3 work_groups() const noexcept;

    KernelLaunchParams();
};

namespace operators {

// The actual definitions are given in  device_support/operators.h
template <typename T>
struct Identity;
template <typename T>
struct Uminus;
template <typename T>
struct Add;
template <typename T>
struct Sub;
template <typename T>
struct LeftMultiply;
template <typename T>
struct RightMultiply;
template <typename T>
struct FusedLeftMultiplyAdd;
template <typename T>
struct FusedRightMultiplyAdd;
template <typename T>
struct FusedLeftMultiplySub;
template <typename T>
struct FusedRightMultiplySub;

class ROUGHPY_DEVICES_EXPORT Operator
{
public:
    enum class Kind
    {
        Identity = 0,
        UnaryMinus,
        Addition,
        Subtraction,
        LeftMultiply,
        RightMultiply,
        FusedLeftMultiplyAdd,
        FusedRightMultiplyAdd,
        FusedLeftMultiplySub,
        FusedRightMultiplySub
    };

    static constexpr Kind Identity = Kind::Identity;
    static constexpr Kind UnaryMinus = Kind::UnaryMinus;
    static constexpr Kind Addition = Kind::Addition;
    static constexpr Kind Subtraction = Kind::Subtraction;
    static constexpr Kind LeftMultiply = Kind::LeftMultiply;
    static constexpr Kind RightMultiply = Kind::RightMultiply;
    static constexpr Kind FusedLeftMultiplyAdd = Kind::FusedLeftMultiplyAdd;
    static constexpr Kind FusedRightMultiplyAdd = Kind::FusedRightMultiplyAdd;
    static constexpr Kind FusedLeftMultiplySub = Kind::FusedLeftMultiplySub;
    static constexpr Kind FusedRightMultiplySub = Kind::FusedRightMultiplySub;

    virtual ~Operator();

    RPY_NO_DISCARD virtual Kind kind() const noexcept = 0;
};

class ROUGHPY_DEVICES_EXPORT IdentityOperator : Operator
{
public:
    template <typename T>
    using op_type = Identity<T>;

    RPY_NO_DISCARD Kind kind() const noexcept override { return Identity; };

    ConstReference operator()(ConstReference value) const noexcept
    {
        return value;
    }
};

class ROUGHPY_DEVICES_EXPORT UnaryMinusOperator : Operator
{
public:
    template <typename T>
    using op_type = Uminus<T>;

    RPY_NO_DISCARD Kind kind() const noexcept override { return UnaryMinus; };

    ConstReference operator()(ConstReference value) const noexcept
    {
        return value;
    }
};

class ROUGHPY_DEVICES_EXPORT AdditionOperator : public Operator
{
public:
    template <typename T>
    using op_type = Add<T>;

    Kind kind() const noexcept override { return Addition; }

    ConstReference operator()(ConstReference a, ConstReference b) const noexcept
    {
        return a + b;
    }
};

class ROUGHPY_DEVICES_EXPORT SubtractionOperator : public Operator
{
public:
    template <typename T>
    using op_type = Sub<T>;

    Kind kind() const noexcept override { return Subtraction; }

    ConstReference operator()(ConstReference a, ConstReference b) const noexcept
    {
        return a - b;
    }
};

class ROUGHPY_DEVICES_EXPORT LeftMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit LeftMultiplyOperator(ConstReference value) : multiplier(value) {}

    template <typename T>
    using op_type = LeftMultiply<T>;

    Kind kind() const noexcept override { return LeftMultiply; }

    ConstReference operator()(ConstReference value) const noexcept
    {
        return multiplier * value;
    }
};

class ROUGHPY_DEVICES_EXPORT RightMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit RightMultiplyOperator(ConstReference value) : multiplier(value) {}

    template <typename T>
    using op_type = RightMultiply<T>;

    Kind kind() const noexcept override { return RightMultiply; }

    ConstReference operator()(ConstReference value) const noexcept
    {
        return value * multiplier;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedLeftMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedLeftMultiplyOperator(ConstReference value) : multiplier(value)
    {}

    template <typename T>
    using op_type = FusedLeftMultiplyAdd<T>;

    Kind kind() const noexcept override { return FusedLeftMultiplyAdd; }

    ConstReference
    operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left + multiplier * right;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedRightMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedRightMultiplyOperator(ConstReference value)
        : multiplier(value)
    {}

    template <typename T>
    using op_type = FusedRightMultiplyAdd<T>;

    Kind kind() const noexcept override { return FusedRightMultiplyAdd; }

    ConstReference
    operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left + right * multiplier;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedLeftMultiplySubOperator : public Operator
{
    ConstReference multiplier;

public:
    FusedLeftMultiplySubOperator(ConstReference value) : multiplier(value) {}

    template <typename T>
    using op_type = FusedLeftMultiplySub<T>;

    Kind kind() const noexcept override { return FusedLeftMultiplySub; }

    ConstReference
    operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left - multiplier * right;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedRightMultiplySubOperator : public Operator
{
    ConstReference multiplier;

public:
    FusedRightMultiplySubOperator(ConstReference value) : multiplier(value) {}

    template <typename T>
    using op_type = FusedRightMultiplySub<T>;

    Kind kind() const noexcept override { return FusedRightMultiplySub; }

    ConstReference
    operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left - right * multiplier;
    }
};

}// namespace operators

class ROUGHPY_DEVICES_EXPORT KernelArguments
{
public:
    virtual ~KernelArguments();

    virtual containers::Vec<void*> raw_pointers() const = 0;
    virtual Device get_device() const noexcept = 0;
    virtual Type get_type(dimn_t idx) const noexcept = 0;

    virtual void bind(Buffer buffer) = 0;
    virtual void bind(ConstReference value) = 0;
    virtual void bind(const operators::Operator& op) = 0;

    virtual void update_arg(dimn_t idx, Buffer buffer) = 0;
    virtual void update_arg(dimn_t idx, ConstReference value) = 0;
    virtual void update_arg(dimn_t idx, const operators::Operator& op) = 0;

    virtual dimn_t num_args() const noexcept = 0;
    virtual dimn_t true_num_args() const noexcept;
};

/**
 * @class KernelInterface
 * @brief Interface representing a kernel for execution on a platform.
 *
 * This interface provides methods to get the name of the kernel, the number of
 * arguments, and launching the kernel asynchronously or synchronously with
 * given arguments.
 */
class ROUGHPY_DEVICES_EXPORT KernelInterface : public dtl::InterfaceBase
{
public:
    using object_t = Kernel;

    /**
     * @brief Returns the name of the kernel.
     *
     * @return The name of the kernel.
     */
    RPY_NO_DISCARD virtual string name() const;

    /**
     * @brief Get the number of arguments required by the kernel.
     *
     * This method is a virtual member function of the KernelInterface class.
     * It returns the number of arguments required by the kernel.
     *
     * @return The number of arguments required by the kernel.
     */
    RPY_NO_DISCARD virtual dimn_t num_args() const;

    /**
     * @brief Asynchronously launches a kernel on a specified queue.
     *
     * This method asynchronously launches a kernel on the specified queue with
     * the given launch parameters and the provided kernel arguments.
     *
     * @param queue The queue on which to launch the kernel.
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     * @return An event representing the asynchronous kernel launch.
     */
    RPY_NO_DISCARD virtual Event launch_kernel_async(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) const;

    /**
     * @brief Synchronously launches a kernel on a specified queue.
     *
     * This method synchronously launches a kernel on the specified queue with
     * the given launch parameters and the provided kernel arguments.
     *
     * @param queue The queue on which to launch the kernel.
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     * @return The status of the event representing the synchronous kernel
     * launch.
     */
    virtual EventStatus launch_kernel_sync(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) const;
};

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RoughPy_Platform_EXPORTS
namespace dtl {
extern template class ObjectBase<KernelInterface, Kernel>;
}
#  else
namespace dtl {
template class RPY_DLL_IMPORT ObjectBase<KernelInterface, Kernel>;
}
#  endif
#else
namespace dtl {
extern template class ROUGHPY_DEVICES_EXPORT
        ObjectBase<KernelInterface, Kernel>;
}
#endif

/**
 * @class Kernel
 * @brief Class representing a kernel.
 * @see KernelLaunchParams
 *
 * This class represents a kernel and provides methods for launching the kernel.
 * The launch methods allow the kernel to be launched asynchronously or
 * synchronously in a specified queue with specified launch parameters and
 * arguments. The operator() can be used as a shorthand for launching the kernel
 * with a variable number of arguments.
 */
class ROUGHPY_DEVICES_EXPORT Kernel
    : public dtl::ObjectBase<KernelInterface, Kernel>
{
    using base_t = dtl::ObjectBase<KernelInterface, Kernel>;

public:
    using base_t::base_t;

    /**
     * @brief Checks if the object is a no-op (null).
     *
     * @return True if the object is a no-op (null), false otherwise.
     *
     * @note The function is_noop() is a const member function that does not
     * throw any exceptions. It returns a boolean value indicating whether the
     * object is a no-op (null) or not. The function is implemented by calling
     * the is_null() function. If the object is a no-op (null), the function
     * will return true, otherwise it will return false.
     */
    RPY_NO_DISCARD bool is_nop() const noexcept { return is_null(); }

    /**
     * @brief Retrieves the name of the kernel.
     *
     * This method returns the name of the kernel represented by the Kernel
     * object. If the underlying implementation is not set (impl() returns
     * nullptr), an empty string is returned.
     *
     * @return The name of the kernel. If the underlying implementation is not
     * set, returns an empty string.
     */
    RPY_NO_DISCARD string name() const;

    /**
     * @brief Retrieves the number of arguments of the kernel.
     *
     * This method returns the number of arguments of the kernel represented by
     * the Kernel object. If the underlying implementation is not set (impl()
     * returns nullptr), 0 is returned.
     *
     * @return The number of arguments of the kernel. If the underlying
     * implementation is not set, returns 0.
     */
    RPY_NO_DISCARD dimn_t num_args() const;

    /**
     * @brief Launches a kernel asynchronously in a queue.
     *
     * This method launches a kernel asynchronously in the specified queue with
     * the provided launch parameters and kernel arguments. If the kernel or the
     * launch parameters are not set, or if the queue is not valid for this
     * kernel, an empty Event object is returned.
     *
     * @param queue The queue in which to launch the kernel.
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     *
     * @return An Event object representing the kernel launch event. If the
     * kernel or the launch parameters are not set, or if the queue is not valid
     * for this kernel, an empty Event object is returned.
     */
    RPY_NO_DISCARD Event launch_async_in_queue(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) const;

    /**
     * @brief Launches a kernel synchronously in a queue.
     *
     * This method launches a kernel synchronously in the specified queue with
     * the provided launch parameters and kernel arguments. If the kernel or the
     * launch parameters are not set, or if the queue is not valid for this
     * kernel, an empty Event object is returned.
     *
     * @param queue The queue in which to launch the kernel.
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     *
     * @return An EventStatus indicating the status of the kernel launch. If the
     * kernel or the launch parameters are not set, or if the queue is not valid
     * for this kernel, the EventStatus::Error value is returned.
     *
     * @see EventStatus
     */
    RPY_NO_DISCARD EventStatus launch_sync_in_queue(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) const;

    /**
     * @brief Launches a kernel asynchronously in a queue.
     *
     * This method launches a kernel asynchronously in the specified queue with
     * the provided launch parameters and kernel arguments. If the kernel or the
     * launch parameters are not set, or if the queue is not valid for this
     * kernel, an empty Event object is returned.
     *
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     *
     * @return An Event object representing the kernel launch event. If the
     * kernel or the launch parameters are not set, or if the queue is not valid
     * for this kernel, an empty Event object is returned.
     */
    RPY_NO_DISCARD Event launch_async(
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) const;

    /**
     * @brief Launches a kernel synchronously with the specified launch
     * parameters and arguments.
     *
     * This method launches a kernel synchronously using the specified launch
     * parameters and arguments. The kernel launch is performed on the default
     * queue of the device associated with this kernel.
     *
     * @param params The launch parameters for the kernel.
     * @param args The arguments to be passed to the kernel.
     *
     * @return The status of the event generated by the kernel launch. Possible
     * values are:
     *         - CompletedSuccessfully: The kernel execution completed
     * successfully.
     *         - Queued: The kernel execution is queued and waiting to be
     * executed.
     *         - Submitted: The kernel execution has been submitted to the
     * device.
     *         - Running: The kernel execution is currently running on the
     * device.
     *         - Error: There was an error during the kernel execution.
     */
    RPY_NO_DISCARD EventStatus launch_sync(
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) const;

    RPY_NO_DISCARD static containers::Vec<bitmask_t>
    construct_work_mask(const KernelLaunchParams& params);

    template <typename... Args>
    void operator()(const KernelLaunchParams& params, Args&&... args) const;
};

template <typename... Args>
void Kernel::operator()(const KernelLaunchParams& params, Args&&... args) const
{
    KernelArgument kargs[] = {KernelArgument(std::forward<Args>(args))...};
    auto status = launch_sync(params, kargs);
    RPY_CHECK(status == EventStatus::CompletedSuccessfully);
}

template <typename... Args>
RPY_NO_DISCARD Event
launch_async(Kernel kernel, const KernelLaunchParams& params, Args&&... args)
{
    KernelArgument kargs[] = {KernelArgument(std::forward<Args>(args))...};
    return kernel.launch_async(params, kargs);
}

template <typename... Args>
EventStatus
launch_sync(Kernel kernel, const KernelLaunchParams& params, Args&&... args)
{
    KernelArgument kargs[] = {KernelArgument(std::forward<args>(args))...};
    return kernel.launch_sync(params, kargs);
}

template <typename... Args>
RPY_NO_DISCARD Event launch_async(
        Kernel kernel,
        Queue& queue,
        const KernelLaunchParams& params,
        Args&&... args
)
{
    KernelArgument kargs[] = {KernelArgument(std::forward<Args>(args))...};
    return kernel.launch_async_in_queue(queue, params, kargs);
}

template <typename... Args>
EventStatus launch_sync(
        Kernel kernel,
        Queue& queue,
        const KernelLaunchParams& params,
        Args&&... args
)
{
    KernelArgument kargs[] = {KernelArgument(std::forward<Args>(args))...};
    return kernel.launch_sync_in_queue(queue, params, kargs);
}

template <typename... Args>
class TypedKernel : Kernel
{
public:
    explicit TypedKernel(Kernel kernel) : Kernel(std::move(kernel)) {}

    void operator()(const KernelLaunchParams& params, Args&&... args) const
    {
        Kernel::operator()(params, std::forward<Args>(args)...);
    }
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_KERNEL_H_
