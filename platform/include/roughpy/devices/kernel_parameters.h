
/**
 *  Kernel parameters represent the types parts of the signature of a kernel
 *  function. They are used to construct a signature object which, in turn,
 *  is used to produce an argument binding that is passed to the function
 *  itself. These parameters have either a concrete type, provided as a
 *  template argument, or a generic type, with an identifying index. These
 *  replicate templates/generics in the runtime type system. For example, a
 *  buffer containing integers would be specified as params::Buffer<int>,
 *  whereas a buffer with a generic type (label 1) would be specified by
 *  params::Buffer<params::T1>
 *
 *  Eventually, I'd like to use these objects to actually define the functions
 *  in a generic way, which we can then use in combination with a compiler to
 *  build the functions on the fly.
 */
#ifndef ROUGHPY_DEVICES_KERNEL_PARAMETERS_H
#define ROUGHPY_DEVICES_KERNEL_PARAMETERS_H

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace devices {

// Forward declarations to avoid having extra includes here.
class Value;
class Reference;
class ConstReference;
class Operator;

namespace params {

/**
 * @brief Fundamental types that are available to kernels
 *
 * This is a (not necessarily complete) list of the different kinds of
 * parameters that can appear in the signature of a kernel.
 */
enum class ParameterType : uint8_t
{
    ResultBuffer = 0,
    ArgBuffer,
    ResultValue,
    ArgValue,
    Operator
};

/**
 * @brief A generic type used as a placeholder type in kernel signatures.
 * @tparam N Unique label of the parameter
 */
template <int N>
struct GenericParam {
    static constexpr int value = N;
};

using T1 = GenericParam<1>;
using T2 = GenericParam<2>;
using T3 = GenericParam<3>;
using T4 = GenericParam<4>;
using T5 = GenericParam<5>;
using T6 = GenericParam<6>;
using T7 = GenericParam<7>;
using T8 = GenericParam<8>;
using T9 = GenericParam<9>;

/**
 * @brief A tag used to identify parameters when constructing a signature.
 */
struct IsParamBase {
};

/*
 * The first parameter definition is fully documented so as to serve as a
 * template for more parameters.
 */

template <typename T>
struct Buffer : public IsParamBase {
    /**
     *  @brief Type that appears in the signature TypeList.
     *
     *  This is eiter a C++ type (int, float, double, etc.) or a GenericType
     *  instance which is used when constructing the signature object to specify
     *  the RoughPy type of each of the parameter. This is used for type checks
     *  during argument binding. This is usually the type of the contained
     *  object, as is the case here, rather than the argument type itself.
     */
    using bind_type = T;

    /**
     * @brief The type that would appear in the C++ function signature.
     *
     * This is the actual type that should appear in the function call from the
     * C++ code. In this case, it is a RoughPy Buffer object, which indicates
     * a block of memory in RAM, on some device, or elsewhere.
     */
    using arg_type = devices::Buffer;

    /**
     * @brief The kind of the argument.
     *
     * This is used to distinguish between different parameters that might have
     * the same arg_type. For instance, here the arg_type is Buffer, but in this
     * case that buffer should be readable. Thus it has the ArgBuffer kind. A
     * later structure defines a ResultBuffer, which has the ResultBuffer kind.
     * This means the latter buffer should be writeable.
     */
    static constexpr auto kind = ParameterType::ArgBuffer;
};

template <int N>
struct Buffer<GenericParam<N>> : public IsParamBase {
    using bind_type = GenericParam<N>;
    using arg_type = devices::Buffer;
    static constexpr auto kind = ParameterType::ArgBuffer;
};

template <typename T>
struct ResultBuffer : public IsParamBase {
    using bind_type = T;
    using arg_type = devices::Buffer;
    static constexpr auto kind = ParameterType::ResultBuffer;
};

template <int N>
struct ResultBuffer<GenericParam<N>> : public IsParamBase {
    using bind_type = GenericParam<N>;
    using arg_type = devices::Buffer;
    static constexpr auto kind = ParameterType::ResultBuffer;
};

template <typename T>
struct Value : public IsParamBase {
    using bind_type = T;
    using arg_type = ConstReference;
    static constexpr auto kind = ParameterType::ArgValue;
};

template <int N>
struct Value<GenericParam<N>> : public IsParamBase {
    using bind_type = GenericParam<N>;
    using arg_type = ConstReference;
    static constexpr auto kind = ParameterType::ArgValue;
};

template <typename T>
struct ResultValue : public IsParamBase {
    using bind_type = T;
    using arg_type = Reference;
    static constexpr auto kind = ParameterType::ResultValue;
};

template <int N>
struct ResultValue<GenericParam<N>> : public IsParamBase {
    using bind_type = GenericParam<N>;
    using arg_type = Reference;
    static constexpr auto kind = ParameterType::ResultValue;
};

template <typename T>
struct Operator : public IsParamBase {
    using bind_type = T;
    using arg_type = const Operator&;
    static constexpr auto kind = ParameterType::Operator;
};

template <int N>
struct Operator<GenericParam<N>> : public IsParamBase {
    using bind_type = GenericParam<N>;
    using arg_type = const Operator&;
    static constexpr auto kind = ParameterType::Operator;
};

/*
 * Some traits that are used when constructing the signature object.
 */

template <typename T>
inline constexpr bool is_parameter = is_base_of_v<IsParamBase, T>;

template <typename T>
inline constexpr bool is_generic = false;

template <template <typename> class P, int N>
inline constexpr bool is_generic<P<GenericParam<N>>> = true;

template <typename T>
inline constexpr bool is_operator = false;

template <typename T>
inline constexpr bool is_operator<Operator<T>> = true;

template <typename... Params>
struct ParamList {
    static constexpr dimn_t size = sizeof...(Params);

    template <typename Param>
    using push_back = conditional_t<
            is_parameter<Param>,
            Param,
            ParamList<Params..., Param>>;
};

/*
 * And finally, here are all the supporting functions that we might need.
 */

constexpr bool
operator==(const ParameterType kind, const uint8_t other) noexcept
{
    return static_cast<uint8_t>(kind) == other;
}
constexpr bool
operator==(const uint8_t other, const ParameterType kind) noexcept
{
    return static_cast<uint8_t>(kind) == other;
}

constexpr bool
operator!=(const ParameterType kind, const uint8_t other) noexcept
{
    return static_cast<uint8_t>(kind) != other;
}
constexpr bool
operator!=(const uint8_t other, const ParameterType kind) noexcept
{
    return static_cast<uint8_t>(kind) != other;
}

}// namespace params
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_KERNEL_PARAMETERS_H
