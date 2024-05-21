//
// Created by sam on 4/26/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H
#define ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H

#include "algorithms.h"

#include <roughpy/core/errors.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/host_device.h>
#include <roughpy/devices/type.h>

namespace rpy {
namespace devices {

namespace dtl {
template <typename T>
struct IDAndNameOfFType;

template <typename... Ts>
struct TypeList {
};
}// namespace dtl

/**
 * @class FundamentalType
 * @brief A class that represents a fundamental data type.
 *
 * The FundamentalType class is a subclass of the Type class and represents a
 * fundamental data type. It provides methods to get the unique ID and name of
 * the type, as well as the type information and traits.
 */
template <typename T>
class RPY_LOCAL FundamentalType : public Type
{

public:
    FundamentalType(string_view id, string_view name)
        : Type(id, name, devices::type_info<T>(), devices::traits_of<T>())
    {
        // #ifndef RPY_NO_RTTI
        //         register_type(typeid(T), this);
        // #endif
        const auto device = get_host_device();
        device->register_algorithm_drivers<HostDriversImpl, T, T>();
    }

    /**
     * @brief Returns a pointer to the static instance of FundamentalType<T>
     * with the specified type T.
     *
     * @tparam T The type for which the FundamentalType instance is returned.
     *
     * @return A pointer to the static instance of FundamentalType<T>.
     *
     * @note The returned pointer is valid throughout the program execution and
     * should not be deleted.
     * @note The returned pointer is guaranteed to be non-null.
     * @note The returned pointer may be used to access the ID and name of the
     * FundamentalType<T>.
     */
    RPY_NO_DISCARD static const FundamentalType* get() noexcept;

    void display(std::ostream& os, const void* ptr) const override;
};

template <typename T>
const FundamentalType<T>* FundamentalType<T>::get() noexcept
{
    using IDName = dtl::IDAndNameOfFType<T>;
    static const FundamentalType type(IDName::id, IDName::name);
    return &type;
}

template <typename T>
void FundamentalType<T>::display(std::ostream& os, const void* ptr) const
{
    os << *static_cast<const T*>(ptr);
}

namespace dtl {
using FundamentalTypesList = TypeList<
        int8_t,
        int16_t,
        int32_t,
        int64_t,
        uint8_t,
        uint16_t,
        uint32_t,
        uint64_t,
        float,
        double>;

template <typename S, typename T>
struct NotImplemented {
    static void func(void*, const void*)
    {
        RPY_THROW(
                std::runtime_error,
                string_cat(
                        "operation is not implemented for types ",
                        FundamentalType<S>::get()->id(),
                        " and ",
                        FundamentalType<T>::get()->id()
                )
        );
    }
};

#define RPY_DEFINE_OP_OVERRIDE(name, op)                                       \
    template <typename S, typename T, typename SFINAE = void>                  \
    struct name : NotImplemented<S, T> {                                       \
    };                                                                         \
    template <typename S, typename T>                                          \
    struct name<                                                               \
            S,                                                                 \
            T,                                                                 \
            void_t<decltype(std::declval<S&>() op std::declval<const T&>()     \
            )>> {                                                              \
        static void func(void* left, const void* right)                        \
        {                                                                      \
            *static_cast<S*>(left) op* static_cast<const T*>(right);           \
        }                                                                      \
    };

RPY_DEFINE_OP_OVERRIDE(AddInplace, +=)
RPY_DEFINE_OP_OVERRIDE(SubInplace, -=)
RPY_DEFINE_OP_OVERRIDE(MulInplace, *=)
RPY_DEFINE_OP_OVERRIDE(DivInplace, /=)

#undef RPY_DEFINE_OP_OVERRIDE

#define RPY_DEFINE_OP_OVERRIDE(name, op)                                         \
    template <typename S, typename T, typename SFINAE = void>                    \
    struct name {                                                                \
        static bool func(const void*, const void*)                               \
        {                                                                        \
            RPY_THROW(std::runtime_error,                                      \
                string_cat("operation " RPY_STRINGIFY(op)                      \
                         " is not defined for types ",                         \
                FundamentalType<S>::get()->id(), " and ",                      \
                FundamentalType<T>::get()->id())); \
        }                                                                        \
    };                                                                           \
    template <typename S, typename T>                                            \
    struct name<                                                                 \
            S,                                                                   \
            T,                                                                   \
            void_t<decltype(std::declval<const S&>()                             \
                                    op std::declval<const T&>())>> {             \
        static bool func(const void* left, const void* right)                    \
        {                                                                        \
            return *static_cast<const S*>(left) op                               \
                    * static_cast<const T*>(right);                              \
        }                                                                        \
    };

RPY_DEFINE_OP_OVERRIDE(CompareEqual, ==)
RPY_DEFINE_OP_OVERRIDE(CompareLess, <)
RPY_DEFINE_OP_OVERRIDE(CompareLessEqual, <=)
RPY_DEFINE_OP_OVERRIDE(CompareGreater, >)
RPY_DEFINE_OP_OVERRIDE(CompareGreaterEqual, >=)

#undef RPY_DEFINE_OP_OVERRIDE

template <typename S, typename T, bool = is_convertible_v<const T&, S>>
struct Convert {
    static void func(void* out, const void* in)
    {
        (*static_cast<S*>(out)) = static_cast<S>(*static_cast<const T*>(in));
    }
};
template <typename S, typename T>
struct Convert<S, T, false> : NotImplemented<S, T> {
};

template <typename S, typename T>
struct SupportRegistration {

    static void register_support(const Type* type)
    {
        const auto* other_type = FundamentalType<T>::get();
        auto support = type->update_support(other_type);

        support->arithmetic.add_inplace = AddInplace<S, T>::func;
        support->arithmetic.sub_inplace = SubInplace<S, T>::func;
        support->arithmetic.mul_inplace = MulInplace<S, T>::func;
        support->arithmetic.div_inplace = DivInplace<S, T>::func;

        support->comparison.equals = CompareEqual<S, T>::func;
        support->comparison.less = CompareLess<S, T>::func;
        support->comparison.less_equal = CompareLessEqual<S, T>::func;
        support->comparison.greater = CompareGreater<S, T>::func;
        support->comparison.greater_equal = CompareGreaterEqual<S, T>::func;

        support->conversions.convert = Convert<S, T>::func;
    }
};

template <typename ThisT, typename T>
void register_type_support(const Type* type, TypeList<T>)
{
    using reg = SupportRegistration<ThisT, T>;
    reg::register_support(type);
}

template <typename ThisT, typename T, typename... Ts>
void register_type_support(const Type* type, TypeList<T, Ts...>)
{
    using reg = SupportRegistration<ThisT, T>;
    reg::register_support(type);
    register_type_support<ThisT>(type, TypeList<Ts...>());
}

}// namespace dtl

template <typename ThisT>
void register_all_supports()
{
    const auto* tp = FundamentalType<ThisT>::get();
    dtl::register_type_support<ThisT>(tp, dtl::FundamentalTypesList());
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H
