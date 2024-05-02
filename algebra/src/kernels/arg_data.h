//
// Created by sam on 15/04/24.
//

#ifndef ARG_DATA_H
#define ARG_DATA_H

#include "common.h"

namespace rpy {
namespace algebra {
namespace dtl {

template <typename Derive>
class ArgData<VectorData, Derive> : Derive
{
    VectorData* p_data;

public:
    template <typename... Args>
    explicit ArgData(VectorData& data, Args&&... args)
        : Derive(std::forward<Args>(args)...),
          p_data(&data)
    {}

    const scalars::ScalarType* get_type() const
    {
        if (p_data->scalars().is_null() || p_data->scalars().type().is_null()) {
            return Derive::get_type();
        }
        return p_data->scalar_type();
    }

    void resize(dimn_t size)
    {
        if (p_data->scalar_type().is_null()) {
            p_data->mut_scalars() = scalars::ScalarArray();
        }
        if (p_data->empty() || p_data->size() < size) { p_data->resize(size); }
    }

    RPY_NO_DISCARD string get_suffix() const noexcept
    {
        return (!p_data->keys().empty() ? 'S' : 'D') + Derive::get_suffix();
    }

    template <typename F>
    decltype(auto) eval_device(F&& func)
    {
        return Derive::eval_device([func, this](auto&&... next_arg) {
            const auto size = p_data->size();
            auto scalars = p_data->mut_scalars()[{0, size}].mut_buffer();
            if (p_data->keys().empty()) {
                // desnse data;
                return func(
                        scalars,
                        std::forward<decltype(next_arg)>(next_arg)...
                );
            }

            const auto keys = p_data->mut_keys()[{0, size}].mut_buffer();
            return func(
                    keys,
                    scalars,
                    std::forward<decltype(next_arg)>(next_arg)...
            );
        });
    }

    template <typename F>
    decltype(auto) eval_host(F&& func)
    {
        return Derive::eval_host([func, this](auto&&... next_arg) {
            const auto size = p_data->size();
            auto scalars
                    = p_data->mut_scalars().mut_view()[{0, size}].mut_buffer();
            if (p_data->keys().empty()) {
                // desnse data;
                return func(
                        scalars,
                        std::forward<decltype(next_arg)>(next_arg)...
                );
            }

            const auto keys
                    = p_data->mut_keys().mut_view()[{0, size}].mut_buffer();
            return func(
                    keys,
                    scalars,
                    std::forward<decltype(next_arg)>(next_arg)...
            );
        });
    }

    template <typename F>
    decltype(auto) eval_generic(F&& func)
    {
        return Derive::eval_generic([f = std::forward<F>(func),
                                     this](auto&&... next_arg) {
            return f(*p_data, std::forward<decltype(next_arg)>(next_arg)...);
        });
    }

    RPY_NO_DISCARD dimn_t size() const noexcept
    {
        return std::max(p_data->size(), Derive::size());
    }

    RPY_NO_DISCARD devices::Device get_device() const noexcept
    {
        std::array<devices::Device, 2> devices{
                Derive::get_device(),
                p_data->scalars().device()
        };

        return devices::get_best_device(devices);
    }

    RPY_NO_DISCARD string_view get_type_id() const noexcept
    {
        return p_data->scalar_type()->id();
    }
};

template <typename Derive>
class ArgData<const VectorData, Derive> : Derive
{
    const VectorData* p_data;

public:
    using Derive::get_type_id;

    template <typename... Args>
    explicit ArgData(const VectorData& data, Args&&... args)
        : Derive(std::forward<Args>(args)...),
          p_data(&data)
    {}

    using Derive::resize;

    RPY_NO_DISCARD const scalars::ScalarType* get_type() const
    {
        return nullptr;
    }

    RPY_NO_DISCARD string get_suffix() const noexcept
    {
        return (!p_data->keys().empty() ? 's' : 'd') + Derive::get_suffix();
    }

    template <typename F>
    decltype(auto) eval_device(F&& func)
    {
        return Derive::eval_device([f = std::forward<F>(func),
                                    this](auto&&... next_arg) {
            const auto size = p_data->size();
            auto scalars = p_data->scalars()[{0, size}].buffer();
            if (p_data->keys().empty()) {
                // desnse data;
                return f(
                        scalars,
                        std::forward<decltype(next_arg)>(next_arg)...
                );
            }

            const auto keys = p_data->keys()[{0, size}].buffer();
            return f(
                    keys,
                    scalars,
                    std::forward<decltype(next_arg)>(next_arg)...
            );
        });
    }

    template <typename F>
    decltype(auto) eval_host(F&& func)
    {
        return Derive::eval_host([f = std::forward<F>(func),
                                  this](auto&&... next_arg) {
            const auto size = p_data->size();
            auto scalars = p_data->scalars().view()[{0, size}].buffer();
            if (p_data->keys().empty()) {
                // desnse data;
                return f(
                        scalars,
                        std::forward<decltype(next_arg)>(next_arg)...
                );
            }

            const auto keys = p_data->keys().view()[{0, size}].buffer();
            return f(
                    keys,
                    scalars,
                    std::forward<decltype(next_arg)>(next_arg)...
            );
        });
    }

    template <typename F>
    decltype(auto) eval_generic(F&& func)
    {
        return Derive::eval_generic([f = std::forward<F>(func),
                                     this](auto&&... next_arg) {
            return f(*p_data, std::forward<decltype(next_arg)>(next_arg)...);
        });
    }

    RPY_NO_DISCARD dimn_t size() const noexcept
    {
        return std::max(p_data->size(), Derive::size());
    }

    RPY_NO_DISCARD devices::Device get_device() const noexcept
    {
        std::array<devices::Device, 2> devices{
                Derive::get_device(),
                p_data->scalars().device()
        };

        return devices::get_best_device(devices);
    }
};

template <typename Derive>
class ArgData<scalars::Scalar, Derive> : Derive
{
    scalars::Scalar* p_data;

public:
    template <typename... Args>
    explicit ArgData(scalars::Scalar& data, Args&&... args)
        : Derive(std::forward<Args>(args)...),
          p_data(&data)
    {}

    using Derive::get_type;
    using Derive::get_type_id;
    using Derive::resize;

    RPY_NO_DISCARD string get_suffix() const noexcept
    {
        return 'V' + Derive::get_suffix();
    }

    template <typename F>
    decltype(auto) eval_device(F&& func)
    {
        return Derive::eval_device([f = std::forward<F>(func),
                                    this](auto&&... next_arg) {
            return f(
                    to_kernel_arg(*p_data),
                    std::forward<decltype(next_arg)>(next_arg)...
            );
        });
    }

    template <typename F>
    decltype(auto) eval_host(F&& func)
    {
        return Derive::eval_host([f = std::forward<F>(func),
                                  this](auto&&... next_arg) {
            return f(
                    to_kernel_arg(*p_data),
                    std::forward<decltype(next_arg)>(next_arg)...
            );
        });
    }

    template <typename F>
    decltype(auto) eval_generic(F&& func)
    {
        return Derive::eval_generic([f = std::forward<F>(func),
                                     this](auto&&... next_arg) {
            return f(*p_data, std::forward<decltype(next_arg)>(next_arg)...);
        });
    }

    RPY_NO_DISCARD dimn_t size() const noexcept { return Derive::size(); }

    RPY_NO_DISCARD devices::Device get_device() const noexcept
    {
        return Derive::get_device();
    }
};

template <typename Derive>
class ArgData<const scalars::Scalar, Derive> : Derive
{
    const scalars::Scalar* p_data;

public:
    template <typename... Args>
    explicit ArgData(const scalars::Scalar& data, Args&&... args)
        : Derive(std::forward<Args>(args)...),
          p_data(&data)
    {}

    using Derive::get_type;
    using Derive::get_type_id;
    using Derive::resize;

    RPY_NO_DISCARD string get_suffix() const noexcept
    {
        return 'v' + Derive::get_suffix();
    }

    template <typename F>
    decltype(auto) eval_device(F&& func)
    {
        return Derive::eval_device([f = std::forward<F>(func),
                                    this](auto&&... next_arg) {
            return f(
                    to_kernel_arg(*p_data),
                    std::forward<decltype(next_arg)>(next_arg)...
            );
        });
    }

    template <typename F>
    decltype(auto) eval_host(F&& func)
    {
        return Derive::eval_host([f = std::forward<F>(func),
                                  this](auto&&... next_arg) {
            return f(
                    to_kernel_arg(*p_data),
                    std::forward<decltype(next_arg)>(next_arg)...
            );
        });
    }

    template <typename F>
    decltype(auto) eval_generic(F&& func)
    {
        return Derive::eval_generic([f = std::forward<F>(func),
                                     this](auto&&... next_arg) {
            return f(*p_data, std::forward<decltype(next_arg)>(next_arg)...);
        });
    }

    RPY_NO_DISCARD dimn_t size() const noexcept { return Derive::size(); }

    RPY_NO_DISCARD devices::Device get_device() const noexcept
    {
        return Derive::get_device();
    }
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// ARG_DATA_H
