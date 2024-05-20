//
// Created by sam on 5/16/24.
//

#ifndef ROUGHPY_DEVICES_TYPE_DISPATCHER_H
#define ROUGHPY_DEVICES_TYPE_DISPATCHER_H

#include <roughpy/core/container/unordered_map.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/smart_ptr.h>
#include <roughpy/core/strings.h>
#include <roughpy/core/sync.h>

#include "type.h"

namespace rpy {
namespace devices {

namespace dtl {

template <typename T>
using StringViewifyType = string_view;

template <typename... Ts>
using StringViewTupleify = std::tuple<StringViewifyType<Ts>...>;

template <typename... Ts, size_t... Is>
auto to_type_ids_impl(
        const std::tuple<Ts...>& arg,
        std::integer_sequence<size_t, Is...> is
) noexcept
{
    return std::make_tuple(std::get<Is>(arg)->id()...);
}

template <typename... Ts>
StringViewTupleify<Ts...> to_type_ids(const std::tuple<Ts...>& arg) noexcept
{
    return to_type_ids_impl(arg, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
using TypePtrify = const Type*;

}// namespace dtl

template <typename DispatchedType, typename... Args>
class TypeDispatcher : public RcBase<TypeDispatcher<DispatchedType, Args...>>
{
    using dispatched_type = DispatchedType;
    using dispatched_ptr = std::unique_ptr<const DispatchedType>;
    using index_type = dtl::StringViewTupleify<Args...>;
    using lock_type = std::mutex;
    using cache_type = containers::HashMap<index_type, dispatched_ptr>;

    GuardedValue<cache_type, lock_type> m_cache;

public:
    RPY_NO_DISCARD bool supports_types(dtl::TypePtrify<Args>... types
    ) const noexcept;

    RPY_NO_DISCARD const dispatched_type&
    get_implementor(dtl::TypePtrify<Args>... types) const;

    template <template <typename...> class Implementor, typename... Ts>
    void register_implementation();
};

template <typename DispatchedType, typename... Args>
bool TypeDispatcher<DispatchedType, Args...>::supports_types(
        dtl::TypePtrify<Args>... types
) const noexcept
{
    return m_cache->contains(std::make_tuple(types->id()...));
}

template <typename DispatchedType, typename... Args>
const typename TypeDispatcher<DispatchedType, Args...>::dispatched_type&
TypeDispatcher<DispatchedType, Args...>::get_implementor(
        dtl::TypePtrify<Args>... types
) const
{
    auto cache = *m_cache;
    auto it = cache->find(std::make_tuple(types->id()...));
    if (it != cache->end()) { return *it->second; }

    RPY_THROW(
            std::runtime_error,
            string_cat(
                    "type combination is not supported: ",
                    string_join(", ", types->id()...)
            )
    );
}

template <typename DispatchedType, typename... Args>
template <template <typename...> class Implementor, typename... Ts>
void TypeDispatcher<DispatchedType, Args...>::register_implementation()
{
    auto types = std::make_tuple(get_type<Ts>()...);
    auto index = dtl::to_type_ids(types);

    auto cache = *m_cache;
    auto& entry = (*cache)[index];
    if (entry == nullptr) {
        entry = dispatched_ptr(new Implementor<Ts...>(types));
    }
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_TYPE_DISPATCHER_H
