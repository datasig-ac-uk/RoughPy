//
// Created by sammorley on 29/11/24.
//

#ifndef ROUGHPY_CORE_META_H
#define ROUGHPY_CORE_META_H


namespace rpy::meta {


template <typename... Ts>
struct TypeList
{
    static constexpr size_t size = sizeof...(Ts);

    template <typename... Us>
    using Append = TypeList<Ts..., Us...>;

    template <typename... Us>
    using Prepend = TypeList<Us..., Ts...>;

};



}


#endif //ROUGHPY_CORE_META_H
