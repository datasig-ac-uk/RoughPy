//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_GENERICS_TYPE_PTR_H
#define ROUGHPY_GENERICS_TYPE_PTR_H

#include <roughpy/core/smart_ptr.hpp>
#include "roughpy/core/meta.hpp"
#include "roughpy/core/types.hpp"

#include <roughpy/platform/reference_counting.h>

namespace rpy::generics {

class Type;

using TypePtr = Rc<const Type>;


template <typename T>
inline constexpr string_view type_id_of;


using BuiltinTypesList = meta::TypeList<
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t,
    float,
    double
    >;



void intrusive_ptr_add_ref(const Type* ptr) noexcept;
void intrusive_ptr_release(const Type* ptr) noexcept;



}

#endif //ROUGHPY_GENERICS_TYPE_PTR_H
