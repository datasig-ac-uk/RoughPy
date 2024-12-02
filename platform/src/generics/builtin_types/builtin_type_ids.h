//
// Created by sammorley on 17/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_IDS_H
#define ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_IDS_H

#include "roughpy/core/types.h"

#include "roughpy/generics/type_ptr.h"

namespace rpy::generics {


// Specializations for integer types
template <>
inline constexpr string_view type_id_of<int8_t> = "i8";
template <>
inline constexpr string_view type_id_of<int16_t> = "i16";
template <>
inline constexpr string_view type_id_of<int32_t> = "i32";
template <>
inline constexpr string_view type_id_of<int64_t> = "i64";

// Specializations for unsigned integer types
template <>
inline constexpr string_view type_id_of<uint8_t> = "u8";
template <>
inline constexpr string_view type_id_of<uint16_t> = "u16";
template <>
inline constexpr string_view type_id_of<uint32_t> = "u32";
template <>
inline constexpr string_view type_id_of<uint64_t> = "u64";

// Specializations for floating point types
template <>
inline constexpr string_view type_id_of<float> = "f32";
template <>
inline constexpr string_view type_id_of<double> = "f64";




}


#endif //ROUGHPY_GENERICS_INTERNAL_BUILTIN_TYPE_IDS_H
