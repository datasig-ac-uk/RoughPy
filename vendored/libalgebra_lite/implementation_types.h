//
// Created by user on 23/07/22.
//

#ifndef LIBALGEBRA_LITE_IMPLEMENTATION_TYPES_H
#define LIBALGEBRA_LITE_IMPLEMENTATION_TYPES_H

#include <cstddef>
#include <cstdint>
#include <utility>


#include "detail/macros.h"

#ifdef LAL_USE_LIBAGEBRA
#include <libalgebra/libalgebra.h>
#endif


namespace lal {

using dimn_t = std::size_t;
using idimn_t = std::ptrdiff_t;
using deg_t = std::int32_t;

using let_t = std::size_t;

using std::pair;


} // namespace lal



#ifdef _WIN32
#ifdef Libalgebra_Lite_EXPORTS
#define LAL_EXPORT_TEMPLATE_CLASS(TMPL, ...) \
    extern template class TMPL<__VA_ARGS__>;
#define LAL_EXPORT_TEMPLATE_STRUCT(TMPL, ...) \
    extern template struct TMPL<__VA_ARGS__>;
#else
#define LAL_EXPORT_TEMPLATE_CLASS(TMPL, ...) \
    template class LIBALGEBRA_LITE_EXPORT TMPL<__VA_ARGS__>;
#define LAL_EXPORT_TEMPLATE_STRUCT(TMPL, ...) \
    template struct LIBALGEBRA_LITE_EXPORT TMPL<__VA_ARGS__>;
#endif
#else
#define LAL_EXPORT_TEMPLATE_CLASS(TMPL, ...) \
    extern template class LIBALGEBRA_LITE_EXPORT TMPL<__VA_ARGS__>;
#define LAL_EXPORT_TEMPLATE_STRUCT(TMPL, ...) \
    extern template struct LIBALGEBRA_LITE_EXPORT TMPL<__VA_ARGS__>;
#endif


#endif //LIBALGEBRA_LITE_IMPLEMENTATION_TYPES_H
