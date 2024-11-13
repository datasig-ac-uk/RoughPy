//
// Created by sam on 10/01/24.
//

#ifndef ROUGHPY_PLATFORM_ERRORS_H
#define ROUGHPY_PLATFORM_ERRORS_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "roughpy_platform_export.h"
#include <boost/stacktrace.hpp>


namespace rpy {
namespace errors {

ROUGHPY_PLATFORM_EXPORT
string format_error_message(string_view user_message, const char* filename, int lineno, const char* func, const boost::stacktrace::stacktrace& st);



}// namespace errors
}// namespace rpy

#ifndef __CLION_IDE__

#else

#  define RPY_CHECK(...)
#  define RPY_THROW(...)

#endif

#endif// ROUGHPY_PLATFORM_ERRORS_H
