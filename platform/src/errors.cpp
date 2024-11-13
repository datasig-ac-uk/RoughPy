//
// Created by sam on 10/01/24.
//

#include "errors.h"

#include <stdexcept>
#include <string_view>

#include <roughpy/core/check.h>



namespace rpy::errors {


template
ROUGHPY_PLATFORM_EXPORT void throw_exception<std::runtime_error>(std::string_view user_msg,
  const char* filename,
  int lineno,
  const char* func
);

template ROUGHPY_PLATFORM_EXPORT void throw_exception<std::logic_error>(std::string_view user_msg,
  const char* filename,
  int lineno,
  const char* func);

template ROUGHPY_PLATFORM_EXPORT void throw_exception<std::invalid_argument>(std::string_view user_msg,
  const char* filename,
  int lineno,
  const char* func);

template ROUGHPY_PLATFORM_EXPORT void throw_exception<std::domain_error>(std::string_view user_msg,
  const char* filename,
  int lineno,
  const char* func);

template ROUGHPY_PLATFORM_EXPORT void throw_exception<std::out_of_range>(std::string_view user_msg,
  const char* filename,
  int lineno,
  const char* func);



}

