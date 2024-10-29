//
// Created by sam on 10/01/24.
//

#include "errors.h"

#include <sstream>

using namespace rpy;

string errors::format_error_message(
        string_view user_message,
        const char* filename,
        int lineno,
        const char* func
)
{
    std::stringstream ss;
    ss << user_message << " at lineno " << lineno << " in " << filename
       << " in function " << func << '\n'
       << '\n';
    return ss.str();
}