//
// Created by sam on 2/22/24.
//

#include "roughpy_module.h"

#include <sstream>

using namespace rpy;

void rpy::python::check_for_excess_arguments(const py::kwargs& kwargs)
{

    if (!kwargs.empty()) {
        std::stringstream ss;
        ss << "unrecognised keyword arguments provided:\n";
        bool first = true;
        for (auto item : kwargs) {
            if (!first) {
                ss << ", ";
                first = false;
            }
            ss << item.first;
        }

        RPY_THROW(py::key_error, ss.str().c_str());
    }
}
