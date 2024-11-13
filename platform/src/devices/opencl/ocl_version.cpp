//
// Created by sam on 24/10/23.
//

#include "ocl_version.h"

#include <ostream>
#include <regex>
#include <stdexcept>
#include <string>

#include "roughpy/core/check.h"  // for throw_exception, RPY_CHECK
#include "roughpy/core/types.h"  // for string

#include <roughpy/platform/errors.h>

using namespace rpy;
using namespace devices;

OCLVersion::OCLVersion(const string& version_string)
{
    static const std::regex s_ocl_version_regex(
            R"rgx(([0-9]+)[.]([0-9]+))rgx",
            std::regex::ECMAScript
    );

    std::smatch match;
    std::regex_search(version_string, match, s_ocl_version_regex);
    if (!match.empty()) {
        RPY_CHECK(
                match.size() == 3,
                "string did not match an OpenCL version string  " + match.str()
        );

        m_raw = CL_MAKE_VERSION(
                std::stoi(match[1].str()),
                std::stoi(match[2].str()),
                0
        );

    } else {
        RPY_THROW(
                std::invalid_argument,
                "string '" + version_string
                        + "' does not match an OpenCL version string"
        );
    }
}

std::ostream& rpy::devices::operator<<(std::ostream& os, const OCLVersion& version)
{
    return os << version.major() << '.'
              << version.minor() << '.'
              << version.patch();
}
