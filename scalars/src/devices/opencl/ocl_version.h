//
// Created by sam on 24/10/23.
//

#ifndef ROUGHPY_OCL_VERSION_H
#define ROUGHPY_OCL_VERSION_H

#include "ocl_headers.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <ostream>

namespace rpy {
namespace devices {

class OCLVersion
{
    cl_version m_raw;

public:
    constexpr OCLVersion() noexcept : m_raw(0) {}

    constexpr OCLVersion(cl_version raw_version) noexcept
        : m_raw(raw_version)
    {}

    constexpr OCLVersion(cl_version major, cl_version minor, cl_version patch=0) noexcept
            : m_raw(CL_MAKE_VERSION(major, minor, patch))
    {}

    explicit OCLVersion(const string& version_string);

    RPY_NO_DISCARD
    constexpr cl_version major() const noexcept {
        return CL_VERSION_MAJOR(m_raw);
    }
    RPY_NO_DISCARD
    constexpr cl_version minor() const noexcept {
        return CL_VERSION_MINOR(m_raw);
    }
    RPY_NO_DISCARD
    constexpr cl_version patch() const noexcept {
        return CL_VERSION_PATCH(m_raw);
    }

    constexpr bool operator==(const OCLVersion& other) const noexcept {
        return m_raw == other.m_raw;
    }

    constexpr bool operator<=(const OCLVersion& other) const noexcept {
        return m_raw <= other.m_raw;
    }

    constexpr bool operator<(const OCLVersion& other) const noexcept {
        return m_raw < other.m_raw;
    }

    constexpr bool operator>=(const OCLVersion& other) const noexcept {
        return m_raw >= other.m_raw;
    }

    constexpr bool operator>(const OCLVersion& other) const noexcept {
        return m_raw >= other.m_raw;
    }

};


std::ostream& operator<<(std::ostream& os, const OCLVersion&  version);


}// namespace devices
}// namespace rpy

#endif// ROUGHPY_OCL_VERSION_H
