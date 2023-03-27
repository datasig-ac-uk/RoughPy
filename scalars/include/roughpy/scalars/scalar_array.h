#ifndef ROUGHPY_SCALARS_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_SCALAR_ARRAY_H_

#include "scalars_fwd.h"
#include "roughpy_scalars_export.h"

#include <cassert>



#include "scalar_pointer.h"

namespace rpy { namespace scalars {

class ROUGHPY_SCALARS_EXPORT ScalarArray : public ScalarPointer {

protected:
    dimn_t m_size = 0;

public:

    ScalarArray() = default;

    explicit ScalarArray(const ScalarType *type)
        : ScalarPointer(type)
    {}
    ScalarArray(const ScalarType *type, void* data, dimn_t size)
        : ScalarPointer(type, data), m_size(size)
    {}
    ScalarArray(const ScalarType *type, const void* data, dimn_t size)
        : ScalarPointer(type, data), m_size(size)
    {}
    ScalarArray(ScalarPointer begin, dimn_t size)
        : ScalarPointer(begin), m_size(size) {}

    ScalarArray(const ScalarArray &other) = default;
    ScalarArray(ScalarArray &&other) noexcept;
    ScalarArray &operator=(const ScalarArray &other) = default;
    ScalarArray &operator=(ScalarArray &&other) noexcept;


    constexpr dimn_t size() const noexcept { return m_size; }
};
}}

#endif // ROUGHPY_SCALARS_SCALAR_ARRAY_H_
