#ifndef ROUGHPY_GENERICS_MOCKING_MOCK_TYPE_H
#define ROUGHPY_GENERICS_MOCKING_MOCK_TYPE_H

#include <gmock/gmock.h>

#include "roughpy/generics/type.h"

namespace rpy::generics {
namespace mocking {

class ROUGHPY_PLATFORM_EXPORT MockType : public rpy::mem::RefCountedMiddle<Type>
{
public:
    virtual ~MockType();

    MOCK_METHOD(const std::type_info&, type_info, (), (const, noexcept, override));
    MOCK_METHOD(BasicProperties, basic_properties, (), (const, noexcept, override));
    MOCK_METHOD(size_t, object_size, (), (const, noexcept, override));
    MOCK_METHOD(string_view, name, (), (const, noexcept, override));
    MOCK_METHOD(string_view, id, (), (const, noexcept, override));
    MOCK_METHOD(void*, allocate_object, (), (const, override));
    MOCK_METHOD(void, free_object, (void*), (const, override));
    MOCK_METHOD(void, copy_or_fill, (void*, const void*, size_t, bool), (const, override));
    MOCK_METHOD(void, destroy_range, (void*, size_t), (const, override));
    MOCK_METHOD(const std::ostream&, display, (std::ostream&, const void*), (const, override));
};

} // namespace mocking
} // namespace rpy::generics

#endif // ROUGHPY_GENERICS_MOCKING_MOCK_TYPE_H
