#include "roughpy/generics/array.h"

#include "roughpy/platform/alloc.h"

using namespace rpy;
using namespace rpy::generics;

Array::Array(const Array& other)
{
    copy_from(other);
}

Array::Array(Array&& other) noexcept :
    p_type{std::move(other.p_type)},
    m_size{other.m_size},
    m_capacity{other.m_capacity},
    m_alignment{other.m_alignment},
    m_data{other.m_data}
{
    other.m_data = nullptr;
    other.m_size = 0;
    other.m_capacity = 0;
}

Array::Array(const TypePtr type, dimn_t size, std::size_t alignment) :
    p_type{type},
    m_size{size},
    m_capacity{size},
    m_alignment(alignment)
{
    if (m_size) {
        m_data = rpy::mem::aligned_alloc(m_alignment, m_capacity * p_type->object_size());
        p_type->copy_or_fill(m_data, nullptr, m_size, true);
    }
}

Array::~Array()
{
    rpy::mem::aligned_free(m_data, 0);
}

Array& Array::operator=(const Array& other)
{
    copy_from(other);
    return *this;
}

Array& Array::operator=(Array&& other) noexcept
{
    p_type = std::move(other.p_type);
    m_size = other.m_size;
    m_capacity = other.m_capacity;
    m_alignment = other.m_alignment;
    m_data = other.m_data;

    other.m_data = nullptr;
    other.m_size = 0;
    other.m_capacity = 0;

    return *this;
}

void Array::resize(dimn_t size)
{
    if (!type()) {
        throw ArrayTypeException{};
    }

    if (size < m_size) {
        // Reducing size preserves capacity
        p_type->destroy_range(ptr_at_unsafe(size), m_size - size);
    } else if (size > m_size) {
        reserve(size);

        // Init any values in new range
        p_type->copy_or_fill(
            ptr_at_unsafe(m_size),
            nullptr,
            size - m_size,
            true
        );
    }

    m_size = size;
}

void Array::reserve(dimn_t capacity)
{
    if (!type()) {
        throw ArrayTypeException{};
    }

    // Realloc and copy old data
    if (capacity > m_capacity) {
        m_capacity = capacity;
        void* new_data = rpy::mem::aligned_alloc(m_alignment, m_capacity * p_type->object_size());
        p_type->copy_or_fill(new_data, m_data, m_size, false);
        rpy::mem::aligned_free(m_data);
        m_data = new_data;
    }
}

void Array::copy_from(const Array& other)
{
    p_type = other.p_type;
    m_size = other.m_size;
    m_capacity = other.m_capacity;
    m_alignment = other.m_alignment;
    m_data = rpy::mem::aligned_alloc(m_alignment, m_capacity * p_type->object_size());
    p_type->copy_or_fill(m_data, other.m_data, m_size, true);
}

void Array::validate_idx(dimn_t idx) const
{
    if (!type()) {
        throw ArrayTypeException{};
    }

    if (idx >= m_size) {
        throw ArrayIndexException{};
    }
}
