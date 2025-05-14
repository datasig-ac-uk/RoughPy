#ifndef ROUGHPY_GENERICS_ARRAY_H
#define ROUGHPY_GENERICS_ARRAY_H

#include "roughpy/generics/values.h"

namespace rpy::generics {

class ROUGHPY_PLATFORM_EXPORT ArrayIndexException : public std::exception
{
};

class ROUGHPY_PLATFORM_EXPORT ArrayTypeException : public std::exception
{
};

/**
 * @brief FIXME
 * - Documentation
 * - note It's valid to not have type for size = 0
 * - review alignment being stored in array, add accessor for tests?
 */
class ROUGHPY_PLATFORM_EXPORT Array
{
private:
    TypePtr p_type;
    dimn_t m_size = 0;
    dimn_t m_capacity = 0;
    std::size_t m_alignment = alignof(void*);
    void* m_data = nullptr;

public:
    Array() = default;

    Array(const Array& other);
    Array(Array&& other) noexcept;

    explicit Array(const TypePtr type, dimn_t size = 0, std::size_t alignment = alignof(std::max_align_t));

    ~Array();

    Array& operator=(const Array& other);
    Array& operator=(Array&& other) noexcept;

    void resize(dimn_t size);
    void reserve(dimn_t capacity);

    RPY_NO_DISCARD
    const Type* type() const noexcept
    {
        return p_type.get();
    }

    RPY_NO_DISCARD
    constexpr dimn_t capacity() const noexcept
    {
        return m_capacity;
    }

    RPY_NO_DISCARD
    constexpr dimn_t size() const noexcept
    {
        return m_size;
    }

    RPY_NO_DISCARD
    constexpr bool empty() const noexcept
    {
        return m_size == 0;
    }

    RPY_NO_DISCARD
    const void* data() const noexcept
    {
        return m_data;
    }

    RPY_NO_DISCARD
    void* data() noexcept
    {
        return m_data;
    }

    RPY_NO_DISCARD
    ConstRef operator[](dimn_t idx) const
    {
        validate_idx(idx);
        ConstRef result{type(), ptr_at_unsafe(idx)};
        return result;
    }

    RPY_NO_DISCARD
    Ref operator[](dimn_t idx)
    {
        validate_idx(idx);
        Ref result{type(), ptr_at_unsafe(idx)};
        return result;
    }

    RPY_NO_DISCARD
    std::optional<ConstRef> get(dimn_t idx) const noexcept
    {
        std::optional<ConstRef> result;
        if (type() && (idx < m_size)) {
            result = get_unchecked(idx);
        }
        return result;
    }

    RPY_NO_DISCARD
    std::optional<Ref> get_mut(dimn_t idx) noexcept
    {
        std::optional<Ref> result;
        if (type() && (idx < m_size)) {
            result = get_unchecked_mut(idx);
        }
        return result;
    }

    RPY_NO_DISCARD
    inline ConstRef get_unchecked(dimn_t idx) const
    {
        return ConstRef(&*p_type, ptr_at_unsafe(idx));
    }

    RPY_NO_DISCARD
    inline Ref get_unchecked_mut(dimn_t idx)
    {
        return Ref(&*p_type, ptr_at_unsafe(idx));
    }

private:
    void copy_from(const Array& other);
    void validate_idx(dimn_t idx) const;

    RPY_NO_DISCARD
    inline std::uintptr_t ptr_offset_unsafe(dimn_t idx) const
    {
        std::uintptr_t offset = idx * p_type->object_size();
        return offset;
    }

    RPY_NO_DISCARD
    inline const void* ptr_at_unsafe(dimn_t idx) const
    {
        const char* byte_ptr = reinterpret_cast<const char*>(data());
        const void* ptr = byte_ptr + ptr_offset_unsafe(idx);
        return ptr;
    }

    RPY_NO_DISCARD
    inline void* ptr_at_unsafe(dimn_t idx)
    {
        char* byte_ptr = reinterpret_cast<char*>(data());
        void* ptr = byte_ptr + ptr_offset_unsafe(idx);
        return ptr;
    }
};

} // namespace rpy::generics

#endif // ROUGHPY_GENERICS_ARRAY_H