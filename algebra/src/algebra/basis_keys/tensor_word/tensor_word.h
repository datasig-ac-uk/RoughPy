//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_TENSOR_WORD_H
#define ROUGHPY_TENSOR_WORD_H

#include <boost/container/small_vector.hpp>

#include <roughpy/core/container/vector.h>
#include <roughpy/platform/alloc.h>

#include <initializer_list>

namespace rpy {
namespace algebra {

namespace dtl {

using TensorLetterType = let_t;
inline constexpr dimn_t tensor_word_inline_size
        = sizeof(void*) / sizeof(TensorLetterType);

}// namespace dtl

class TensorWord
    : public platform::SmallObjectBase,
      containers::SmallVec<dtl::TensorLetterType, dtl::tensor_word_inline_size>
{
    using container_t = containers::
            SmallVec<dtl::TensorLetterType, dtl::tensor_word_inline_size>;

public:
    using typename container_t::const_iterator;
    using typename container_t::iterator;

    TensorWord() = default;
    TensorWord(const TensorWord&) = default;
    TensorWord(TensorWord&&) noexcept = default;

    template <typename I, typename = enable_if_t<is_integral_v<I>>>
    TensorWord(std::initializer_list<I> args)
    {
        reserve(args.size());
        for (auto&& arg : args) { emplace_back(arg); }
    }

    TensorWord(const TensorWord& left, const TensorWord& right);

    TensorWord& operator=(const TensorWord&) = default;
    TensorWord& operator=(TensorWord&&) noexcept = default;

    using container_t::begin;
    using container_t::end;

    RPY_NO_DISCARD bool is_letter() const noexcept { return size() == 1; }
    RPY_NO_DISCARD let_t get_letter() const noexcept
    {
        RPY_DBG_ASSERT(is_letter());
        return *begin();
    }

    RPY_NO_DISCARD deg_t degree() const noexcept { return size(); }

    RPY_NO_DISCARD optional<TensorWord> left_parent() const noexcept;
    RPY_NO_DISCARD optional<TensorWord> right_parent() const noexcept;

    void print(std::ostream& os) const;

    RPY_NO_DISCARD deg_t min_alphabet_size() const noexcept;
};

hash_t hash_value(const TensorWord& tensor_word);

inline TensorWord operator*(const TensorWord& left, const TensorWord& right)
{
    return {left, right};
}

inline TensorWord operator*(let_t left, const TensorWord& right)
{
    return {TensorWord{left}, right};
}

inline TensorWord operator*(const TensorWord& left, let_t right)
{
    return {left, TensorWord{right}};
}

bool operator==(const TensorWord& left, const TensorWord& right) noexcept;

inline bool operator!=(const TensorWord& left, const TensorWord& right) noexcept
{
    return !(left == right);
}

bool operator<(const TensorWord& left, const TensorWord& right) noexcept;
inline bool operator>(const TensorWord& left, const TensorWord& right) noexcept
{
    return right < left;
}
inline bool operator<=(const TensorWord& left, const TensorWord& right) noexcept
{
    return !(left > right);
}
inline bool operator>=(const TensorWord& left, const TensorWord& right) noexcept
{
    return !(left < right);
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_TENSOR_WORD_H
