//
// Created by sam on 8/13/24.
//

#ifndef LIE_WORD_H
#define LIE_WORD_H

#include "roughpy/devices/core.h"

#include <roughpy/core/container/vector.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/alloc.h>

namespace rpy {
namespace algebra {

namespace dtl {

using LieLetterType = int16_t;

inline constexpr dimn_t small_vec_size = sizeof(void*) / sizeof(LieLetterType);

}// namespace dtl

class LieWord : public platform::SmallObjectBase,
                containers::SmallVec<dtl::LieLetterType, dtl::small_vec_size>
{
    using container_t
            = containers::SmallVec<dtl::LieLetterType, dtl::small_vec_size>;

    template <typename I>
    static constexpr dtl::LieLetterType letter(I let) noexcept
    {
        return static_cast<dtl::LieLetterType>(let);
    }

    template <typename I>
    static constexpr dtl::LieLetterType offset(I offset) noexcept
    {
        return -static_cast<dtl::LieLetterType>(offset);
    }

    template <typename It>
    static constexpr bool is_offset(It val) noexcept
    {
        return (*val) < 0;
    }

    template <typename It>
    static constexpr bool is_letter(It val) noexcept
    {
        return (*val) > 0;
    }

    template <typename It>
    static constexpr let_t get_letter(It pos) noexcept
    {
        RPY_DBG_ASSERT(is_letter(pos));
        return static_cast<let_t>(*pos);
    }

    template <typename It>
    static constexpr It follow_offset(It offset) noexcept
    {
        RPY_DBG_ASSERT(is_offset(offset));
        return offset - *offset;
    }

    template <typename It>
    static constexpr It get_companion(It val) noexcept
    {
        return val + 1;
    }

    template <typename LetterFn, typename BinOp>
    static auto compute_over_tree(
            const_iterator root,
            LetterFn&& letter_fn,
            BinOp&& binary_op
    ) -> decltype(letter_fn(std::declval<let_t>()))
    {
        auto left = root;
        auto right = get_companion(root);

        return_type_t<LetterFn> left_result, right_result;

        if (is_offset(left)) {
            left_result = compute_over_tree(
                    follow_offset(left),
                    letter_fn,
                    binary_op
            );
        } else {
            left_result = letter_fn(get_letter(left));
        }

        if (is_offset(right)) {
            right_result = compute_over_tree(
                    follow_offset(right),
                    letter_fn,
                    binary_op
            );
        } else {
            right_result = letter_fn(get_letter(right));
        }

        return binary_op(left_result, right_result);
    }

    static dimn_t
    copy_tree(container_t& container, const_iterator root) noexcept;

    static bool check_equal(const_iterator left, const_iterator right) noexcept;

    LieWord(container_t&& container) : container_t{std::move(container)} {}

public:
    using iterator = typename container_t::iterator;
    using const_iterator = typename container_t::const_iterator;

    using container_t::begin;
    using container_t::end;
    using container_t::size;

    LieWord() = default;
    LieWord(const LieWord&) = default;
    LieWord(LieWord&&) noexcept = default;

    explicit LieWord(const_iterator root, dimn_t size_hint = 0);

    LieWord(let_t one_letter) : container_t{letter(one_letter)} {}

    LieWord(let_t left, let_t right) : container_t{letter(left), letter(right)}
    {}

    LieWord(const LieWord& left, const LieWord& right);

    LieWord& operator=(const LieWord&) = default;
    LieWord& operator=(LieWord&&) noexcept = default;

    RPY_NO_DISCARD bool is_valid() const noexcept { return !empty(); }
    RPY_NO_DISCARD bool is_letter() const noexcept { return size() == 1; }

    RPY_NO_DISCARD let_t get_letter() const noexcept
    {
        RPY_DBG_ASSERT(is_letter());
        return static_cast<let_t>(*begin());
    }

    RPY_NO_DISCARD deg_t degree() const noexcept;
    RPY_NO_DISCARD LieWord left_parent() const noexcept;
    RPY_NO_DISCARD LieWord right_parent() const noexcept;

    void print(std::ostream& out) const;

    RPY_NO_DISCARD deg_t min_alphabet_size();

    friend bool operator==(const LieWord& left, const LieWord& right) noexcept
    {
        if (left.empty() && right.empty()) { return true; }
        if (left.is_letter() && right.is_letter()) {
            return get_letter(left.begin()) == get_letter(right.begin());
        }
        return check_equal(left.begin(), right.begin());
    }

    static inline hash_t letter_hash(let_t letter) noexcept
    {
        Hash<let_t> hasher;
        return hasher(letter);
    }

    static inline hash_t hash_binop(hash_t left, hash_t right) noexcept
    {
        hash_combine(left, right);
        return left;
    }

    template <typename LetterFn, typename BinOp>
    decltype(auto) foliage_map(LetterFn&& letter, BinOp&& binary_op) const
    {
        RPY_DBG_ASSERT(is_valid());

        if (is_letter()) { return letter(get_letter()); }

        return compute_over_tree(
                begin(),
                std::forward<LetterFn>(letter),
                std::forward<BinOp>(binary_op)
        );
    }
};

inline bool operator!=(const LieWord& left, const LieWord& right) noexcept
{
    return !(left == right);
}

inline std::ostream& operator<<(std::ostream& os, const LieWord& word)
{
    word.print(os);
    return os;
}

hash_t hash_value(const LieWord& word) noexcept;

inline LieWord operator*(const LieWord& left, const LieWord& right)
{
    return LieWord(left, right);
}

inline LieWord operator*(const LieWord& left, let_t right)
{
    return LieWord(left, LieWord(right));
}

inline LieWord operator*(let_t left, const LieWord& right)
{
    return LieWord{LieWord(left), right};
}

inline hash_t hash_value(const LieWord& word) noexcept
{
    return word.foliage_map(LieWord::letter_hash, LieWord::hash_binop);
}

}// namespace algebra

namespace devices {

template <>
TypePtr get_type<algebra::LieWord>();

}

}// namespace rpy

#endif// LIE_WORD_H
