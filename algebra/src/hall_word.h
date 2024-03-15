//
// Created by sam on 3/14/24.
//

#ifndef ROUGHPY_HALL_WORD_H
#define ROUGHPY_HALL_WORD_H

#include "basis_key.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <boost/container/small_vector.hpp>

namespace rpy {
namespace algebra {

class HallWord : public BasisKeyInterface
{
    using container_t = boost::container::small_vector<int16_t, 2>;
    using iterator = typename container_t::iterator;
    using const_iterator = typename container_t::const_iterator;

    container_t m_tree;

    template <typename It>
    static inline bool is_offset(It val) noexcept
    {
        return *val < 0;
    }
    template <typename It>
    static inline bool is_letter(It val) noexcept
    {
        return *val > 0;
    }

    template <typename It>
    static inline It follow_offset(It offset) noexcept
    {
        return offset - *offset;
    }

    template <typename It>
    static inline It get_companion(It val) noexcept
    {
        return val + 1;
    }

    RPY_NO_DISCARD dimn_t size() const noexcept { return m_tree.size(); }

    static void copy_tree(container_t& new_tree, const_iterator root);

    template <typename LetterFn, typename BinOp>
    static auto compute_over_tree(
            const_iterator root,
            LetterFn&& letter_fn,
            BinOp&& binary_op
    ) -> decltype(letter_fn(std::declval<let_t>()))
    {
        if (is_letter(root)) { return letter_fn(static_cast<let_t>(*root)); }

        RPY_DBG_ASSERT(is_offset(root));

        auto left = follow_offset(root);
        auto right = get_companion(root);

        return binary_op(
                compute_over_tree(left, letter_fn, binary_op),
                compute_over_tree(right, letter_fn, binary_op)
        );
    }

    explicit HallWord(container_t&& data) : m_tree(std::move(data)) {}

public:
    static constexpr string_view key_name = "hall_word";

    HallWord(let_t letter);
    HallWord(let_t left, let_t right);

    HallWord(const HallWord* left, const HallWord* right);

    ~HallWord() override;

    string_view key_type() const noexcept override;
    BasisPointer basis() const noexcept override;

    RPY_NO_DISCARD deg_t degree() const noexcept;
    RPY_NO_DISCARD pair<BasisKey, optional<BasisKey>> parents() const;

    RPY_NO_DISCARD string to_string() const;

    RPY_NO_DISCARD let_t first_letter() const;
};
RPY_NO_DISCARD inline const HallWord* cast_to_hallword(const BasisKey& key)
{
    RPY_CHECK(key.is_valid_pointer());
    auto* ptr = key.get_pointer();
    RPY_CHECK(ptr->key_type() == HallWord::key_name);
    return static_cast<const HallWord*>(ptr);
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_HALL_WORD_H
