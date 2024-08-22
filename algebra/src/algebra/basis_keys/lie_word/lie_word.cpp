//
// Created by sam on 8/13/24.
//

#include "lie_word.h"
#include "lie_word_type.h"

#include <roughpy/core/strings.h>

using namespace rpy;
using namespace rpy::algebra;

dimn_t LieWord::copy_tree(
        container_t& container,
        typename LieWord::const_iterator root
) noexcept
{
    dimn_t additional_size = 0;
    auto left_root = root;
    auto right_root = get_companion(left_root);

    auto new_left_root = container.insert(container.end(), 0);
    auto new_right_root = container.insert(container.end(), 0);

    if (is_letter(left_root)) {
        *new_left_root = *left_root;
    } else {
        // New left data will start after this pair
        *new_left_root = offset(2);
        additional_size += copy_tree(container, follow_offset(left_root));
    }

    if (is_letter(right_root)) {
        *new_right_root = *right_root;
    } else {
        *new_right_root = offset(additional_size);
        additional_size += copy_tree(container, follow_offset(right_root));
    }

    return 2 + additional_size;
}

LieWord::LieWord(const_iterator root, dimn_t size_hint)
{
    reserve(size_hint);
    copy_tree(*this, root);
}

LieWord::LieWord(const LieWord& left, const LieWord& right)
{
    if (RPY_UNLIKELY(left.empty() && right.empty())) {
        // Do Nothing
    } else if (left.is_letter() && right.is_letter()) {
        /*
         * If both left and right are letters, then we don't need offsets and we
         * can just put the letters in their respective places.
         */
        emplace_back(letter(get_letter(left.begin())));
        emplace_back(letter(get_letter(right.begin())));
    } else if (left.is_letter()) {
        /*
         * If the left is a letter then it can just be placed, but right still
         * needs an offset.
         */
        reserve(2 + right.size());
        emplace_back(letter(get_letter(left.begin())));
        emplace_back(offset(1));
        insert(end(), right.begin(), right.end());
    } else if (right.is_letter()) {
        /*
         * If the right is a letter then it can just be placed, but left still
         * needs an offset.
         * In this case, the offset for the start of the left word is 2
         */
        reserve(2 + right.size());
        emplace_back(offset(2));
        emplace_back(letter(get_letter(right.begin())));
        insert(end(), left.begin(), left.end());
    } else {
        reserve(2 + left.size() + right.size());
        /*
         * The array looks like this
         *
         *   x  x  l  l  l ... l  r  r  r ... r
         *   ^     |              |
         *   | +1 +2              |
         *      ^                 |
         *      | +1 ...          +size(l)
         */

        emplace_back(offset(2));
        emplace_back(offset(1 + left.size()));
        RPY_DBG_ASSERT(size() == 2);

        insert(end(), left.begin(), left.end());
        insert(end(), right.begin(), right.end());
    }
}

deg_t LieWord::degree() const noexcept
{
    const auto sz = size();
    if (sz <= 2) { return static_cast<deg_t>(sz); }

    return compute_over_tree(
            begin(),
            [](let_t) { return 1; },
            [](deg_t left, deg_t right) { return left + right; }
    );
}

namespace {

inline string to_string_letterfn(let_t letter)
{
    return std::to_string(letter);
}

inline string to_string_binop(const string& left, const string& right)
{
    return string_cat('[', left, ',', right, ']');
}

}// namespace

void LieWord::print(std::ostream& out) const
{
    if (!empty()) {
        if (is_letter()) {
            out << get_letter(begin());
        } else {
            out << compute_over_tree(
                    begin(),
                    to_string_letterfn,
                    to_string_binop
            );
        }
    }
}
deg_t LieWord::min_alphabet_size()
{
    if (empty()) { return 0; }
    if (is_letter()) { static_cast<deg_t>(get_letter(begin())); }
    return static_cast<deg_t>(*ranges::max_element(*this));
}

LieWord LieWord::left_parent() const noexcept
{
    if (size() <= 1) { return {}; }
    auto root = begin();
    if (is_letter(root)) { return LieWord{get_letter(root)}; }

    RPY_DBG_ASSERT(is_offset(root));
    dimn_t size_hint = size();
    if (is_letter(get_companion(begin()))) {
        size_hint -= 1;
    } else {
        size_hint /= 2;
    }

    return LieWord(follow_offset(root), size_hint);
}
LieWord LieWord::right_parent() const noexcept
{
    if (empty()) { return {}; }
    if (is_letter()) { return LieWord{get_letter(begin())}; }

    auto root = get_companion(begin());
    if (is_letter(root)) { return LieWord{get_letter(root)}; }

    dimn_t size_hint = size();
    if (is_letter(begin())) {
        size_hint -= 1;
    } else {
        size_hint /= 2;
    }

    return LieWord(follow_offset(root), size_hint);
}

bool LieWord::check_equal(const_iterator left, const_iterator right) noexcept
{
    if (is_letter(left) && is_letter(right)) {
        if (get_letter(left) != get_letter(right)) { return false; }
    } else if (is_offset(left) && is_offset(right)) {
        if (!check_equal(follow_offset(left), follow_offset(right))) {
            return false;
        }
    } else {
        return false;
    }

    left = get_companion(left);
    right = get_companion(right);
    if (is_letter(left) && is_letter(right)) {
        if (get_letter(left) != get_letter(right)) { return false; }
    } else if (is_offset(left) && is_offset(right)) {
        if (!check_equal(follow_offset(left), follow_offset(right))) {
            return false;
        }
    } else {
        return false;
    }

    return true;
}
template <>
devices::TypePtr devices::get_type<LieWord>()
{
    return LieWordType::get();
}
