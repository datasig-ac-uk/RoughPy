//
// Created by sam on 8/13/24.
//

#include "lie_word.h"

#include <roughpy/core/strings.h>

using namespace rpy;
using namespace rpy::algebra;

dimn_t LieWord::copy_tree(
        container_t& container,
        typename LieWord::const_iterator root
) noexcept
{
    if (is_letter(root)) {
        container.emplace_back(root);
        return 1;
    }

    auto left_root = follow_offset(root);
    auto right_root = get_companion(root);

    container.emplace_back(2);
    auto& offset = container.emplace_back(0);

    auto size = copy_tree(container, left_root);
    offset = offset(size);

    size += copy_tree(container, right_root);
    return size;
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
        emplace_back(offset(left.size()));

        auto it = insert(end(), left.begin(), left.end());
        insert(it, right.begin(), right.end());
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

optional<typename LieWord::const_iterator> LieWord::left_parent() const noexcept
{
    if (empty()) { return {}; }
    return begin();
}
optional<typename LieWord::const_iterator>
LieWord::right_parent() const noexcept
{
    if (size() <= 1) { return {}; }
    return get_companion(begin());
}
