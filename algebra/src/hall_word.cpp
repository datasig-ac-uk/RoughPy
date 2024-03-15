//
// Created by sam on 3/14/24.
//

#include "hall_word.h"

#include <sstream>

using namespace rpy;
using namespace rpy::algebra;

HallWord::HallWord(let_t letter) : m_tree{static_cast<int16_t>(letter)} {}
HallWord::HallWord(let_t left, let_t right)
    : m_tree{static_cast<int16_t>(left), static_cast<int16_t>(right)}
{}
HallWord::HallWord(const HallWord* left, const HallWord* right)
{
    m_tree.reserve(2 + left->m_tree.size() + right->m_tree.size());
    m_tree.push_back(-2);
    m_tree.push_back(-static_cast<int16_t>(1 + left->m_tree.size()));

    auto it = m_tree.insert(
            m_tree.end(),
            left->m_tree.begin(),
            left->m_tree.end()
    );

    m_tree.insert(it, right->m_tree.begin(), right->m_tree.end());
}

HallWord::~HallWord() = default;

string_view HallWord::key_type() const noexcept { return key_name; }
BasisPointer HallWord::basis() const noexcept
{
    return rpy::algebra::BasisPointer();
}

namespace {

inline deg_t degree_of_letter(let_t) noexcept { return 1; }

inline deg_t degree_binop(deg_t left, deg_t right) { return left + right; }

}// namespace
deg_t HallWord::degree() const noexcept
{
    if (m_tree.empty()) { return 0; }
    if (m_tree.size() == 1) { return 1; }
    if (m_tree.size() == 2) { return 2; }

    return compute_over_tree(m_tree.begin(), degree_of_letter, degree_binop);
}

void HallWord::copy_tree(
        HallWord::container_t& new_tree,
        HallWord::const_iterator root
)
{
    if (is_letter(root)) {
        new_tree.push_back(static_cast<let_t>(*root));
    } else {
        copy_tree(new_tree, follow_offset(root));
    }
    if (is_letter(++root)) {
        new_tree.push_back(static_cast<let_t>(*root));
    } else {
        copy_tree(new_tree, follow_offset(root));
    }
}

pair<BasisKey, optional<BasisKey>> HallWord::parents() const
{
    if (m_tree.size() <= 1) { return {BasisKey(this), {}}; }
    if (m_tree.size() == 2) {
        return {BasisKey(new HallWord(static_cast<let_t>(m_tree[0]))),
                BasisKey(new HallWord(static_cast<let_t>(m_tree[1])))};
    }

    auto it = m_tree.begin();
    container_t left_tree;
    if (is_letter(it)) {
        left_tree.push_back(static_cast<let_t>(*it));
    } else {
        RPY_DBG_ASSERT(is_offset(it));
        copy_tree(left_tree, follow_offset(it));
    }

    container_t right_tree;
    if (is_letter(++it)) {
        right_tree.push_back(static_cast<let_t>(*it));
    } else {
        RPY_DBG_ASSERT(is_offset(it));
        copy_tree(right_tree, follow_offset(it));
    }

    return {BasisKey(new HallWord(std::move(left_tree))),
            BasisKey(new HallWord(std::move(right_tree)))};
}

namespace {

inline string to_string_letterfn(let_t letter)
{
    return std::to_string(letter);
}

inline string to_string_binop(string left, string right)
{
    std::stringstream ss;
    ss << '[' << left << ',' << right << ']';
    return ss.str();
}

}// namespace

string HallWord::to_string() const
{
    if (m_tree.empty()) { return ""; }
    if (m_tree.size() == 1) {
        return to_string_letterfn(static_cast<let_t>(m_tree[0]));
    }

    return compute_over_tree(
            m_tree.begin(),
            to_string_letterfn,
            to_string_binop
    );
}
