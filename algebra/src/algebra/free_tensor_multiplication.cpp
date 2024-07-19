//
// Created by sam on 6/11/24.
//

#include "free_tensor.h"

using namespace rpy;
using namespace rpy::algebra;

bool FreeTensorMultiplication::basis_compatibility_check(const Basis& basis
) noexcept
{

    return false;
}

FreeTensorMultiplication::FreeTensorMultiplication(
        deg_t width,
        deg_t tile_letters,
        deg_t degree
)
    : m_width(width),
      m_tile_letters(tile_letters)
{
    m_max_degree = std::max(2 * tile_letters + 1, degree);
    m_sizes.reserve(m_max_degree + 1);
    m_offsets.reserve(m_max_degree + 1);
    m_sizes.push_back(1);
    m_offsets.push_back(0);
    for (deg_t i = 0; i < m_max_degree; ++i) {
        m_sizes.push_back(m_sizes[i] * m_width);
        m_offsets.emplace_back(1 + m_offsets[i] * m_width);
    }

    if (m_tile_letters == 1) {
        m_reverses.reserve(m_width);
        for (deg_t i = 0; i < m_width; ++i) { m_reverses.push_back(i); }
    } else if (m_tile_letters > 1) {
        containers::SmallVec<dimn_t, 2> word(m_tile_letters);

        auto reverse_index = [this, &word]() {
            dimn_t result = 0;
            for (dimn_t i = 0; i < m_width; ++i) {
                result *= m_width;
                result += word[i];
            }
            return result;
        };
        auto next_word = [this, &word]() {
            for (auto& letter : word | views::reverse) {
                if (++letter == m_width) { letter = 0; }
                break;
            }
        };

        for (dimn_t i = 0; i < m_sizes[m_tile_letters]; ++i, next_word()) {
            m_reverses.push_back(reverse_index());
        }
    }
}

namespace {

struct TensorInfo {
    deg_t max_degree;
    deg_t left_max_deg = 0;
    deg_t right_max_deg = 0;
    deg_t left_min_deg = 0;
    deg_t right_min_deg = 0;
    deg_t tile_size = 0;
};

template <typename T, typename Op>
RPY_INLINE_ALWAYS void square_multiply(
        T* RPY_RESTRICT out,
        const T* RPY_RESTRICT left,
        const T* RPY_RESTRICT right,
        const dimn_t left_bound,
        const dimn_t right_bound,
        const dimn_t stride,
        Op&& op
)
{
    for (dimn_t i = 0; i < left_bound; ++i) {
        for (dimn_t j = 0; j < right_bound; ++j) {
            out[i * stride + j] = op(left[i] * right[j]);
        }
    }
}

template <typename T, typename Op>
void dense_fma_untiled(
        Slice<T> out,
        Slice<const T> left,
        Slice<const T> right,
        Op&& op,
        deg_t max_degree,
        const containers::Vec<dimn_t> sizes,
        const containers::Vec<dimn_t> offsets,
        const TensorInfo& info
)
{

    for (deg_t out_deg = max_degree; out_deg > 0; --out_deg) {

        T* optr = out.data() + offsets[out_deg];
        deg_t left_deg_max
                = std::min(info.left_max_deg, out_deg - info.right_min_deg);
        deg_t left_deg_min
                = std::max(info.left_min_deg, out_deg - info.right_max_deg);

        for (deg_t left_deg = left_deg_max; left_deg >= left_deg_min;
             --left_deg) {
            deg_t right_deg = out_deg - left_deg;
            const auto* lptr = left.data() + offsets[left_deg];
            const auto* rptr = right.data() + offsets[right_deg];
            square_multiply(
                    optr,
                    lptr,
                    rptr,
                    sizes[left_deg],
                    sizes[right_deg],
                    sizes[left_deg],
                    op
            );
        }
    }
}

template <typename T, typename Op>
void dense_fma_tiled(
        Slice<T> out,
        Slice<const T> left,
        Slice<const T> right,
        Op&& op,
        const TensorInfo& info
)
{}

template <typename T, typename Op>
void dense_fma(
        Slice<T> out,
        Slice<const T> left,
        Slice<const T> right,
        Op&& op,
        const TensorInfo& info
)
{
    deg_t untiled_max = info.max_degree;
    if (info.tile_size > 0 && 2 * info.tile_size > info.max_degree) {
        untiled_max -= 2 * info.tile_size + 1;
    }

    dense_fma_untiled(
            out,
            left,
            right,
            std::forward<Op>(op),
            untiled_max,
            info
    );
}

}// namespace

void FreeTensorMultiplication::fma(
        Vector& out,
        const Vector& left,
        const Vector& right,
        Identity&& op
) const
{}

void FreeTensorMultiplication::fma(
        Vector& out,
        const Vector& left,
        const Vector& right,
        Uminus&& op
) const
{}

void FreeTensorMultiplication::fma(
        Vector& out,
        const Vector& left,
        const Vector& right,
        PostMultiply&& op
) const
{}

void FreeTensorMultiplication::fma_dense(
        Vector& out,
        const Vector& left,
        const Vector& right,
        Identity&& op
) const
{}

void FreeTensorMultiplication::fma_dense(
        Vector& out,
        const Vector& left,
        const Vector& right,
        Uminus&& op
) const
{}
void FreeTensorMultiplication::fma_dense(
        Vector& out,
        const Vector& left,
        const Vector& right,
        PostMultiply&& op
) const
{}

void FreeTensorMultiplication::multiply_into(
        Vector& out,
        const Vector& right,
        Identity&& op
) const
{}

void FreeTensorMultiplication::multiply_into(
        Vector& out,
        const Vector& right,
        Uminus&& op
) const
{}

void FreeTensorMultiplication::multiply_into(
        Vector& out,
        const Vector& right,
        PostMultiply&& op
) const
{}
void FreeTensorMultiplication::multiply_into_dense(
        Vector& out,
        const Vector& right,
        Identity&& op
) const
{}
void FreeTensorMultiplication::multiply_into_dense(
        Vector& out,
        const Vector& right,
        PostMultiply&& op
) const
{}

void FreeTensorMultiplication::multiply_into_dense(
        Vector& out,
        const Vector& right,
        Uminus&& op
) const
{}
