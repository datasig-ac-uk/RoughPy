#include "dense_basic.h"



#include <roughpy_compute/common/cache_array.hpp>
#include <roughpy_compute/dense/views.hpp>

#include <roughpy_compute/dense/basic/free_tensor_antipode.hpp>
#include <roughpy_compute/dense/basic/free_tensor_fma.hpp>
#include <roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp>
#include <roughpy_compute/dense/basic/shuffle_tensor_product.hpp>
#include <roughpy_compute/dense/basic/vector_addition.hpp>
#include <roughpy_compute/dense/basic/vector_inplace_addition.hpp>
#include <roughpy_compute/dense/basic/vector_inplace_scalar_multiply.hpp>
#include <roughpy_compute/dense/basic/vector_scalar_multiply.hpp>


#include "algebra_config.hpp"
#include "py_ternary_array_fn.hpp"



using namespace rpy::compute;

/*******************************************************************************
 * Free tensor FMA
 ******************************************************************************/
namespace {

template <typename Scalar_>
struct DenseFTFma
{
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    rpy::compute::CacheArray<size_t, CoreDims + 1> degree_begin;

    DenseFTFma(AlgebraConfig const& config)
        : degree_begin(config.depth + 2)
    {
        auto const s_width = static_cast<size_t>(config.width);
        degree_begin[0] = 0;
        for (size_t i = 1; i <= config.depth + 1; ++i) {
            degree_begin[i] = 1 + degree_begin[i - 1] * s_width;
        }
    }

    template <typename OutIter, typename LhsIter, typename RhsIter>
    void operator()(OutIter out_iter,
                    LhsIter lhs_iter,
                    RhsIter rhs_iter,
                    AlgebraConfig const& config) const
    {

        DenseTensorView<OutIter> out_view(
            out_iter,
            {degree_begin.data(), config.width, config.depth},
            0,
            config.depth
        );

        DenseTensorView<LhsIter> lhs_view(
            lhs_iter,
            {degree_begin.data(), config.width, config.depth},
            config.lhs_min_degree,
            config.lhs_max_degree
        );

        DenseTensorView<RhsIter> rhs_view(
            rhs_iter,
            {degree_begin.data(), config.width, config.depth},
            config.rhs_min_degree,
            config.rhs_max_degree
        );

        basic::ft_fma(out_view, lhs_view, rhs_view);
    }
};

}// namespace


PyObject* dense_ft_fma(PyObject* out, PyObject* lhs, PyObject* rhs)
{
    return ternary_function_outer<DenseFTFma>(out, lhs, rhs);
}
