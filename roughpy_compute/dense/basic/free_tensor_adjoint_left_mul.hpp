#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ADJOINT_LEFT_MUL_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ADJOINT_LEFT_MUL_HPP


#include "roughpy_compute/dense/views.hpp"


namespace rpy::compute::basic {
inline namespace v1 {
template<typename OutIter, typename OpIter, typename ArgIter>
void ft_adj_lmul(DenseTensorView<OutIter> out,
                 DenseTensorView<OpIter> op,
                 DenseTensorView<ArgIter> arg) {
    using Degree = typename DenseTensorView<OutIter>::Degree;
    using Index = typename DenseTensorView<OutIter>::Index;

    /*
     * There are several choices for strategy here. There is a relationship
     * between the output degre (odeg), the operator degree (opdeg) and the
     * argument degree (adeg) given by
     *
     *   adeg = opdeg + odeg.
     *
     * This means we have a choice of traversal pattern over each of the
     * arguments in terms of which combination of degrees we access and in what
     * order these combinations appear. For instance, we might major the output
     * degree, in which case we compute all of the terms of a given odeg before
     * moving on to the next. This gives some notion of locality in the output,
     * but not in either of the input arguments. Alternatively, we might major
     * the operator degree or the argument degree. There are probably scenarios
     * where each choice of major is optimal, but I don't yet understand the
     * relationships. However, it is clear that the argument degree will usually
     * be the largest at play at any given time, so it is reasonable to major
     * the argument degree.
     *
     * This, of course, is ignoring the ability to tile the operation in the
     * same way as we perform a tiled fma or antipode operation. This is a
     * future improvement because it requires the same kind of layout framework
     * as those operations do.
     */

    const auto arg_max_degree = std::min(arg.max_degree() - op.min_degree(), out.max_degree());
    auto arg_min_degree = std::max(arg.min_degree() - op.max_degree(), out.min_degree());

    for (Degree arg_degree = arg_max_degree; arg_degree >= arg_min_degree; --
         arg_degree)
    {
        auto out_min_degree = std::max(arg_degree - op.max_degree(), out.min_degree());
        auto out_max_degree = std::min(arg_degree - op.min_degree(), out.max_degree());

        auto arg_frag = arg.at_level(arg_degree);

        for (Degree out_degree = out_max_degree;
            out_degree >= out_min_degree;
            --out_degree)
        {
            const auto op_degree = arg_degree - out_degree ;

            auto op_frag = op.at_level(op_degree);
            auto out_frag = out.at_level(out_degree);


            // ReSharper disable CppDFANullDereference
            for (Index op_idx=0; op_idx < op_frag.size(); ++op_idx) {
                const auto op_offset = op_idx * out_frag.size();
                const auto& op_val = op_frag[op_idx];

                for (Index i=0; i < out_frag.size(); ++i) {
                    out_frag[i] += op_val * arg_frag[i + op_offset];
                }


            }
            // ReSharper restore CppDFANullDereference



        }
    }
}
} // version namespace
} // namespace rpy::compute::basic


#endif //ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ADJOINT_LEFT_MUL_HPP
