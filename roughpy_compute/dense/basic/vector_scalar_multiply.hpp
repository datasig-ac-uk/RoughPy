#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP


namespace rpy::compute::basic {
inline namespace v1 {

template <typename S, typename Basis>
void vector_scalar_multiply(
    DenseVectorView<S*, Basis> out,
    DenseVectorView<S const*, Basis> in,
    S const& scalar)
{
    using Index = typename DenseVectorView<S*, Basis>::Index;

    auto const common_size = std::min(out.size(), in.size());

    for (Index i=0; i < common_size; ++i) {
        out[i] = in[i] * scalar;
    }

}

} // version namespace
}; // namespace rpy::compute::basic


#endif //ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
