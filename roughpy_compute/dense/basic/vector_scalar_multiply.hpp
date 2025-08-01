#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP


namespace rpy::compute::basic {
inline namespace v1 {

template <typename S>
void vector_scalar_multiply(
    DenseVectorView<S*> out,
    DenseVectorView<S const*> in,
    S const& scalar)
{
    using Size = typename DenseVectorView<S*>::Size;

    auto const common_size = std::min(out.size(), in.size());

    for (Size i=0; i < common_size; ++i) {
        out[i] = in[i] * scalar;
    }

}

} // version namespace
}; // namespace rpy::compute::basic


#endif //ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
