#include "cpu/dense_ft_fma.hpp"

#include <cstring>
#include <type_traits>

#include "roughpy_compute/dense/basic/free_tensor_fma.hpp"

namespace {

namespace ffi = xla::ffi;
using namespace rpy::compute;

// Defensive check that indexing types match as we cast data from JAX into C++
inline constexpr ffi::DataType FfiIndexType = ffi::DataType::S64;
static_assert(
    std::is_same_v<
        TensorBasis::Index,
        ffi::NativeType<FfiIndexType>
    >,
    "TensorBasis::Index size must match byte width for degree_begin"
);

} // namespace

namespace rpy::jax::cpu {

template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T>& buffer)
{
    auto dims = buffer.dimensions();
    if (dims.size() == 0) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer.element_count(), dims.back());
}

ffi::Error cpu_dense_ft_fma_impl(
    int a_width,
    int a_depth,
    int b_width,
    int b_depth,
    int c_width,
    int c_depth,
    ffi::Buffer<FfiIndexType> a_degree_begin,
    ffi::Buffer<FfiIndexType> b_degree_begin,
    ffi::Buffer<FfiIndexType> c_degree_begin,
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> b,
    ffi::Buffer<ffi::F32> c,
    ffi::ResultBuffer<ffi::F32> result
) {
    auto [a_size, a_dim] = GetDims(a);
    auto [a_db_size, a_db_dim] = GetDims(a_degree_begin);
    if (a_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma a must be an array");
    }
    if (a_db_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin must be an array");
    }
    if (a_db_size != a_depth + 2) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin size must be depth + 2");
    }

    auto [b_size, b_dim] = GetDims(b);
    auto [b_db_size, b_db_dim] = GetDims(b_degree_begin);
    if (b_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma a must be an array");
    }
    if (b_db_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin must be an array");
    }
    if (b_db_size != b_depth + 2) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin size must be depth + 2");
    }

    auto [c_size, c_dim] = GetDims(c);
    auto [c_db_size, c_db_dim] = GetDims(c_degree_begin);
    if (c_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma a must be an array");
    }
    if (c_db_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin must be an array");
    }
    if (c_db_size != c_depth + 2) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin size must be depth + 2");
    }

    auto [result_size, result_dim] = GetDims(*result);
    if (result_dim != a_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma result array must match a array");
    }
    if (result_size != a_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma result size must match a size");
    }

    TensorBasis a_basis = { a_degree_begin.typed_data(), a_width, a_depth };
    TensorBasis b_basis = { b_degree_begin.typed_data(), b_width, b_depth };
    TensorBasis c_basis = { c_degree_begin.typed_data(), c_width, c_depth };

    // JAX array are immutable so rather than the ternary ft_fma call where the
    // first argument is overwritten, this is a quaternary method producing a
    // new result, so to supports ft_fma's behaviour of overwriting 'out'
    // buffer, 'a' contents are copied into result buffer to operate on it.
    std::memcpy(result->typed_data(), a.typed_data(), a.size_bytes());

    // FIXME change behaviour/basis/depths to match _internals.dense_ft_fma
    // FIXME hardcoded to float: use correct type
    DenseTensorView<float*> result_view(result->typed_data(), a_basis, 0, 0);
    DenseTensorView<const float*> lhs_view(b.typed_data(), b_basis, 0, 0);
    DenseTensorView<const float*> rhs_view(c.typed_data(), c_basis, 0, 0);

    basic::v1::ft_fma(result_view, lhs_view, rhs_view);

    return ffi::Error::Success();
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_fma,
    rpy::jax::cpu::cpu_dense_ft_fma_impl,
    xla::ffi::Ffi::Bind()
        .Attr<int>("a_width")
        .Attr<int>("a_depth")
        .Attr<int>("b_width")
        .Attr<int>("b_depth")
        .Attr<int>("c_width")
        .Attr<int>("c_depth")
        .Arg<ffi::Buffer<FfiIndexType>>()
        .Arg<ffi::Buffer<FfiIndexType>>()
        .Arg<ffi::Buffer<FfiIndexType>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
