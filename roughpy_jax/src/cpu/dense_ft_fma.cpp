#include <cstring>
#include <type_traits>

#include "xla_includes.hpp"

#include "roughpy_compute/dense/basic/free_tensor_fma.hpp"

namespace {

namespace ffi = xla::ffi;

// FIXME split into 32 and 64 variants with templates?
using RpyFloatType = float;
inline constexpr ffi::DataType XlaIndexType = ffi::DataType::S32;
inline constexpr ffi::DataType XlaFloatType = ffi::DataType::F32;
static_assert(
    std::is_same_v<
        RpyFloatType,
        ffi::NativeType<XlaFloatType>
    >,
    "XlaFloatType must match underlying float type"
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

// FIXME This reworked/duplicate from roughpy/compute/_src/call_config.cpp, can
// We reuse private CallConfig to centralise this behaviour?
ffi::Error update_algebra_params(
    int width,
    int depth,
    int& out_min_degree,
    int& lhs_min_degree,
    int& rhs_min_degree,
    int& out_max_degree,
    int& lhs_max_degree,
    int& rhs_max_degree
)
{
    if (lhs_max_degree == -1 || lhs_max_degree >= depth) {
        lhs_max_degree = depth;
    }
    if (lhs_max_degree < lhs_min_degree) {
        return ffi::Error::InvalidArgument("lhs_min_degree must be less than lhs_max_degree");
    }

    if (rhs_max_degree == -1 || rhs_max_degree >= depth) {
        rhs_max_degree = depth;
    }
    if (rhs_max_degree < rhs_min_degree) {
        return ffi::Error::InvalidArgument("rhs_min_degree must be less than rhs_max_degree");
    }

    if (out_max_degree == -1 || out_max_degree >= depth) {
        out_max_degree = depth;
    }
    if (out_max_degree < out_min_degree) {
        return ffi::Error::InvalidArgument("out_min_degree must be less than out_max_degree");
    }

    return ffi::Error::Success();
}

ffi::Error cpu_dense_ft_fma_impl(
    int width,
    int depth,
    int out_depth, // FIXME should we keep roughpy_compute 'out' naming? Not strictly correct in JAX
    int lhs_depth,
    int rhs_depth,
    ffi::Buffer<XlaIndexType> degree_begin,
    ffi::Buffer<XlaFloatType> out,
    ffi::Buffer<XlaFloatType> lhs,
    ffi::Buffer<XlaFloatType> rhs,
    ffi::ResultBuffer<XlaFloatType> result
) {
    using namespace rpy::compute;

    auto [out_size, out_dim] = GetDims(out);
    if (out_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma out must be an array");
    }

    auto [lhs_size, lhs_dim] = GetDims(lhs);
    if (lhs_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma lhs must be an array");
    }

    auto [rhs_size, rhs_dim] = GetDims(rhs);
    if (rhs_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma rhs must be an array");
    }

    auto [degree_begin_size, degree_begin_dim] = GetDims(degree_begin);
    if (degree_begin_size != depth + 2) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin size must be depth + 2");
    }

    auto [result_size, result_dim] = GetDims(*result);
    if (result_dim != out_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma result array must match a array");
    }

    if (result_size != out_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma result size must match a size");
    }

    // The internal tensor basis degree_begin array is 64 bit ptrdiffs, so copy
    // the int32 index data over to a temporary array on the heap.
    std::vector<TensorBasis::BasisBase::Index> degree_begin_i64(degree_begin_size);
    std::copy(
        degree_begin.typed_data(),
        degree_begin.typed_data() + degree_begin_size,
        degree_begin_i64.begin()
    );

    // FIXME for review: narrowing conversion on width and depth, underlying types
    TensorBasis basis = { degree_begin_i64.data(), width, depth };

    // FIXME for review: this is hardcoded to match roughpy compute. Should
    // these be set via args universally? Also centralise code.
    int out_min_degree = 0;
    int lhs_min_degree = 0;
    int rhs_min_degree = 0;
    int out_max_degree = out_depth;
    int lhs_max_degree = lhs_depth;
    int rhs_max_degree = rhs_depth;
    auto depth_check_error = update_algebra_params(
        width,
        depth,
        out_min_degree,
        lhs_min_degree,
        rhs_min_degree,
        out_max_degree,
        lhs_max_degree,
        rhs_max_degree
    );

    if (depth_check_error.failure()) {
        return depth_check_error;
    }

    // JAX array are immutable so rather than the ternary ft_fma call where the
    // first argument is overwritten, this is a quaternary method preserving
    // `out` and returning new result. To use old interface, `result` is mutable
    // buffer copied from `out` that that ft_fma can work on.
    std::copy_n(out.typed_data(), out_size, result->typed_data());

    // FIXME for discussion in review: during weekly catch-up we changed size
    // from out_size to out_max_degree, but for now I have left as out_size
    // because result dims are set by the calling JAX code, so in order to init
    // the array as we do below, we would need to move the out/lhs/rhs min/max
    // degree calculations out into Python too. Also, I think we would also need
    // checks that buffers passed into this method are safe, i.e. the call to
    // basic::ft_fma will not go outside bounds.
    // std::copy_n(out.typed_data(), out_max_degree, result->typed_data());

    DenseTensorView<RpyFloatType*> result_view(result->typed_data(), basis, out_min_degree, out_max_degree);
    DenseTensorView<const RpyFloatType*> lhs_view(lhs.typed_data(), basis, lhs_min_degree, lhs_max_degree);
    DenseTensorView<const RpyFloatType*> rhs_view(rhs.typed_data(), basis, rhs_min_degree, rhs_max_degree);

    // Compute fma into result originally copied from out array
    basic::ft_fma(result_view, lhs_view, rhs_view);

    return ffi::Error::Success();
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_fma,
    rpy::jax::cpu::cpu_dense_ft_fma_impl,
    xla::ffi::Ffi::Bind()
        .Attr<int>("width")
        .Attr<int>("depth")
        .Attr<int>("out_depth")
        .Attr<int>("lhs_depth")
        .Attr<int>("rhs_depth")
        .Arg<ffi::Buffer<XlaIndexType>>()
        .Arg<ffi::Buffer<XlaFloatType>>()
        .Arg<ffi::Buffer<XlaFloatType>>()
        .Arg<ffi::Buffer<XlaFloatType>>()
        .Ret<ffi::Buffer<XlaFloatType>>()
);
