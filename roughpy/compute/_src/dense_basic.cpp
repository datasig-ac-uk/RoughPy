// ReSharper disable CppTooWideScopeInitStatement
// ReSharper disable CppDFAUnusedValue
#include "dense_basic.h"

#include <array>

#include <roughpy_compute/common/cache_array.hpp>
#include <roughpy_compute/common/sparse_matrix.hpp>
#include <roughpy_compute/dense/views.hpp>

#include <roughpy_compute/dense/basic/apply_sparse_linear_map.hpp>
#include <roughpy_compute/dense/basic/free_tensor_adjoint_left_mul.hpp>
#include <roughpy_compute/dense/basic/free_tensor_antipode.hpp>
#include <roughpy_compute/dense/basic/free_tensor_fma.hpp>
#include <roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp>
#include <roughpy_compute/dense/basic/shuffle_tensor_product.hpp>

// The vector operations are of limited use to us here. We
// #include <roughpy_compute/dense/basic/vector_addition.hpp>
// #include <roughpy_compute/dense/basic/vector_inplace_addition.hpp>
// #include <roughpy_compute/dense/basic/vector_inplace_scalar_multiply.hpp>
// #include <roughpy_compute/dense/basic/vector_scalar_multiply.hpp>

#include <roughpy/pycore/compat.h>
#include <roughpy/pycore/object_handle.hpp>

#include "call_config.hpp"

#include "py_binary_array_fn.hpp"
#include "py_ternary_array_fn.hpp"
#include "tensor_basis.h"

using namespace rpy;
using namespace rpy::compute;

/*******************************************************************************
 * Free tensor FMA
 ******************************************************************************/
namespace {
template <typename Scalar_>
struct DenseFTFma {
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 3;
    static constexpr npy_intp arg_basis_mapping[3] = {0, 0, 0};

    CallConfig const* config_;

    explicit DenseFTFma(CallConfig const& config) : config_(&config) {}

    template <typename OutIter, typename LhsIter, typename RhsIter>
    void operator()(OutIter out_iter, LhsIter lhs_iter, RhsIter rhs_iter) const
    {
        auto const* basis
                = static_cast<TensorBasis const*>(config_->basis_data[0]);

        DenseTensorView<OutIter> out_view(
                out_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );

        DenseTensorView<LhsIter> lhs_view(
                lhs_iter,
                *basis,
                config_->degree_bounds[1].min_degree,
                config_->degree_bounds[1].max_degree
        );

        DenseTensorView<RhsIter> rhs_view(
                rhs_iter,
                *basis,
                config_->degree_bounds[2].min_degree,
                config_->degree_bounds[2].max_degree
        );

        basic::ft_fma(out_view, lhs_view, rhs_view);
    }
};

}// namespace

PyObject* py_dense_ft_fma(
        PyObject* self [[maybe_unused]],
        PyObject* args,
        PyObject* kwargs
)
{

    static constexpr char const* const kwords[]
            = {"out",
               "lhs",
               "rhs",
               "basis",
               "out_depth",
               "lhs_depth",
               "rhs_depth",
               nullptr};

    PyObject *out_obj, *lhs_obj, *rhs_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 3> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOOO|iii",
                RPY_PY_KWORD_CAST(kwords),
                &out_obj,
                &lhs_obj,
                &rhs_obj,
                &basis_obj,
                &degree_bounds[0].max_degree,
                &degree_bounds[1].max_degree,
                &degree_bounds[2].max_degree
        )) {
        return nullptr;
    }

    // if (!update_depth_params(config)) {
    //     PyErr_SetString(PyExc_ValueError, "incompatible depth parameters");
    //     return nullptr;
    // }

    const BasisBase* basis_data[1];

    TensorBasis basis;
    auto const degree_begins_handle = to_basis(basis_obj, basis);

    if (!degree_begins_handle) {
        // Error already set
        return nullptr;
    }
    basis_data[0] = &basis;

    CallConfig config{
            degree_bounds.data(),
            basis_data,
            nullptr,
    };

    return ternary_function_outer<DenseFTFma>(
            out_obj,
            lhs_obj,
            rhs_obj,
            config
    );
}

/*******************************************************************************
 * Free tensor Inplace multiply
 ******************************************************************************/
namespace {

template <typename Scalar_>
struct DenseFTInplaceMul {
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {0, 0};

    CallConfig const* config_;

    explicit DenseFTInplaceMul(CallConfig const& config) : config_(&config) {}

    template <typename OutIter, typename RhsIter>
    void operator()(OutIter out_iter, RhsIter rhs_iter) const
    {
        auto const* basis
                = static_cast<TensorBasis const*>(config_->basis_data[0]);

        DenseTensorView<OutIter> out_view(
                out_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );

        DenseTensorView<RhsIter> rhs_view(
                rhs_iter,
                *basis,
                config_->degree_bounds[1].min_degree,
                config_->degree_bounds[1].max_degree
        );

        basic::ft_inplace_mul(out_view, rhs_view);
    }
};

}// namespace

PyObject* py_dense_ft_inplace_mul(
        PyObject* Py_UNUSED(self),
        PyObject* args,
        PyObject* kwargs
)
{
    static constexpr char const* const kwords[]
            = {"lhs", "rhs", "basis", "out_depth", "rhs_depth", nullptr};

    PyObject *out_obj, *rhs_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOO|ii",
                RPY_PY_KWORD_CAST(kwords),
                &out_obj,
                &rhs_obj,
                &basis_obj,
                &degree_bounds[0].max_degree,
                &degree_bounds[1].max_degree
        )) {
        return nullptr;
    }

    const BasisBase* basis_data[1];

    TensorBasis basis;
    const auto handle = to_basis(basis_obj, basis);
    if (!handle) {
        // error already set
        return nullptr;
    }
    basis_data[0] = &basis;

    CallConfig config{degree_bounds.data(), basis_data, nullptr};

    return binary_function_outer<DenseFTInplaceMul>(out_obj, rhs_obj, config);
}

/*******************************************************************************
 * free tensor antipode
 ******************************************************************************/
namespace {

template <typename S>
struct DenseAntipode {
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {0, 0};

    CallConfig const* config_;

    template <typename OutIter, typename ArgIter>
    void operator()(
            OutIter out_iter,
            ArgIter arg_iter,
            CallConfig const& config
    ) const
    {}

    explicit constexpr DenseAntipode(CallConfig const& config)
        : config_(&config)
    {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const
    {
        auto const* basis
                = static_cast<TensorBasis const*>(config_->basis_data[0]);

        DenseTensorView<OutIter> out(
                out_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );
        DenseTensorView<ArgIter> arg(
                arg_iter,
                *basis,
                config_->degree_bounds[1].min_degree,
                config_->degree_bounds[1].max_degree
        );

        basic::ft_antipode(
                out,
                arg,
                basic::BasicAntipodeConfig{},
                basic::DefaultSigner{}
        );
    }
};
}// namespace

PyObject* py_dense_antipode(
        PyObject* self [[maybe_unused]],
        PyObject* args,
        PyObject* kwargs
)
{
    static constexpr char const* const kwords[]
            = {"out", "arg", "basis", "out_depth", "arg_depth", nullptr};

    PyObject *out_obj, *arg_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOO|ii",
                RPY_PY_KWORD_CAST(kwords),
                &out_obj,
                &arg_obj,
                &basis_obj,
                &degree_bounds[1].max_degree
        )) {
        return nullptr;
    }

    const BasisBase* basis_data[1];
    TensorBasis basis;
    const auto handle = to_basis(basis_obj, basis);
    if (!handle) {
        // error already set
        return nullptr;
    }
    basis_data[0] = &basis;

    CallConfig config{degree_bounds.data(), basis_data, nullptr};

    return binary_function_outer<DenseAntipode>(out_obj, arg_obj, config);
}

/*******************************************************************************
 * Free tensor left multiplication adjoint
 ******************************************************************************/

namespace {

template <typename S>
struct DenseFTAdjLMul {
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 3;
    static constexpr npy_intp arg_basis_mapping[3] = {0, 0, 0};

    const CallConfig* config_;

    explicit DenseFTAdjLMul(CallConfig const& config) : config_(&config) {}

    template <typename OutIter, typename OpIter, typename ArgIter>
    void operator()(OutIter out_iter, OpIter op_iter, ArgIter arg_iter) const
    {
        auto const* basis
                = static_cast<TensorBasis const*>(config_->basis_data[0]);

        DenseTensorView<OutIter> out(
                out_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );

        DenseTensorView<OpIter> op(
                op_iter,
                *basis,
                config_->degree_bounds[1].min_degree,
                config_->degree_bounds[1].max_degree
        );

        DenseTensorView<ArgIter> arg(
                arg_iter,
                *basis,
                config_->degree_bounds[2].min_degree,
                config_->degree_bounds[2].max_degree
        );

        basic::ft_adj_lmul(out, op, arg);
    }
};

}// namespace

PyObject* py_dense_ft_adj_lmul(
        PyObject* self [[maybe_unused]],
        PyObject* args,
        PyObject* kwargs
)
{
    static constexpr char const* const kwords[]
            = {"out",
               "operator",
               "argument",
               "basis",
               "out_depth",
               "lhs_depth",
               "rhs_depth",
               nullptr};

    PyObject *out_obj, *op_obj, *arg_obj;

    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 3> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOOO|iii",
                RPY_PY_KWORD_CAST(kwords),
                &out_obj,
                &op_obj,
                &arg_obj,
                &basis_obj,
                &degree_bounds[0].max_degree,
                &degree_bounds[1].max_degree,
                &degree_bounds[2].max_degree
        )) {
        return nullptr;
    }

    const BasisBase* basis_data[1];

    TensorBasis basis;
    const auto handle = to_basis(basis_obj, basis);
    if (!handle) { return nullptr; }
    basis_data[0] = &basis;

    CallConfig config{degree_bounds.data(), basis_data, nullptr};

    return ternary_function_outer<DenseFTAdjLMul>(
            out_obj,
            op_obj,
            arg_obj,
            config
    );
}

/*******************************************************************************
 * Lie to tensor
 ******************************************************************************/

namespace {

template <typename S>
struct DenseSTFma
{
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;
    static constexpr npy_intp n_args = 3;
    static constexpr npy_intp arg_basis_mapping[3] = {0, 0, 0};

    const CallConfig* config_;

    explicit DenseSTFma(CallConfig const& config) : config_(&config) {}

    template <typename OutIter, typename LhsIter, typename RhsIter>
    void operator()(OutIter out_iter, LhsIter lhs_iter, RhsIter rhs_iter)
    {
        auto const* tensor_basis
                = static_cast<const TensorBasis *>(config_->basis_data[0]);


        DenseTensorView<OutIter> out_view(
            out_iter,
            *tensor_basis,
            config_->degree_bounds[0].min_degree,
            config_->degree_bounds[0].max_degree
            );

        DenseTensorView<LhsIter> lhs_view(
            lhs_iter, *tensor_basis,
            config_->degree_bounds[1].min_degree,
            config_->degree_bounds[1].max_degree
            );

        DenseTensorView<RhsIter> rhs_view(
            rhs_iter, *tensor_basis,
            config_->degree_bounds[2].min_degree,
            config_->degree_bounds[2].max_degree
        );

        basic::st_fma(out_view, lhs_view, rhs_view);
    }
};

}

PyObject* py_dense_st_fma(PyObject* Py_UNUSED(self), PyObject* args, PyObject* kwargs)
{
    static constexpr char const* const kwords[] = {
        "out", "lhs", "rhs", "basis", nullptr
    };

    PyObject* out_obj, *lhs_obj, *rhs_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 3> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs,
        "OOOO",
        RPY_PY_KWORD_CAST(kwords),
        &out_obj,
        &lhs_obj,
        &rhs_obj,
        &basis_obj
        )) {
        return nullptr;
    }

    const BasisBase* basis_data[1];

    TensorBasis basis;
    const auto degree_begin_handle = to_basis(basis_obj, basis);

    if (!degree_begin_handle) {
        // error already set
        return nullptr;
    }

    basis_data[0] = &basis;

    CallConfig config {
        degree_bounds.data(),
        basis_data,
        nullptr
    };

    return ternary_function_outer<DenseSTFma>(out_obj, lhs_obj, rhs_obj, config);
}



/*******************************************************************************
 * Lie to tensor
 ******************************************************************************/

/*
 * Lie to tensor is a bit different from other operations because it mixes the
 * basis types for the arguments and carries an additional matrix sparse matrix
 * that has to be passed down to the driver routine. This matrix can be either
 * CSC or CSR format (with CSC being the default obtained from the PyLieBasis
 * we defined). We have to support both. Since this is different from the usual
 * process, we're going to not use the wrapping code and instead write the
 * array type checking by hand.
 */

namespace {

struct CompressedMatrixData {
    const npy_intp* indptr;
    const npy_intp* indices;
    const void* data;
    npy_intp nnz;
    npy_intp n_offsets;
    npy_intp n_inner;
    CompressedDim format;
};

template <typename S, CompressedDim Compression = CompressedCol>
struct DenseLieToTensor {
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {1, 0};

    using Matrix = CompressedMatrix<
            const S*,
            const npy_intp*,
            const npy_intp*,
            Compression>;

    const CallConfig* config_;
    const CompressedMatrixData* matrix_;

    explicit constexpr DenseLieToTensor(
            const CallConfig& config,
            const CompressedMatrixData& matrix
    )
        : config_(&config),
          matrix_(&matrix)
    {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const noexcept
    {
        const auto& lie_basis
                = *static_cast<const LieBasis*>(config_->basis_data[0]);
        const auto& tensor_basis
                = *static_cast<const TensorBasis*>(config_->basis_data[1]);

        DenseVectorFragment<OutIter> out(out_iter, tensor_basis.size());

        DenseVectorFragment<ArgIter> arg(arg_iter, lie_basis.size());

        Matrix matrix{
                static_cast<const S*>(matrix_->data),
                matrix_->indices,
                matrix_->nnz,
                matrix_->indptr,
                matrix_->n_offsets,
                matrix_->n_inner
        };

        if (config_->ops != nullptr) {
            basic::apply_sparse_linear_map(
                    out,
                    matrix,
                    arg,
                    ops::MultiplyBy<Scalar>{*static_cast<const S*>(config_->ops)
                    }
            );
        } else {
            basic::apply_sparse_linear_map(out, matrix, arg);
        }
    }
};

struct SparseMatrixArrays {
    PyArrayObject* indptr = nullptr;
    PyArrayObject* indices = nullptr;
    PyArrayObject* data = nullptr;

    SparseMatrixArrays() = default;

    SparseMatrixArrays(SparseMatrixArrays&& old) noexcept
    {
        indptr = old.indptr;
        old.indptr = nullptr;
        indices = old.indices;
        old.indices = nullptr;
        data = old.data;
        old.data = nullptr;
    }

    SparseMatrixArrays& operator=(SparseMatrixArrays&& old) noexcept
    {
        if (this != &old) {
            indptr = old.indptr;
            old.indptr = nullptr;
            indices = old.indices;
            old.indices = nullptr;
            data = old.data;
            old.data = nullptr;
        }
        return *this;
    }

    ~SparseMatrixArrays()
    {
        Py_XDECREF(indptr);
        Py_XDECREF(indices);
        Py_XDECREF(data);
    }

    explicit operator bool() const noexcept
    {
        return indptr != nullptr && indices != nullptr && data != nullptr;
    }
};

SparseMatrixArrays get_sparse_matrix(
        CompressedMatrixData& matrix_data,
        PyObject* matrix_obj,
        const npy_intp out_dim,
        const npy_intp arg_dim,
        PyArray_Descr* data_dtype
)
{
    SparseMatrixArrays arrays;

    PyObject* attr = PyObject_GetAttrString(matrix_obj, "indptr");
    if (!attr) { return {}; }

    // PyArray_FromAny steals a reference to dtype, so increment before we go.
    PyArray_Descr* intp_descr = PyArray_DescrFromType(NPY_INTP);
    Py_INCREF(intp_descr);

    arrays.indptr = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
            attr,
            intp_descr,
            1,
            1,
            NPY_ARRAY_CARRAY_RO,
            nullptr
    ));
    Py_DECREF(attr);
    if (!arrays.indptr) {
        Py_DECREF(intp_descr);
        return {};
    }

    attr = PyObject_GetAttrString(matrix_obj, "indices");
    if (!attr) { return {}; }
    arrays.indices = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
            attr,
            intp_descr,
            1,
            1,
            NPY_ARRAY_CARRAY_RO,
            nullptr
    ));
    Py_DECREF(attr);
    if (!arrays.indices) { return {}; }

    // intp_descr is no longer safely usable, it has been stolen twice.
    // ReSharper disable once CppDFAUnusedValue
    intp_descr = nullptr;

    // PyArray_DESCR gets a borrowed reference to dtype
    Py_INCREF(data_dtype);
    attr = PyObject_GetAttrString(matrix_obj, "data");
    if (!attr) { return {}; }
    arrays.data = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
            attr,
            data_dtype,
            1,
            1,
            NPY_ARRAY_CARRAY_RO,
            nullptr
    ));
    Py_DECREF(attr);
    if (!arrays.data) { return {}; }

    // get the shape of the matrix, we need this to determine if it is csc or
    // csr format and check the consistency with the input args
    npy_intp nrows, ncols;
    {
        PyObject* l2t_shape = PyObject_GetAttrString(matrix_obj, "shape");
        if (l2t_shape == nullptr) { return {}; }

        if (!PyArg_ParseTuple(l2t_shape, "nn", &nrows, &ncols)) {
            Py_DECREF(l2t_shape);
            return {};
        }
        Py_DECREF(l2t_shape);
    }

    // Internal consistency checks. It is not necessary that the matrix shape
    // matches precisely the dimensions of the input/output matrix since we
    // might be computing a truncated output, and this is handled by the
    // driver routine. However, the nrows must be at least the dimension of
    // the output tensor and ncols at least the dimension of the input Lie
    if (nrows != out_dim) {
        PyErr_Format(
                PyExc_ValueError,
                "the provided matrix does not achieve the required "
                "depth required for the output of dimension %zd",
                out_dim
        );
        return {};
    }
    if (ncols != arg_dim) {
        PyErr_Format(
                PyExc_ValueError,
                "the provided matrix does not achieve the required "
                "depth required for the input of dimension %zd",
                arg_dim
        );
        return {};
    }

    // make sure the data points to a valid compressed sparse matrix
    const npy_intp nnz = PyArray_SIZE(arrays.data);
    if (nnz != PyArray_SHAPE(arrays.indices)[0]) {
        PyErr_SetString(
                PyExc_ValueError,
                "mismatched size between data and indices "
                "both should have size equal to number of non-zero elements"
        );
        return {};
    }

    // Note: there is little risk of confusion here because for all practical
    // width/depth combinations the dimension of tensors and Lies will not
    // be equal. But to be sure, fall back to the getformat method in this case.
    const npy_intp* indptr_shape = PyArray_SHAPE(arrays.indptr);

    matrix_data.indptr
            = static_cast<const npy_intp*>(PyArray_DATA(arrays.indptr));
    matrix_data.indices
            = static_cast<const npy_intp*>(PyArray_DATA(arrays.indices));
    matrix_data.data = PyArray_DATA(arrays.data);
    matrix_data.nnz = nnz;

    if (nrows == ncols) {
        PyObject* l2t_format
                = PyObject_CallMethod(matrix_obj, "getformat", nullptr);
        if (l2t_format == nullptr) { return {}; }

        if (PyUnicode_CompareWithASCIIString(matrix_obj, "csc") == 0) {
            matrix_data.format = CompressedCol;
            matrix_data.n_offsets = ncols;
            matrix_data.n_inner = nrows;
        } else if (PyUnicode_CompareWithASCIIString(matrix_obj, "csr") == 0) {
            matrix_data.format = CompressedRow;
            matrix_data.n_offsets = nrows;
            matrix_data.n_inner = ncols;
        } else {
            Py_DECREF(l2t_format);
            PyErr_SetString(PyExc_ValueError, "invalid format for l2t matrix");
            return {};
        }

        Py_DECREF(l2t_format);
    } else if (indptr_shape[0] == nrows + 1) {
        matrix_data.format = CompressedRow;
        matrix_data.n_offsets = nrows;
        matrix_data.n_inner = ncols;
    } else if (indptr_shape[0] == ncols + 1) {
        matrix_data.format = CompressedCol;
        matrix_data.n_offsets = ncols;
        matrix_data.n_inner = nrows;
    } else {
        PyErr_SetString(PyExc_ValueError, "invalid shape for indptr array");
        return {};
    }

    return arrays;
}

bool get_scale_factor(
        PyObject* scale_factor_obj,
        void* scalar_scratch,
        CallConfig& config,
        PyArray_Descr* out_dtype
)
{
    if (scale_factor_obj == nullptr || Py_IsNone(scale_factor_obj)) {
        return true;
    }

    if (PyArray_IsScalar(scale_factor_obj, Floating)) {
        if (PyArray_CastScalarToCtype(
                    scale_factor_obj,
                    scalar_scratch,
                    out_dtype
            )
            < 0) {
            return false;
        }
    } else {
        PyErr_SetString(
                PyExc_ValueError,
                "scale_factor must be a numpy floating point scalar"
        );
        return false;
    }

    config.ops = scalar_scratch;

    return true;
}

}// namespace

PyObject* py_dense_lie_to_tensor(
        PyObject* Py_UNUSED(self),
        PyObject* args,
        PyObject* kwargs
)
{
    static constexpr char const* const kwords[]
            = {"out",
               "arg",
               "l2t_matrix",
               "lie_basis",
               "tensor_basis",
               "scale_factor",
               nullptr};

    PyObject *out_obj, *arg_obj, *l2t_matrix;
    PyObject* lie_basis_obj = nullptr;
    PyObject* tensor_basis_obj = nullptr;
    PyObject* scale_factor_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;

    alignas(16) std::array<std::byte, 16> scalar_scratch;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOOO|OO",
                RPY_PY_KWORD_CAST(kwords),
                &out_obj,
                &arg_obj,
                &l2t_matrix,
                &lie_basis_obj,
                &tensor_basis_obj,
                &scale_factor_obj
        )) {
        return nullptr;
    }
    const BasisBase* basis_data[2];

    LieBasis lie_basis;
    const auto lie_basis_handle = to_basis(lie_basis_obj, lie_basis);
    if (!lie_basis_handle) { return nullptr; }
    basis_data[0] = &lie_basis;

    TensorBasis tensor_basis;
    PyObjHandle tensor_basis_handle;
    if (tensor_basis_obj != nullptr) {
        tensor_basis_handle = to_basis(tensor_basis_obj, tensor_basis);
        if (tensor_basis.width != lie_basis.width) {
            PyErr_SetString(
                    PyExc_ValueError,
                    "mismatched width for Lie and tensor bases"
            );
            return nullptr;
        }
    } else {
        auto* new_tb = PyTensorBasis_get(lie_basis.width, lie_basis.depth);
        if (new_tb == nullptr) { return nullptr; }

        tensor_basis_handle.reset(reinterpret_cast<PyObject*>(new_tb));

        tensor_basis.width = lie_basis.width;
        tensor_basis.depth = lie_basis.depth;

        // the newly constructed Tensor basis is guaranteed to have a contiguous
        // degree_begin array.
        tensor_basis.degree_begin = static_cast<npy_intp*>(PyArray_DATA(
               PyTensorBasis_degree_begin(new_tb)
        ));
    }

    basis_data[1] = &tensor_basis;

    CallConfig config{degree_bounds.data(), basis_data, nullptr};

    if (!update_algebra_params(
                config,
                DenseLieToTensor<float>::n_args,
                DenseLieToTensor<float>::arg_basis_mapping
        )) {
        return nullptr;
    }

    if (!PyArray_Check(out_obj)) {
        PyErr_SetString(PyExc_ValueError, "out must be a numpy aray");
        return nullptr;
    }
    auto* out_arr = reinterpret_cast<PyArrayObject*>(out_obj);

    const auto n_dims = PyArray_NDIM(out_arr);
    const auto dtype = PyArray_TYPE(out_arr);
    const auto* shape = PyArray_DIMS(out_arr);

    if (n_dims < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid shape");
        return nullptr;
    }

    // PyArray_Descr* dtype_descr = PyArray_DescrFromType(dtype);
    // Py_INCREF(dtype_descr);
    PyObjHandle arg_data(
            PyArray_ContiguousFromAny(arg_obj, dtype, n_dims, n_dims),
            false
    );
    if (!arg_data) { return nullptr; }

    auto* arg_arr = reinterpret_cast<PyArrayObject*>(arg_data.obj());

    const auto arg_ndims = PyArray_NDIM(arg_arr);
    if (arg_ndims != n_dims) {
        PyErr_SetString(
                PyExc_ValueError,
                "mismatch between argument dimensions and output dimensions"
        );
        return nullptr;
    }

    const auto* arg_shape = PyArray_DIMS(arg_arr);
    if (!check_dims(arg_shape, arg_ndims - 1, shape, n_dims - 1)) {
        PyErr_SetString(
                PyExc_ValueError,
                "arg and out must have the same shape"
        );
        return nullptr;
    }

    // The sparse matrix needs special attention. These can be any object that
    // provides compressed-sparse matrix elements (indptr, indices, data) and
    // might be either compressed row or compressed column format. We must also
    // accommodate both options.
    CompressedMatrixData matrix_data;
    const SparseMatrixArrays arrays = get_sparse_matrix(
            matrix_data,
            l2t_matrix,
            shape[n_dims - 1],
            arg_shape[n_dims - 1],
            PyArray_DESCR(out_arr)
    );

    if (!arrays) { return nullptr; }

    if (!get_scale_factor(
                scale_factor_obj,
                scalar_scratch.data(),
                config,
                PyArray_DESCR(out_arr)
        )) {
        return nullptr;
    }

#define RPC_SM_FORMAT_SWITCH(Scalar, format)                                   \
    switch (format) {                                                          \
        case CompressedCol:                                                    \
            return outer_loop_binary(                                          \
                    out_arr,                                                   \
                    arg_arr,                                                   \
                    DenseLieToTensor<Scalar, CompressedCol>{                   \
                            config,                                            \
                            matrix_data                                        \
                    }                                                          \
            );                                                                 \
        case CompressedRow:                                                    \
            return outer_loop_binary(                                          \
                    out_arr,                                                   \
                    arg_arr,                                                   \
                    DenseLieToTensor<Scalar, CompressedRow>{                   \
                            config,                                            \
                            matrix_data                                        \
                    }                                                          \
            );                                                                 \
    }

    switch (dtype) {
        case NPY_FLOAT32: RPC_SM_FORMAT_SWITCH(float, matrix_data.format);
        case NPY_FLOAT64: RPC_SM_FORMAT_SWITCH(double, matrix_data.format);
        default:
            PyErr_SetString(
                    PyExc_ValueError,
                    "unsupported dtype for l2t matrix"
            );
            return nullptr;
    }

#undef RPC_SM_FORMAT_SWITCH
}

/*******************************************************************************
 * Tensor to Lie
 ******************************************************************************/

namespace {

template <typename S, CompressedDim Compression = CompressedCol>
struct DenseTensorToLie {
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {0, 1};

    using Matrix = CompressedMatrix<
            const S*,
            const npy_intp*,
            const npy_intp*,
            Compression>;

    const CallConfig* config_;
    const CompressedMatrixData* matrix_;

    explicit constexpr DenseTensorToLie(
            const CallConfig& config,
            const CompressedMatrixData& matrix
    )
        : config_(&config),
          matrix_(&matrix)
    {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const noexcept
    {
        const auto& lie_basis
                = *static_cast<const LieBasis*>(config_->basis_data[0]);
        const auto& tensor_basis
                = *static_cast<const TensorBasis*>(config_->basis_data[1]);

        DenseVectorFragment<OutIter> out(out_iter, lie_basis.size());

        DenseVectorFragment<ArgIter> arg(arg_iter, tensor_basis.size());

        Matrix matrix{
                static_cast<const S*>(matrix_->data),
                matrix_->indices,
                matrix_->nnz,
                matrix_->indptr,
                matrix_->n_offsets,
                matrix_->n_inner
        };

        if (config_->ops != nullptr) {
            basic::apply_sparse_linear_map(
                    out,
                    matrix,
                    arg,
                    ops::MultiplyBy<S>{*static_cast<const S*>(config_->ops)}
            );
        } else {
            basic::apply_sparse_linear_map(out, matrix, arg);
        }
    }
};

}// namespace

PyObject* py_dense_tensor_to_lie(
        PyObject* Py_UNUSED(self),
        PyObject* args,
        PyObject* kwargs
)
{
    static constexpr char const* const kwords[]
            = {"out",
               "arg",
               "t2l_matrix",
               "lie_basis",
               "tensor_basis",
               "scale_factor",
               nullptr};

    PyObject *out_obj, *arg_obj, *t2l_matrix;
    PyObject* lie_basis_obj = nullptr;
    PyObject* tensor_basis_obj = nullptr;
    PyObject* scale_factor_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;
    alignas(16) std::array<std::byte, 16> scalar_scratch;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOOO|OO",
                RPY_PY_KWORD_CAST(kwords),
                &out_obj,
                &arg_obj,
                &t2l_matrix,
                &lie_basis_obj,
                &tensor_basis_obj,
                &scale_factor_obj
        )) {
        return nullptr;
    }

    const BasisBase* basis_data[2];

    LieBasis lie_basis;
    const auto lie_basis_handle = to_basis(lie_basis_obj, lie_basis);
    if (!lie_basis_handle) { return nullptr; }
    basis_data[0] = &lie_basis;

    TensorBasis tensor_basis;
    PyObjHandle tensor_basis_handle;
    if (tensor_basis_obj != nullptr) {
        tensor_basis_handle = to_basis(tensor_basis_obj, tensor_basis);
        if (tensor_basis.width != lie_basis.width) {
            PyErr_SetString(
                    PyExc_ValueError,
                    "mismatched width for Lie and tensor bases"
            );
            return nullptr;
        }
    } else {
        auto* new_tb = PyTensorBasis_get(lie_basis.width, lie_basis.depth);
        if (new_tb == nullptr) { return nullptr; }

        tensor_basis_handle.reset(reinterpret_cast<PyObject*>(new_tb));

        tensor_basis.width = lie_basis.width;
        tensor_basis.depth = lie_basis.depth;

        // the newly constructed Tensor basis is guaranteed to have a contiguous
        // degree_begin array.
        tensor_basis.degree_begin = static_cast<npy_intp*>(PyArray_DATA(
            PyTensorBasis_degree_begin(new_tb)
        ));
    }

    basis_data[1] = &tensor_basis;

    CallConfig config{degree_bounds.data(), basis_data, nullptr};

    if (!update_algebra_params(
                config,
                DenseTensorToLie<float>::n_args,
                DenseTensorToLie<float>::arg_basis_mapping
        )) {
        return nullptr;
    }

    if (!PyArray_Check(out_obj)) {
        PyErr_SetString(PyExc_ValueError, "out must be a numpy aray");
        return nullptr;
    }
    auto* out_arr = reinterpret_cast<PyArrayObject*>(out_obj);

    const auto n_dims = PyArray_NDIM(out_arr);
    const auto dtype = PyArray_TYPE(out_arr);
    const auto* shape = PyArray_DIMS(out_arr);

    if (n_dims < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid shape");
        return nullptr;
    }

    PyObjHandle arg_data(
            PyArray_ContiguousFromAny(arg_obj, dtype, n_dims, n_dims),
            false
    );
    if (!arg_data) { return nullptr; }

    auto* arg_arr = reinterpret_cast<PyArrayObject*>(arg_data.obj());

    const auto arg_ndims = PyArray_NDIM(arg_arr);
    if (arg_ndims != n_dims) {
        PyErr_SetString(
                PyExc_ValueError,
                "mismatch between argument dimensions and output dimensions"
        );
        return nullptr;
    }

    const auto* arg_shape = PyArray_DIMS(arg_arr);
    if (!check_dims(arg_shape, arg_ndims - 1, shape, n_dims - 1)) {
        PyErr_SetString(
                PyExc_ValueError,
                "arg and out must have the same shape"
        );
        return nullptr;
    }

    // The sparse matrix needs special attention. These can be any object that
    // provides compressed-sparse matrix elements (indptr, indices, data) and
    // might be either compressed row or compressed column format. We must also
    // accommodate both options.
    CompressedMatrixData matrix_data;
    const SparseMatrixArrays arrays = get_sparse_matrix(
            matrix_data,
            t2l_matrix,
            shape[n_dims - 1],
            arg_shape[n_dims - 1],
            PyArray_DESCR(out_arr)
    );

    if (!arrays) { return nullptr; }

    if (!get_scale_factor(
                scale_factor_obj,
                scalar_scratch.data(),
                config,
                PyArray_DESCR(out_arr)
        )) {
        return nullptr;
    }

#define RPC_SM_FORMAT_SWITCH(Scalar, format)                                   \
    switch (format) {                                                          \
        case CompressedCol:                                                    \
            return outer_loop_binary(                                          \
                    out_arr,                                                   \
                    arg_arr,                                                   \
                    DenseTensorToLie<Scalar, CompressedCol>{                   \
                            config,                                            \
                            matrix_data                                        \
                    }                                                          \
            );                                                                 \
        case CompressedRow:                                                    \
            return outer_loop_binary(                                          \
                    out_arr,                                                   \
                    arg_arr,                                                   \
                    DenseTensorToLie<Scalar, CompressedRow>{                   \
                            config,                                            \
                            matrix_data                                        \
                    }                                                          \
            );                                                                 \
    }

    switch (dtype) {
        case NPY_FLOAT32: RPC_SM_FORMAT_SWITCH(float, matrix_data.format);
        case NPY_FLOAT64: RPC_SM_FORMAT_SWITCH(double, matrix_data.format);
        default:
            PyErr_SetString(
                    PyExc_ValueError,
                    "unsupported dtype for l2t matrix"
            );
            return nullptr;
    }

#undef RPC_SM_FORMAT_SWITCH
}