// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "scalars.h"

#include <sstream>

#include <pybind11/operators.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/scalar_types.h>

#include "dlpack.h"
#include "r_py_polynomial.h"
// #include "scalar.h"
#include "scalar_type.h"
#include "pytype_conversion.h"

#include "args/dlpack_helpers.h"

using namespace rpy;
using namespace rpy::python;
using namespace pybind11::literals;

static const char* SCALAR_DOC = R"edoc(
A generic scalar value.
)edoc";

void python::assign_py_object_to_scalar(
        scalars::Scalar& dst,
        pybind11::handle object
)
{
    if (py::isinstance<py::float_>(object)) {
        dst = object.cast<double>();
    } else if (py::isinstance<py::int_>(object)) {
        dst = object.cast<int64_t>();
    } else if (RPyPolynomial_Check(object.ptr())) {
        dst = RPyPolynomial_cast(object.ptr());
    } else if (py::isinstance<scalars::Scalar>(object)) {
        dst = object.cast<const scalars::Scalar&>();
    } else {
        // TODO: other checks

        auto tp = py::str(py::type::of(object));
        RPY_THROW(
                py::value_error,
                "bad conversion from " + tp.cast<string>() + " to "
                        + string((*dst.type())->name())
        );
    }
}

void python::init_scalars(pybind11::module_& m)
{
    using scalars::Scalar;
    py::options options;
    options.disable_function_signatures();

    python::init_scalar_types(m);
    python::init_monomial(m);
    m.add_object("SCALAR_MAPPING", python::init_scalar_mapping());
    //
    // if (PyType_Ready(&PyScalar_Type) < 0) {
    //     throw py::error_already_set();
    // }
    // m.add_object("Scalar", (PyObject*) &PyScalar_Type);

    py::class_<Scalar> klass(m, "Scalar", SCALAR_DOC);

    klass.def("scalar_type", [](const Scalar& self) {
        return scalar_type_to_py_type(*self.type());
    });

    RPY_WARNING_PUSH
    RPY_CLANG_DISABLE_WARNING(-Wself-assign-overloaded)

    klass.def(-py::self);
    //    klass.def(py::self + py::self);
    //    klass.def(py::self - py::self);
    //    klass.def(py::self * py::self);
    //    klass.def(py::self / py::self);
    //
    //    klass.def(py::self += py::self);
    //    klass.def(py::self -= py::self);
    //    klass.def(py::self *= py::self);
    //    klass.def(py::self /= py::self);

    //    klass.def(py::self == py::self);
    //    klass.def(py::self != py::self);

    klass.def("__add__", [](const Scalar& self, const Scalar& other) {
        return self + other;
    });
    klass.def("__add__", [](const Scalar& self, scalar_t other) {
        return self + Scalar(other);
    });
    klass.def("__add__", [](const Scalar& self, long long other) {
        return self + Scalar(other);
    });
    klass.def("__radd__", [](const Scalar& self, scalar_t other) {
        return Scalar(other) + self;
    });
    klass.def("__radd__", [](const Scalar& self, long long other) {
        return Scalar(other) + self;
    });
    klass.def("__iadd__", [](Scalar& self, const Scalar& other) {
        return self += other;
    });
    klass.def("__iadd__", [](Scalar& self, scalar_t other) {
        return self += Scalar(other);
    });
    klass.def("__iadd__", [](Scalar& self, long long other) {
        return self += Scalar(other);
    });

    klass.def("__sub__", [](const Scalar& self, scalar_t other) {
        return self - Scalar(other);
    });
    klass.def("__sub__", [](const Scalar& self, long long other) {
        return self - Scalar(other);
    });
    klass.def("__rsub__", [](const Scalar& self, scalar_t other) {
        return Scalar(other) - self;
    });
    klass.def("__rsub__", [](const Scalar& self, long long other) {
        return Scalar(other) - self;
    });
    klass.def("__isub__", [](Scalar& self, scalar_t other) {
        return self -= Scalar(other);
    });
    klass.def("__isub__", [](Scalar& self, long long other) {
        return self -= Scalar(other);
    });

    klass.def("__mul__", [](const Scalar& self, const Scalar& other) {
        return self * other;
    });
    klass.def("__mul__", [](const Scalar& self, scalar_t other) {
        return self * Scalar(other);
    });
    klass.def("__mul__", [](const Scalar& self, long long other) {
        return self * Scalar(other);
    });
    klass.def("__rmul__", [](const Scalar& self, scalar_t other) {
        return Scalar(other) * self;
    });
    klass.def("__rmul__", [](const Scalar& self, long long other) {
        return Scalar(other) * self;
    });
    klass.def("__imul__", [](Scalar& self, const Scalar& other) {
        return self *= other;
    });
    klass.def("__imul__", [](Scalar& self, scalar_t other) {
        return self *= Scalar(other);
    });
    klass.def("__imul__", [](Scalar& self, long long other) {
        return self *= Scalar(other);
    });

    klass.def("__div__", [](const Scalar& self, scalar_t other) {
        if (other == 0.0) { throw py::value_error("division by zero"); }
        return self / Scalar(other);
    });
    klass.def("__div__", [](const Scalar& self, long long other) {
        if (other == 0) { throw py::value_error("division by zero"); }
        return self / Scalar(other);
    });
    klass.def("__rdiv__", [](const Scalar& self, scalar_t other) {
        return Scalar(other) / self;
    });
    klass.def("__rdiv__", [](const Scalar& self, long long other) {
        return Scalar(other) / self;
    });
    klass.def("__idiv__", [](Scalar& self, scalar_t other) {
        if (other == 0.0) { throw py::value_error("division by zero"); }
        return self /= Scalar(other);
    });
    klass.def("__idiv__", [](Scalar& self, long long other) {
        if (other == 0) { throw py::value_error("division by zero"); }
        return self /= Scalar(other);
    });

    klass.def("__eq__", [](const Scalar& lhs, const Scalar& rhs) {
        return lhs == rhs;
    });
    klass.def("__eq__", [](const Scalar& self, scalar_t other) {
        return self == Scalar(other);
    });
    klass.def("__eq__", [](scalar_t other, const Scalar& self) {
        return self == Scalar(other);
    });
    klass.def("__eq__", [](const Scalar& self, long long other) {
        return self == Scalar(other);
    });
    klass.def("__eq__", [](long long other, const Scalar& self) {
        return self == Scalar(other);
    });

    klass.def("__ne__", [](const Scalar& lhs, const Scalar& rhs) {
        return lhs != rhs;
    });
    klass.def("__ne__", [](const Scalar& self, scalar_t other) {
        return self != Scalar(other);
    });
    klass.def("__ne__", [](scalar_t other, const Scalar& self) {
        return self != Scalar(other);
    });
    klass.def("__ne__", [](const Scalar& self, long long other) {
        return self != Scalar(other);
    });
    klass.def("__ne__", [](long long other, const Scalar& self) {
        return self != Scalar(other);
    });

    klass.def("__str__", [](const Scalar& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });
    klass.def("__repr__", [](const Scalar& self) {
        std::stringstream ss;
        ss << "Scalar(type=";
        auto tp = self.type();
        if (tp) {
            ss << string((*tp)->name());
        } else {
            auto info = self.type_info();
            if (info.code == devices::TypeCode::Int) {
                ss << "int" << CHAR_BIT * info.bytes;
            } else if (info.code == devices::TypeCode::UInt) {
                ss << "uint" << CHAR_BIT * info.bytes;
            }
        }
        ss << ", value approx " << self << ")";

        return ss.str();
    });

    klass.def("to_float", [](const Scalar& self) {
        return scalars::scalar_cast<double>(self);
    });

    RPY_WARNING_POP
}

/*
 * Everything that follows is for the implementation of py_to_buffer,
 * which is our primary way of constructing RoughPy objects from python
 * objects.
 */

#define DOCASE(NAME)                                                           \
    case static_cast<uint8_t>(scalars::ScalarTypeCode::NAME):                  \
        type = scalars::ScalarTypeCode::NAME;                                  \
        break

RPY_UNUSED static const scalars::ScalarType*
dlpack_dtype_to_scalar_type(DLDataType dtype, DLDevice device)
{
    using rpy::devices::DeviceType;

    scalars::ScalarTypeCode type;
    switch (dtype.code) {
        DOCASE(Float);
        DOCASE(Int);
        DOCASE(UInt);
        DOCASE(OpaqueHandle);
        DOCASE(BFloat);
        DOCASE(Complex);
        DOCASE(Bool);
        default: RPY_THROW(std::runtime_error, "unrecognised type code");
    }

    return *scalars::scalar_type_of(
            {type,
             static_cast<uint8_t>(dtype.bits / CHAR_BIT),
             static_cast<uint8_t>(dtype.bits / CHAR_BIT),
             static_cast<uint8_t>(dtype.lanes)}
            // {static_cast<DeviceType>(device.device_type), device.device_id}
    );
}

#undef DOCASE

static inline void dl_copy_strided(
        std::int32_t ndim,
        std::int64_t* shape,
        std::int64_t* strides,
        const scalars::ScalarArray& src,
        scalars::ScalarArray& dst
)
{
    const auto dst_type_o = dst.type();
    RPY_DBG_ASSERT(dst_type_o);
    const auto* dst_type = *dst_type_o;
    if (ndim == 1) {

        if (strides[0] == 1) {
            dst_type->convert_copy(dst, src);
        } else {
            for (std::int64_t i = 0; i < shape[0]; ++i) {
                dst[i] = src[i * strides[0]];
            }
        }
    } else {

        auto* next_shape = shape + 1;
        auto* next_stride = strides + 1;

        for (std::int64_t j = 0; j < shape[0]; ++j) {
            auto src_inner
                    = src[{static_cast<dimn_t>(j * strides[0]), src.size()}];
            auto dst_inner
                    = dst[{static_cast<dimn_t>(j * shape[0]), dst.size()}];
            dl_copy_strided(
                    ndim - 1,
                    next_shape,
                    next_stride,
                    src_inner,
                    dst_inner
            );
        }
    }
}

static inline void update_dtype_and_allocate(
        scalars::KeyScalarArray& result,
        python::PyToBufferOptions& options,
        idimn_t no_values,
        idimn_t no_keys
)
{

    if (options.type != nullptr) {
        result = scalars::KeyScalarArray(options.type);
        result.allocate_scalars(no_values);
        result.allocate_keys(no_keys);
    } else if (no_values > 0) {
        RPY_THROW(py::type_error, "unable to deduce a suitable scalar type");
    }
}

static bool try_fill_buffer_dlpack(
        scalars::KeyScalarArray& buffer,
        python::PyToBufferOptions& options,
        const py::handle& object
)
{
    py::capsule dlpack;
    dlpack = object.attr("__dlpack__")();

    auto* tensor = reinterpret_cast<DLManagedTensor*>(dlpack.get_pointer());
    if (tensor == nullptr) {
        RPY_THROW(py::value_error, "__dlpack__ returned invalid object");
    }

    auto& dltensor = tensor->dl_tensor;

    RPY_CHECK(
            dltensor.device.device_type == kDLCPU,
            "no device support is currently available"
    );

    auto* data = reinterpret_cast<char*>(dltensor.data);
    auto ndim = dltensor.ndim;
    auto* shape = dltensor.shape;
    auto* strides = dltensor.strides;

    // This function throws if no matching dtype is found

    const auto tensor_stype_info = convert_from_dl_datatype(dltensor.dtype);
    const auto tensor_stype = scalars::scalar_type_of(tensor_stype_info);

    if (options.type == nullptr) {
        if (tensor_stype) {
            options.type = *tensor_stype;
        } else {
            options.type = scalars::ScalarType::for_info(tensor_stype_info);
        }
    }
    RPY_DBG_ASSERT(options.type != nullptr);

    if (buffer.type() == nullptr) {
        buffer = scalars::KeyScalarArray(options.type);
    }

    if (data == nullptr) {
        // The array is empty, empty result.
        return true;
    }

    RPY_CHECK(shape != nullptr);
    options.shape.assign(shape, shape + ndim);

    idimn_t size = 1;
    for (auto i = 0; i < ndim; ++i) { size *= static_cast<idimn_t>(shape[i]); }

    if (strides == nullptr) {
        buffer.allocate_scalars(size);
        options.type->convert_copy(
                buffer,
                {tensor_stype_info, data, static_cast<dimn_t>(size)}
        );
    } else {
        buffer.allocate_scalars(size);
        dl_copy_strided(
                ndim,
                shape,
                strides,
                {tensor_stype_info, data, static_cast<dimn_t>(size)},
                buffer
        );
    }
    //
    //    if (tensor->deleter != nullptr) {
    //        options.cleanup = [tensor]() {
    //            tensor->deleter(tensor);
    //        };
    //    }

    return true;
}

static void
check_and_set_dtype(python::PyToBufferOptions& options, py::handle arg)
{
    if (options.type == nullptr) {

        if (py::isinstance<scalars::Scalar>(arg)) {
            const auto& scal = arg.cast<const scalars::Scalar&>();
            auto arg_type = scal.type();
            if (arg_type) {
                options.type = *arg_type;
            } else {
                options.type = scalars::ScalarType::for_info(scal.type_info());
            }
        } else if (options.no_check_imported) {
            options.type = *scalars::ScalarType::of<double>();
        } else {
            options.type = python::py_type_to_scalar_type(py::type::of(arg));
        }
    }
}

static bool check_ground_type(
        py::handle object,
        GroundDataType& ground_type,
        python::PyToBufferOptions& options
)
{
    py::handle scalar;
    if (python::is_scalar(object)) {
        if (ground_type == GroundDataType::UnSet) {
            ground_type = GroundDataType::Scalars;
        } else if (ground_type != GroundDataType::Scalars) {
            RPY_THROW(
                    py::value_error,
                    "inconsistent scalar/key-scalar-pair data"
            );
        }
        scalar = object;
    } else if (is_kv_pair(object, options.alternative_key)) {
        if (ground_type == GroundDataType::UnSet) {
            ground_type = GroundDataType::KeyValuePairs;
        } else if (ground_type != GroundDataType::KeyValuePairs) {
            RPY_THROW(
                    py::value_error,
                    "inconsistent scalar/key-scalar-pair data"
            );
        }
        scalar = object.cast<py::tuple>()[1];
    } else {
        // TODO: Check non-int/float scalar types
        return false;
    }

    check_and_set_dtype(options, scalar);

    // TODO: Insert check for compatibility if the scalar type is set

    return true;
}

static void compute_size_and_type_recurse(
        python::PyToBufferOptions& options,
        std::vector<py::object>& leaves,
        const py::handle& object,
        GroundDataType& ground_type,
        dimn_t depth
)
{

    if (!py::isinstance<py::sequence>(object)) {
        RPY_THROW(py::type_error, "unexpected type in array argument");
    }
    if (depth > options.max_nested) {
        RPY_THROW(
                py::value_error,
                "maximum nested array limit reached in this context"
        );
    }

    auto sequence = py::reinterpret_borrow<py::sequence>(object);
    auto length = static_cast<idimn_t>(py::len(sequence));

    if (options.shape.size() == depth) {
        // We've not visited this depth before,
        // add our length to the list
        options.shape.push_back(length);
    } else if (ground_type == GroundDataType::Scalars) {
        // We have visited this depth before,
        // check our length is consistent with the others
        if (length != options.shape[depth]) {
            RPY_THROW(py::value_error, "ragged arrays are not supported");
        }
    }

    if (length == 0) {
        // if the length is zero, there is nothing left to do
        return;
    }

    /*
     * Now we handle the meat of the recursion.
     * If we find scalars in side this level, then we're
     * at the bottom, and we stop recursing. Otherwise, we
     * find another layer of nested sequences and we
     * recurse into those.
     */

    auto item0 = sequence[0];
    if (check_ground_type(item0, ground_type, options)) {
        // We've hit the bottom, container holds either
        // scalars or key-scalar pairs.

        // Check all the scalar types are the same
        for (auto&& item : sequence) {
            check_ground_type(item, ground_type, options);
        }

        leaves.push_back(std::move(sequence));
    } else if (py::isinstance<py::sequence>(item0)) {
        for (auto&& sibling : sequence) {
            compute_size_and_type_recurse(
                    options,
                    leaves,
                    sibling,
                    ground_type,
                    depth + 1
            );
        }
    } else if (py::isinstance<py::dict>(item0)) {
        auto dict = py::reinterpret_borrow<py::dict>(item0);

        if (depth == options.max_nested) {
            RPY_THROW(
                    py::value_error,
                    "maximum nested depth reached in this context"
            );
        }
        switch (ground_type) {
            case GroundDataType::UnSet:
                ground_type = GroundDataType::KeyValuePairs;
            case GroundDataType::KeyValuePairs: break;
            default:
                RPY_THROW(py::type_error, "mismatched types in array argument");
        }

        if (!dict.empty()) {
            auto kv = *dict.begin();
            check_and_set_dtype(options, kv.second);
        }

        leaves.push_back(dict);

    } else {
        RPY_THROW(py::type_error, "unexpected type in array argument");
    }
}

ArgSizeInfo python::compute_size_and_type(
        python::PyToBufferOptions& options,
        std::vector<py::object>& leaves,
        py::handle arg
)
{
    ArgSizeInfo info = {0, 0};

    RPY_CHECK(py::isinstance<py::sequence>(arg));

    GroundDataType ground_type = GroundDataType::UnSet;
    compute_size_and_type_recurse(options, leaves, arg, ground_type, 0);

    if (ground_type == GroundDataType::KeyValuePairs) {
        options.shape.clear();

        for (const auto& obj : leaves) {
            auto size = static_cast<idimn_t>(py::len(obj));

            options.shape.push_back(size);
            info.num_values += size;
            info.num_keys += size;
        }
    } else {
        info.num_values = 1;
        for (auto& shape_i : options.shape) { info.num_values *= shape_i; }
    }

    if (info.num_values == 0 || ground_type == GroundDataType::UnSet) {
        options.shape.clear();
        leaves.clear();
    }

    return info;
}

static void handle_sequence_tuple(
        scalars::Scalar& scalar,
        key_type* key_ptr,
        py::handle tpl_o,
        python::PyToBufferOptions& options
)
{
    auto tpl = py::reinterpret_borrow<py::tuple>(tpl_o);
    auto key = tpl[0];
    if (options.alternative_key != nullptr
        && py::isinstance(key, options.alternative_key->py_key_type)) {
        *key_ptr = options.alternative_key->converter(key);
    } else {
        *key_ptr = key.cast<key_type>();
    }

    python::assign_py_object_to_scalar(scalar, tpl[1]);
}

static void handle_dict(
        scalars::ScalarArray& scalar_array,
        key_type*& key_ptr,
        python::PyToBufferOptions& options,
        py::handle dict_o
)
{
    dimn_t idx = 0;
    for (auto obj : py::reinterpret_borrow<py::dict>(dict_o)) {
        // dict iterator yields pairs [key, obj]
        // Expecting key-value tuples
        auto key = obj.first;
        if (options.alternative_key != nullptr
            && py::isinstance(key, options.alternative_key->py_key_type)) {
            *(key_ptr++) = options.alternative_key->converter(key);
        } else {
            *(key_ptr++) = key.cast<key_type>();
        }

        auto val = scalar_array[idx++];
        python::assign_py_object_to_scalar(val, obj.second);
    }
}

scalars::KeyScalarArray python::py_to_buffer(
        const py::handle& object,
        python::PyToBufferOptions& options
)
{
    scalars::KeyScalarArray result;

    // First handle the single number cases
    if (py::isinstance<py::float_>(object)
        || py::isinstance<py::int_>(object)) {
        if (!options.allow_scalar) {
            RPY_THROW(
                    py::value_error,
                    "scalar value not permitted in this context"
            );
        }

        check_and_set_dtype(options, object);
        update_dtype_and_allocate(result, options, 1, 0);

        auto val = result[0];
        assign_py_object_to_scalar(val, object);
    } else if (is_kv_pair(object, options.alternative_key)) {
        /*
         * Now for tuples of length 2, which we expect to be a kv-pair
         */
        auto tpl_arg = py::reinterpret_borrow<py::tuple>(object);

        auto value = tpl_arg[1];
        check_and_set_dtype(options, value);
        update_dtype_and_allocate(result, options, 1, 1);

        auto val = result[0];
        handle_sequence_tuple(val, result.keys(), object, options);
    } else if (py::hasattr(object, "__dlpack__")) {
        // If we used the dlpack interface, then the result is
        // already constructed.
        try_fill_buffer_dlpack(result, options, object);
    } else if (py::isinstance<py::buffer>(object)) {
        // Fall back to the buffer protocol
        auto info = py::reinterpret_borrow<py::buffer>(object).request();
        auto type_info = py_buffer_to_type_info(info);

        if (options.type == nullptr) {
            options.type = scalars::ScalarType::for_info(type_info);
            result = scalars::KeyScalarArray(options.type);
        }

        update_dtype_and_allocate(result, options, info.size, 0);

        // The only way type can still be null is if there are no elements.
        if (options.type != nullptr) {
            options.type->convert_copy(
                    result,
                    {*scalars::scalar_type_of(type_info),
                     info.ptr,
                     static_cast<dimn_t>(info.size)}
            );
            options.shape.assign(info.shape.begin(), info.shape.end());
        }

    } else if (py::isinstance<py::dict>(object)) {
        auto dict_arg = py::reinterpret_borrow<py::dict>(object);
        options.shape.push_back(static_cast<idimn_t>(dict_arg.size()));

        if (!dict_arg.empty()) {
            auto kv = *dict_arg.begin();
            check_and_set_dtype(options, kv.second);

            update_dtype_and_allocate(
                    result,
                    options,
                    options.shape[0],
                    options.shape[0]
            );

            key_type* key_ptr = result.keys();

            handle_dict(result, key_ptr, options, dict_arg);
        }
    } else if (py::isinstance<py::sequence>(object)) {
        std::vector<py::object> leaves;
        auto size_info = compute_size_and_type(options, leaves, object);

        update_dtype_and_allocate(
                result,
                options,
                size_info.num_values,
                size_info.num_keys
        );

        if (size_info.num_keys == 0) {
            // Scalar info only.
            dimn_t idx = 0;
            for (auto leaf : leaves) {
                auto leaf_seq = py::reinterpret_borrow<py::sequence>(leaf);
                for (auto obj : leaf_seq) {
                    auto val = result[idx++];
                    assign_py_object_to_scalar(val, obj);
                }
            }
        } else {
            auto* key_ptr = result.keys();
            RPY_DBG_ASSERT(size_info.num_values == 0 || key_ptr != nullptr);
            dimn_t idx = 0;
            for (auto leaf : leaves) {
                // Key-value
                if (py::isinstance<py::dict>(leaf)) {
                    handle_dict(result, key_ptr, options, leaf);
                } else {
                    for (auto obj :
                         py::reinterpret_borrow<py::sequence>(leaf)) {
                        auto val = result[idx++];
                        handle_sequence_tuple(val, key_ptr++, obj, options);
                    }
                }
            }
        }
    } else if (object.is_none()) {
    } else {
        RPY_THROW(
                std::invalid_argument,
                "could not parse argument to a valid scalar array type"
        );
    }

    return result;
}

scalars::Scalar
python::py_to_scalar(const scalars::ScalarType* type, py::handle object)
{
    RPY_DBG_ASSERT(type != nullptr);
    scalars::Scalar result(type);
    assign_py_object_to_scalar(result, object);
    return result;
}
