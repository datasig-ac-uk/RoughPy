// Copyright (c) 2025 RoughPy Developers. All rights reserved.
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

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "roughpy/algebra/free_tensor.h"

#include "algebra/algebra.h"
#include "args/convert_timestamp.h"
#include "intervals/intervals.h"
#include "scalars/scalars.h"
#include "streams/streams.h"

#include "args/numpy.h"

namespace {

namespace py = pybind11;
using namespace pybind11::literals; // use _a literal
using namespace rpy::python;

} // namespace

PYBIND11_EMBEDDED_MODULE(_embed_roughpy, m)
{
    init_datetime(m);
    init_scalars(m);
    init_intervals(m);
    init_algebra(m);
    init_streams(m);

    import_numpy();
}


TEST(test_RoughPy_PyModule, CreateFreeTensor)
{
    py::scoped_interpreter guard{};
    py::module_ rp = py::module_::import("_embed_roughpy");
    py::object context = rp.attr("get_context")(
        "width"_a = 2,
        "depth"_a = 3,
        "coeffs"_a = rp.attr("RationalPoly")
    );

    int N = context.attr("tensor_size")().cast<int>();
    py::list ones;
    for (int i = 0; i < N; ++i) {
        ones.append(1);
    }

    py::object a = rp.attr("FreeTensor")(
        ones,
        "ctx"_a = context
    );
    std::string a_str = py::str(a).cast<std::string>();

    // Initial test: check that C++ cast is correct and values match
    auto* a_ptr = a.cast<rpy::algebra::FreeTensor*>();
    std::ostringstream a_ptr_str;
    a_ptr_str << *a_ptr;
    ASSERT_EQ(a_ptr_str.str(), a_str);
}

// FIXME including this will segfault `double free or corruption (out)`
#if 0
TEST(test_RoughPy_PyModule, WillItCrash)
{
    py::scoped_interpreter guard{};
    py::module_ rp = py::module_::import("_embed_roughpy");
}
#endif