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

#include <roughpy/algebra/free_tensor.h>

#include "roughpy_module.h"

namespace {

namespace py = pybind11;
using namespace pybind11::literals; // use _a literal
using namespace rpy::python;

//! Embed all of roughpy rather than import from external .so
PYBIND11_EMBEDDED_MODULE(_roughpy_embed, m) {
    init_roughpy_module(m);
}

//! Test fixture for common python state
class PythonEmbedFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // FIXME see notes in TearDown on why singleton is used
        if (!s_interpreter_singleton) {
            s_interpreter_singleton = std::make_unique<py::scoped_interpreter>();
        }
    }

    void TearDown() override
    {
        // FIXME presently it's not possible to re-import roughpy for each test
        // because it results in memory being freed twice and seg faulting. This
        // workaround reuses the same singleton scoped interpreter around all
        // tests, then clears globals so each test can run anew. This is far
        // from ideal because we are still re-using state between tests.
        py::exec(R"(
            for v in dir():
                exec('del '+ v)
                del v
        )");
    }

private:
    static std::unique_ptr<py::scoped_interpreter> s_interpreter_singleton;
};

std::unique_ptr<py::scoped_interpreter> PythonEmbedFixture::s_interpreter_singleton;

} // namespace


TEST_F(PythonEmbedFixture, CreateFreeTensor)
{
    py::exec(R"(
        import _roughpy_embed as rp
        context = rp.get_context(width=2, depth=3, coeffs=rp.DPReal)
        # Test construction from scalar terms
        a = rp.FreeTensor(
            [i for i in range(context.tensor_size())],
            ctx=context
        )

        # Test construction from polynomial (invalid)
        b = rp.FreeTensor(
            [1 * rp.Monomial(f"b{i}") for i in range(context.tensor_size())],
            ctx=context
        )
    )");

    auto x = py::globals()["b"];
    auto* a_ptr = x.cast<rpy::algebra::FreeTensor*>();
    std::ostringstream a_ptr_str;
    a_ptr_str << *a_ptr;

    // FIXME remove cout when test is complete
    std::cout << a_ptr_str.str() << std::endl;
}


// FIXME Experimental code for running multiple tests when re-importing roughpy.
TEST_F(PythonEmbedFixture, WillItCrash)
{
    auto rp = py::module::import("_roughpy_embed");
    py::object context = rp.attr("get_context")(
        "width"_a = 2,
        "depth"_a = 3,
        "coeffs"_a = rp.attr("RationalPoly")
    );
}
