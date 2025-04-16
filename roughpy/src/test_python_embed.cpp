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

//! Embed all of roughpy in this app rather than import from external .so
PYBIND11_EMBEDDED_MODULE(_roughpy_embed, m) {
    init_roughpy_module(m);
}

// Global scoped interpreter used in all tests.
//
// Note that a scoped_interpreter per-fixture would be the preference for these
// tests, but importing roughpy more than once during the application's
// lifespan results in a segfault due to memory being freed twice.
//
// The fixture class works around this by having a single scoped interpreter
// for all tests and clearing globals between each, whilst depending on the
// cached roughpy import. This is not ideal because the module's state persists
// between tests.
//
static std::unique_ptr<py::scoped_interpreter> g_fixture_interpreter;

//! Test fixture for common python state
class PythonEmbedFixture : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        // Ensure only one interpreter is used for all tests
        assert(!g_fixture_interpreter.get());
        g_fixture_interpreter = std::make_unique<py::scoped_interpreter>();
    }

    void TearDown() override
    {
        // Delete globals between runs - see g_fixture_interpreter for details
        py::exec(R"(
            for v in dir():
                exec('del '+ v)
                del v
        )");
    }
};

} // namespace

TEST_F(PythonEmbedFixture, SelfTest)
{
    // Confirm the g_fixture_interpreter workaround causes no seg fault.
    // This works in tandem with at least one other PythonEmbedFixture TEST_F
    // to confirm that double-free error does not happen.
    py::exec(R"(
        import _roughpy_embed as rp
        import pytest
    )");
}

TEST_F(PythonEmbedFixture, CreateFreeTensor)
{
    py::exec(R"(
        import _roughpy_embed as rp
        import pytest

        context = rp.get_context(width=2, depth=3, coeffs=rp.DPReal)

        # Example construction from scalar terms
        a = rp.FreeTensor(
            [i for i in range(context.tensor_size())],
            ctx=context
        )

        # Example construction from polynomial (invalid with coeffs DPReal)
        with pytest.raises(ValueError):
            rp.FreeTensor(
                [1 * rp.Monomial(f"b{i}") for i in range(context.tensor_size())],
                ctx=context
            )
    )");

    // Example code querying C++ interface of Python object
    auto* a_ptr = py::globals()["a"].cast<rpy::algebra::FreeTensor*>();
    ASSERT_EQ(a_ptr->width(), 2);
    ASSERT_EQ(a_ptr->depth(), 3);
    ASSERT_EQ(a_ptr->coeff_type()->name(), "DPReal");
}
