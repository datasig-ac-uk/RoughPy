// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 23/10/23.
//

#include <roughpy/device/kernel_arg.h>

#include <stdexcept>

using namespace rpy;
using namespace rpy::devices;

KernelArgument::~KernelArgument() = default;

string KernelArgument::name() const noexcept { return std::string(); }
string KernelArgument::type_string() const noexcept { return "void"; }
void KernelArgument::set(Buffer& data) {}
void KernelArgument::set(const Buffer& data) {}
void KernelArgument::set(void* data, const TypeInfo& info) {}
void KernelArgument::set(const void* data, const TypeInfo& info) {}

void KernelArgument::set(half data) {
    this->set(&data, {TypeCode::Float, sizeof(half), 1});
}
void KernelArgument::set(bfloat16 data) {
    this->set(&data, {TypeCode::BFloat, sizeof(bfloat16), 1});
}
void KernelArgument::set(float data) {
    this->set(&data, {TypeCode::Float, sizeof(float), 1});
}
void KernelArgument::set(double data) {
    this->set(&data, {TypeCode::Float, sizeof(double), 1});
}
void KernelArgument::set(const rational_scalar_type& RPY_UNUSED_VAR data) {
    RPY_THROW(std::invalid_argument, "rational scalar types are not supported"
                                     " by this device");
}
void KernelArgument::set(const rational_poly_scalar& RPY_UNUSED_VAR data) {
    RPY_THROW(std::invalid_argument, "polynomial scalar types are not supported"
                                     " by this device");
}
