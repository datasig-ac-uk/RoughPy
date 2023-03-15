//
// Created by sam on 13/03/23.
//


#include "ScalarTests.h"
#include "scalar_pointer.h"
#include "scalar.h"

#include <vector>

using ScalarTypeTests = rpy::scalars::testing::ScalarTests;
using namespace rpy::scalars;


class RAIIAlloc {
    ScalarPointer m_ptr;
    std::size_t m_count;

public:

    RAIIAlloc(const ScalarType* type, std::size_t count)
        : m_ptr(type->allocate(count)), m_count(count)
    {}

    ~RAIIAlloc() {
        if (m_ptr.type() != nullptr) {
            m_ptr.type()->free(m_ptr, m_count);
        }
    }

    constexpr operator ScalarPointer() const noexcept { return m_ptr; }


};


TEST_F(ScalarTypeTests, BasicInfoFloat) {
    auto info = ftype->info();

    ASSERT_EQ(info.n_bytes, sizeof(float));
    ASSERT_EQ(info.alignment, alignof(float));
    ASSERT_EQ(info.name, "float");
    ASSERT_EQ(info.id, "f32");

    ASSERT_EQ(info.device.device_type, ScalarDeviceType::CPU);
    ASSERT_EQ(info.device.device_id, 0);
    ASSERT_EQ(info.basic_info.bits, sizeof(float)*8);
    ASSERT_EQ(info.basic_info.code, static_cast<std::uint8_t>(ScalarTypeCode::Float));
    ASSERT_EQ(info.basic_info.lanes, 1);

}

TEST_F(ScalarTypeTests, BasicInfoDouble) {
    auto info = dtype->info();

    ASSERT_EQ(info.n_bytes, sizeof(double));
    ASSERT_EQ(info.alignment, alignof(double));
    ASSERT_EQ(info.name, "double");
    ASSERT_EQ(info.id, "f64");

    ASSERT_EQ(info.device.device_type, ScalarDeviceType::CPU);
    ASSERT_EQ(info.device.device_id, 0);
    ASSERT_EQ(info.basic_info.bits, sizeof(double)*8);
    ASSERT_EQ(info.basic_info.code, static_cast<std::uint8_t>(ScalarTypeCode::Float));
    ASSERT_EQ(info.basic_info.lanes, 1);
}

TEST_F(ScalarTypeTests, RationalTypeEqualForFloatingScalars) {
    ASSERT_EQ(dtype->rational_type(), dtype);
    ASSERT_EQ(ftype->rational_type(), ftype);
}


TEST_F(ScalarTypeTests, AllocateGivesNonNull) {
    RAIIAlloc alloced(dtype, 1);
    auto ptr = ScalarPointer(alloced);

    EXPECT_NE(ptr.cptr(), nullptr);
}


TEST_F(ScalarTypeTests, OneGivesCorrectValue) {
    auto one = dtype->one();
    auto ptr = one.to_pointer();

    ASSERT_NE(ptr.cptr(), nullptr);
    ASSERT_EQ(*ptr.raw_cast<const double*>(), 1.0);
}

TEST_F(ScalarTypeTests, ZeroGivesCorrectValue) {
    auto zero = dtype->zero();
    auto ptr = zero.to_pointer();

    ASSERT_NE(ptr.cptr(), nullptr);
    ASSERT_EQ(*ptr.raw_cast<const double*>(), 0.0);
}

TEST_F(ScalarTypeTests, MoneGivesCorrectValue) {
    auto mone = dtype->mone();
    auto ptr = mone.to_pointer();

    ASSERT_NE(ptr.cptr(), nullptr);
    ASSERT_EQ(*ptr.raw_cast<const double*>(), -1.0);
}


TEST_F(ScalarTypeTests, ConvertCopyFromNonScalarType) {

    std::vector<std::int32_t> ints {1, 2, 3, 4, 5, 6};

    RAIIAlloc alloced(dtype, ints.size());
    ScalarPointer ptr(alloced);

    dtype->convert_copy(ptr, ints.data(), ints.size(), type_id_of<std::int32_t>());

    auto raw_ptr = ptr.raw_cast<const double*>();
    for (std::size_t i=0; i<ints.size(); ++i) {
        ASSERT_EQ(raw_ptr[i], static_cast<double>(ints[i]));
    }

}

TEST_F(ScalarTypeTests, CopyConvertFromScalarType) {
    std::vector<float> floats {1.01, 2.02, 3.03, 4.04, 5.05, 6.06};

    RAIIAlloc alloced(dtype, floats.size());
    ScalarPointer ptr(alloced);

    dtype->convert_copy(ptr, {ftype, floats.data()}, floats.size());

    auto raw_ptr = ptr.raw_cast<const double *>();
    for (std::size_t i = 0; i < floats.size(); ++i) {
        ASSERT_EQ(raw_ptr[i], static_cast<double>(floats[i]));
    }
}


TEST_F(ScalarTypeTests, ToScalarFromFloat) {
    float val = 1.000001; // doesn't have an exact float representation
    auto dble = ftype->to_scalar_t({ftype, &val});

    ASSERT_FLOAT_EQ(dble, val);
}


TEST_F(ScalarTypeTests, AssignNumAndDenom) {
    double val = 0.0;
    dtype->assign({dtype, &val}, 1LL, 256LL);

    ASSERT_DOUBLE_EQ(val, 1.0 / 256.0);
}
