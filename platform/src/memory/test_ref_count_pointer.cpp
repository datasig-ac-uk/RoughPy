//
// Created by sam on 25/10/24.
//

#include <gtest/gtest.h>
#include <roughpy/platform/memory.h>

using namespace rpy;

namespace {

struct DerivedRefCountableObject : public RcBase {

    dimn_t get_ref_count() const noexcept { return this->ref_count(); }
};

}// namespace

TEST(RcBaseDerivedCountingTests, InitialRefCountIsOne)
{
    Rc<DerivedRefCountableObject> obj = make_rc<DerivedRefCountableObject>();
    EXPECT_EQ(obj->get_ref_count(), 1);
}

TEST(RcBaseDerivedCountingTests, IncreaseAndDecreaseRefCount)
{
    Rc<DerivedRefCountableObject> obj1 = make_rc<DerivedRefCountableObject>();
    EXPECT_EQ(obj1->get_ref_count(), 1);

    {
        Rc<DerivedRefCountableObject> obj2 = obj1;
        EXPECT_EQ(obj1->get_ref_count(), 2);
        EXPECT_EQ(obj2->get_ref_count(), 2);
    }

    EXPECT_EQ(obj1->get_ref_count(), 1);
}

class RcRefCountingTest : public ::testing::Test
{
protected:
    struct CustomImplementationRefCountableObject {
        mutable dimn_t m_rc = 0;

        dimn_t get_ref_count() const noexcept { return this->ref_count(); }

        dimn_t ref_count() const noexcept { return m_rc; }

        void inc_ref() const noexcept { ++m_rc; }
        bool dec_ref() const noexcept { return (--m_rc) == 0; }
    };

    CustomImplementationRefCountableObject m_obj;
    Rc<CustomImplementationRefCountableObject> p_obj = nullptr;

    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
        p_obj = Rc<CustomImplementationRefCountableObject>(&m_obj);
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right before
        // the destructor).
    }
};

TEST_F(RcRefCountingTest, InitialRefCountIsOne)
{
    EXPECT_EQ(p_obj->get_ref_count(), 1);
}

TEST_F(RcRefCountingTest, IncreaseRefCount)
{
    Rc<CustomImplementationRefCountableObject> obj2
            = p_obj;// Incrementing the reference count
    EXPECT_EQ(p_obj->get_ref_count(), 2);
    EXPECT_EQ(obj2->get_ref_count(), 2);
}

TEST_F(RcRefCountingTest, DecreaseRefCount)
{
    {
        Rc<CustomImplementationRefCountableObject> obj2
                = p_obj;// Incrementing the reference count
        EXPECT_EQ(obj2->get_ref_count(), 2);
    }// obj2 goes out of scope here, reference count decrements
    EXPECT_EQ(p_obj->get_ref_count(), 1);
}

TEST_F(RcRefCountingTest, MultipleIncrementsAndDecrements)
{
    {
        Rc<CustomImplementationRefCountableObject> obj2
                = p_obj;// Incrementing the reference count
        Rc<CustomImplementationRefCountableObject> obj3
                = p_obj;                     // Incrementing the reference count
        EXPECT_EQ(p_obj->get_ref_count(), 3);// ref count should be 3 now
        EXPECT_EQ(obj2->get_ref_count(), 3);
        EXPECT_EQ(obj3->get_ref_count(), 3);
    }// obj2 and obj3 go out of scope here, reference count decrements
    EXPECT_EQ(p_obj->get_ref_count(), 1);
}

TEST_F(RcRefCountingTest, CustomIncRefWorks)
{
    p_obj->inc_ref();// Manually incrementing the reference count
    EXPECT_EQ(p_obj->get_ref_count(), 2);
}


TEST_F(RcRefCountingTest, ResetdecreasesRefCount)
{
    EXPECT_EQ(p_obj->get_ref_count(), 1);
    p_obj.reset();
    EXPECT_EQ(m_obj.get_ref_count(), 0);
    EXPECT_EQ(p_obj.get(), nullptr);
}

TEST_F(RcRefCountingTest, ReleaseDoesNotReducRefCount)
{
    EXPECT_EQ(p_obj->get_ref_count(), 1);
    auto* ptr = p_obj.release();
    EXPECT_EQ(m_obj.get_ref_count(), 1);
    EXPECT_EQ(p_obj.get(), nullptr);
    EXPECT_EQ(ptr, &m_obj);
}