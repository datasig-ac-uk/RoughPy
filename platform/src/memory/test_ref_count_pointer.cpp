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

    using Obj = CustomImplementationRefCountableObject;
    using Ptr = Rc<Obj>;



};

TEST_F(RcRefCountingTest, InitialRefCountIsOne)
{
    Obj object;
    Ptr p(&object);

    EXPECT_EQ(p->get_ref_count(), 1);
}

TEST_F(RcRefCountingTest, IncreaseRefCount)
{
    Obj object;
    Ptr p(&object);

    Ptr p2 = p;// Incrementing the reference count
    EXPECT_EQ(p->get_ref_count(), 2);
    EXPECT_EQ(p2->get_ref_count(), 2);
}

TEST_F(RcRefCountingTest, DecreaseRefCount)
{
    Obj object;
    Ptr p(&object);

    EXPECT_EQ(p->get_ref_count(), 1);
    {
        Ptr p2(p);// Incrementing the reference count
        EXPECT_EQ(p2->get_ref_count(), 2);
    }// p2 goes out of scope here, reference count decrements
    EXPECT_EQ(p->get_ref_count(), 1);
}

TEST_F(RcRefCountingTest, MultipleIncrementsAndDecrements)
{
    Obj object;
    Ptr p(&object);

    {
        Ptr p2(p);
        Ptr p3(p);

        EXPECT_EQ(p->get_ref_count(), 3);// ref count should be 3 now
        EXPECT_EQ(p2->get_ref_count(), 3);
        EXPECT_EQ(p3->get_ref_count(), 3);
    }// obj2 and obj3 go out of scope here, reference count decrements

    EXPECT_EQ(p->get_ref_count(), 1);
}


TEST_F(RcRefCountingTest, ResetdecreasesRefCount)
{
    Obj object;
    Ptr p(&object);

    EXPECT_EQ(p->get_ref_count(), 1);
    p.reset();
    EXPECT_EQ(object.get_ref_count(), 0);
    EXPECT_EQ(p.get(), nullptr);
}

TEST_F(RcRefCountingTest, ReleaseDoesNotReducRefCount)
{
    Obj object;
    Ptr p(&object);

    EXPECT_EQ(p->get_ref_count(), 1);
    auto* ptr = p.release();
    EXPECT_EQ(object.get_ref_count(), 1);
    EXPECT_EQ(p.get(), nullptr);
    EXPECT_EQ(ptr, &object);
}