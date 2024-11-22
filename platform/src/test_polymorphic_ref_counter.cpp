//
// Created by sammorley on 22/11/24.
//


#include <gtest/gtest.h>

#include <roughpy/core/macros.h>
#include <roughpy/core/smart_ptr.h>

#include "roughpy/platform/reference_counting.h"

using namespace rpy;

namespace {

class Dummy : public mem::PolymorphicRefCounted
{
    mutable int count = 0;

protected:
    void inc_ref() const noexcept override
    {
        ++count;
    }
    RPY_NO_DISCARD bool dec_ref() const noexcept override
    {
        return --count == 0;
    }

public:
    RPY_NO_DISCARD intptr_t ref_count() const noexcept override
    {
        return count;
    }
};

}


TEST(PolymorphicRefCounter, TestRefCountingWithOverridedMethods)
{
    Rc<Dummy> dummy(new Dummy);
    EXPECT_EQ(1, dummy->ref_count());

    {
        RPY_MAYBE_UNUSED Rc<Dummy> new_dummy(dummy);// NOLINT(*-unnecessary-copy-initialization)
        EXPECT_EQ(2, dummy->ref_count());
    }

    EXPECT_EQ(1, dummy->ref_count());
}



namespace {

class DummyWithMiddle : public mem::RefCountedMiddle<>
 {};

}


TEST(PolymorphicRefCounter, TestRefCountedMiddle)
{
    Rc<DummyWithMiddle> dummy(new DummyWithMiddle);
    EXPECT_EQ(1, dummy->ref_count());

    {
        RPY_MAYBE_UNUSED Rc<DummyWithMiddle> new_dummy(dummy); // NOLINT(*-unnecessary-copy-initialization)
        EXPECT_EQ(2, dummy->ref_count());
    }

    EXPECT_EQ(1, dummy->ref_count());

}