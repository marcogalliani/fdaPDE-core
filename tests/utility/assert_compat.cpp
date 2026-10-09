// fdapde_assert accepts the legacy form fdapde_assert(condition), which aborts, and upstream's typed form
// fdapde_assert(condition, exception_type, message), which throws
#include <fdaPDE/utility.h>
#include <gtest/gtest.h>

#include <stdexcept>

TEST(assert_compat, typed_form_throws_the_requested_exception) {
    EXPECT_NO_THROW(fdapde_assert(1 + 1 == 2, std::invalid_argument, "never thrown"));
    EXPECT_THROW(fdapde_assert(1 + 1 == 3, std::invalid_argument, "thrown"), std::invalid_argument);
    EXPECT_THROW(fdapde_assert(false, std::out_of_range, "thrown"), std::out_of_range);
}

TEST(assert_compat, legacy_form_passes_and_aborts) {
    int calls = 0;
    auto check = [&](int x) {
        ++calls;
        fdapde_assert(x > 0);
    };
    check(1);
    EXPECT_EQ(calls, 1);
    EXPECT_DEATH(check(-1), "Assertion: 'x > 0' failed");
}

TEST(assert_compat, strong_assert_always_checks) {
    EXPECT_THROW(fdapde_strong_assert(false, std::logic_error, "always"), std::logic_error);
}
