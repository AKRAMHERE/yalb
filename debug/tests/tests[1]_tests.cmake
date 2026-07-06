add_test([=[HelloTest.BasicAssertions]=]  /home/thillis/hpc/yalb/debug/tests/tests [==[--gtest_filter=HelloTest.BasicAssertions]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[HelloTest.BasicAssertions]=]  PROPERTIES WORKING_DIRECTORY /home/thillis/hpc/yalb/debug/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  tests_TESTS HelloTest.BasicAssertions)
