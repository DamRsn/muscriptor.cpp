add_executable(muscriptor_bench bench.cpp)
target_link_libraries(muscriptor_bench PRIVATE muscriptor_ggml)
# ../src for format.hpp, which the benchmark shares rather than restating. It is
# private to the library and must not leak into cpp/include -- same reason the
# tests reach in there.
target_include_directories(muscriptor_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if(MSVC)
    target_compile_options(muscriptor_bench PRIVATE /W4)
else()
    target_compile_options(muscriptor_bench PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Same fixture root the tests use, and the same MUSCRIPTOR_TESTDATA override,
# so a benchmark run and a test run are looking at one set of weights. The
# defaults for --size, --device and --weight-dtype are the test suite's too
# (../CMakeLists.txt), except that `all` falls back to `medium` / `cpu` here: a
# benchmark measures one configuration at a time.
foreach(MUSCRIPTOR_BENCH_PAIR "SIZE;medium" "DEVICE;cpu")
    list(GET MUSCRIPTOR_BENCH_PAIR 0 MUSCRIPTOR_BENCH_KEY)
    list(GET MUSCRIPTOR_BENCH_PAIR 1 MUSCRIPTOR_BENCH_FALLBACK)
    set(MUSCRIPTOR_BENCH_VALUE ${MUSCRIPTOR_TEST_${MUSCRIPTOR_BENCH_KEY}})

    if(MUSCRIPTOR_BENCH_VALUE STREQUAL "all")
        set(MUSCRIPTOR_BENCH_VALUE ${MUSCRIPTOR_BENCH_FALLBACK})
    endif()

    set(MUSCRIPTOR_BENCH_${MUSCRIPTOR_BENCH_KEY} ${MUSCRIPTOR_BENCH_VALUE})
endforeach()

target_compile_definitions(muscriptor_bench
    PRIVATE
        MUSCRIPTOR_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../testdata"
        MUSCRIPTOR_TEST_SIZE="${MUSCRIPTOR_BENCH_SIZE}"
        MUSCRIPTOR_TEST_DEVICE="${MUSCRIPTOR_BENCH_DEVICE}"
        MUSCRIPTOR_TEST_WEIGHT_DTYPE="${MUSCRIPTOR_TEST_WEIGHT_DTYPE}"
)
