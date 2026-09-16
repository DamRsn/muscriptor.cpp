include(FetchContent)

# Pinned by full commit rather than by tag, for the same reason ggml is: a tag
# is a movable reference, and a test dependency that changes underneath a clean
# clone changes what the ladder means.
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        b3fb4b9feafcd8d91c5cb510a4775143fdbef02f   # v3.11.0
)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        55f93686c01528224f448c19128836e7df245f72   # v3.12.0
)
FetchContent_MakeAvailable(Catch2 nlohmann_json)

add_executable(muscriptor_tests
    main.cpp
    run_config.cpp
    reference.cpp
    vectors.cpp
    inference.cpp
    test_vocabulary.cpp
    test_instrument_groups.cpp
    test_tracker.cpp
    test_note_assembly.cpp
    test_tokens_to_notes.cpp
    test_instrument_conditioning.cpp
    test_prelude.cpp
    test_transcriber.cpp
    test_gguf.cpp
    test_stft.cpp
    test_conditioning.cpp
    test_prefill.cpp
    test_decode.cpp
    test_generate.cpp
    test_backend.cpp
)
target_link_libraries(muscriptor_tests
    PRIVATE muscriptor_ggml Catch2::Catch2 nlohmann_json::nlohmann_json
)
# ../src because the bisection ladder drives internals -- GgufFile to read the
# reference dumps, Trace to capture intermediates -- that are not part of the
# public surface and must not leak into cpp/include.
target_include_directories(muscriptor_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
# Same MSVC carve-out as the library target: clang-cl reads the MSVC spellings
# first, and there /Wall means -Weverything.
if(MSVC)
    target_compile_options(muscriptor_tests PRIVATE /W4)
else()
    target_compile_options(muscriptor_tests PRIVATE -Wall -Wextra)
endif()

# Where the generated fixtures live. Overridable at run time with
# MUSCRIPTOR_TESTDATA so a build tree can be pointed at another dump. The
# other three are the compiled-in defaults for --size, --device and
# --weight-dtype, declared in ../CMakeLists.txt because the bench shares them.
target_compile_definitions(muscriptor_tests
    PRIVATE
        MUSCRIPTOR_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../testdata"
        MUSCRIPTOR_TEST_SIZE="${MUSCRIPTOR_TEST_SIZE}"
        MUSCRIPTOR_TEST_DEVICE="${MUSCRIPTOR_TEST_DEVICE}"
        MUSCRIPTOR_TEST_WEIGHT_DTYPE="${MUSCRIPTOR_TEST_WEIGHT_DTYPE}"
)

# test_backend.cpp asserts a different contract depending on whether a GPU
# backend exists at all: with one, --device gpu must actually reach it;
# without one, the same request must degrade to CPU.
if(MUSCRIPTOR_METAL)
    target_compile_definitions(muscriptor_tests PRIVATE MUSCRIPTOR_HAS_METAL)
endif()

if(MUSCRIPTOR_VULKAN)
    target_compile_definitions(muscriptor_tests PRIVATE MUSCRIPTOR_HAS_VULKAN)
endif()
