# ggml-cpu once per x86-64 level, picked at run time by src/cpu_dispatch.cpp
# (docs/BUILDING.md#cpu-variants).

# The x86-64 psABI levels v2 and v3, and v2 with AVX and F16C between them.
# Each tag is part of the renamed symbols. clang-cl takes -march as clang does.
set(MUSCRIPTOR_CPU_VARIANT_TAGS v2 f16c v3)

set(MUSCRIPTOR_CPU_FLAGS_v2 -march=x86-64-v2)
set(MUSCRIPTOR_CPU_FLAGS_f16c -march=x86-64-v2 -mavx -mf16c)
set(MUSCRIPTOR_CPU_FLAGS_v3 -march=x86-64-v3)

if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(MUSCRIPTOR_CPU_MANGLING msvc)
    # ggml's SSSE3 path, which no MSVC build of ggml reaches, passes block
    # pointers to _mm_prefetch, which clang-cl declares as taking const char*.
    set(MUSCRIPTOR_CPU_EXTRA_FLAGS $<$<COMPILE_LANGUAGE:C>:-Wno-error=incompatible-pointer-types>)
elseif(MSVC)
    message(FATAL_ERROR
        "MUSCRIPTOR_CPU_VARIANTS needs clang-cl, GCC or Clang. With cl.exe, configure with "
        "-DMUSCRIPTOR_CPU_VARIANTS=OFF.")
else()
    set(MUSCRIPTOR_CPU_MANGLING itanium)
endif()

# llvm-objcopy cannot rename bitcode.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(MUSCRIPTOR_CPU_NO_LTO -fno-lto)
endif()

# --- tools --------------------------------------------------------------

get_filename_component(MUSCRIPTOR_COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
find_program(MUSCRIPTOR_LLVM_OBJCOPY llvm-objcopy
    HINTS ${MUSCRIPTOR_COMPILER_DIR} /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin "$ENV{ProgramFiles}/LLVM/bin")

if(NOT MUSCRIPTOR_LLVM_OBJCOPY)
    message(FATAL_ERROR
        "MUSCRIPTOR_CPU_VARIANTS needs llvm-objcopy, llvm-nm, llvm-objdump and llvm-ar from an LLVM toolchain. "
        "Put them on the PATH or set MUSCRIPTOR_LLVM_OBJCOPY to the first, or configure with "
        "-DMUSCRIPTOR_CPU_VARIANTS=OFF.")
endif()

# The other three from the same toolchain where it has them.
get_filename_component(MUSCRIPTOR_LLVM_DIR ${MUSCRIPTOR_LLVM_OBJCOPY} DIRECTORY)

foreach(MUSCRIPTOR_LLVM_TOOL nm objdump ar)
    string(TOUPPER ${MUSCRIPTOR_LLVM_TOOL} MUSCRIPTOR_LLVM_TOOL_UPPER)
    find_program(MUSCRIPTOR_LLVM_${MUSCRIPTOR_LLVM_TOOL_UPPER} llvm-${MUSCRIPTOR_LLVM_TOOL}
        HINTS ${MUSCRIPTOR_LLVM_DIR} NO_DEFAULT_PATH)
    find_program(MUSCRIPTOR_LLVM_${MUSCRIPTOR_LLVM_TOOL_UPPER} llvm-${MUSCRIPTOR_LLVM_TOOL})

    if(NOT MUSCRIPTOR_LLVM_${MUSCRIPTOR_LLVM_TOOL_UPPER})
        message(FATAL_ERROR "MUSCRIPTOR_CPU_VARIANTS needs llvm-${MUSCRIPTOR_LLVM_TOOL} next to ${MUSCRIPTOR_LLVM_OBJCOPY}.")
    endif()
endforeach()

set(MUSCRIPTOR_CPU_TOOL ${CMAKE_CURRENT_LIST_DIR}/cpu_variant_tool.cmake)
set(MUSCRIPTOR_CPU_TOOL_ARGS
    -DLLVM_NM=${MUSCRIPTOR_LLVM_NM}
    -DLLVM_OBJCOPY=${MUSCRIPTOR_LLVM_OBJCOPY}
    -DLLVM_OBJDUMP=${MUSCRIPTOR_LLVM_OBJDUMP}
    -DLLVM_AR=${MUSCRIPTOR_LLVM_AR})

# --- ggml-cpu, as the variants expect it ---------------------------------
#
# Each check fails loudly when a ggml update changes what the variants copy.

get_target_property(MUSCRIPTOR_GGML_SRC_DIR ggml-cpu SOURCE_DIR)
get_directory_property(MUSCRIPTOR_GGML_ARCH DIRECTORY ${MUSCRIPTOR_GGML_SRC_DIR} DEFINITION GGML_SYSTEM_ARCH)

if(NOT MUSCRIPTOR_GGML_ARCH STREQUAL "x86" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "MUSCRIPTOR_CPU_VARIANTS is for x86-64 targets only; ggml detected '${MUSCRIPTOR_GGML_ARCH}'.")
endif()

get_target_property(MUSCRIPTOR_GGML_CPU_SOURCES ggml-cpu SOURCES)
set(MUSCRIPTOR_CPU_SOURCES)

foreach(MUSCRIPTOR_SOURCE IN LISTS MUSCRIPTOR_GGML_CPU_SOURCES)
    if(MUSCRIPTOR_SOURCE MATCHES "\\$<")
        message(FATAL_ERROR "ggml-cpu lists a generator-expression source, ${MUSCRIPTOR_SOURCE}, which the variants cannot copy.")
    endif()

    cmake_path(ABSOLUTE_PATH MUSCRIPTOR_SOURCE BASE_DIRECTORY ${MUSCRIPTOR_GGML_SRC_DIR})
    list(APPEND MUSCRIPTOR_CPU_SOURCES ${MUSCRIPTOR_SOURCE})
endforeach()

foreach(MUSCRIPTOR_SOURCE ggml-cpu/ggml-cpu.c ggml-cpu/ggml-cpu.cpp ggml-cpu/arch/x86/quants.c ggml-cpu/arch/x86/repack.cpp)
    if(NOT ${MUSCRIPTOR_GGML_SRC_DIR}/${MUSCRIPTOR_SOURCE} IN_LIST MUSCRIPTOR_CPU_SOURCES)
        message(FATAL_ERROR "ggml-cpu no longer lists ${MUSCRIPTOR_SOURCE} among its sources.")
    endif()
endforeach()

# A variant links what ggml-cpu links, which must be ggml-base alone.
get_target_property(MUSCRIPTOR_GGML_CPU_LIBS ggml-cpu LINK_LIBRARIES)

if(NOT MUSCRIPTOR_GGML_CPU_LIBS STREQUAL "ggml-base")
    message(FATAL_ERROR "ggml-cpu links ${MUSCRIPTOR_GGML_CPU_LIBS}; the variants link ggml-base only.")
endif()

# Directory-level definitions are not part of the target's own property.
get_directory_property(MUSCRIPTOR_GGML_DIR_DEFINITIONS DIRECTORY ${MUSCRIPTOR_GGML_SRC_DIR} COMPILE_DEFINITIONS)
get_target_property(MUSCRIPTOR_GGML_CPU_FEATURES ggml-cpu COMPILE_FEATURES)

# ggml sets its own language standards, which this directory's would replace.
set(MUSCRIPTOR_GGML_CPU_PROPERTIES)

foreach(MUSCRIPTOR_PROPERTY C_STANDARD C_STANDARD_REQUIRED C_EXTENSIONS CXX_STANDARD CXX_STANDARD_REQUIRED CXX_EXTENSIONS)
    get_target_property(MUSCRIPTOR_VALUE ggml-cpu ${MUSCRIPTOR_PROPERTY})

    if(NOT MUSCRIPTOR_VALUE STREQUAL "MUSCRIPTOR_VALUE-NOTFOUND")
        list(APPEND MUSCRIPTOR_GGML_CPU_PROPERTIES ${MUSCRIPTOR_PROPERTY} ${MUSCRIPTOR_VALUE})
    endif()
endforeach()

# --- variants -----------------------------------------------------------

get_property(MUSCRIPTOR_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

if(MUSCRIPTOR_MULTI_CONFIG)
    set(MUSCRIPTOR_CPU_VARIANT_DIR ${CMAKE_CURRENT_BINARY_DIR}/ggml-cpu-variants/$<CONFIG>)
else()
    set(MUSCRIPTOR_CPU_VARIANT_DIR ${CMAKE_CURRENT_BINARY_DIR}/ggml-cpu-variants)
endif()

set(MUSCRIPTOR_CPU_VARIANT_LIBRARIES)

foreach(MUSCRIPTOR_TAG ${MUSCRIPTOR_CPU_VARIANT_TAGS})
    set(MUSCRIPTOR_TARGET muscriptor_ggml_cpu_${MUSCRIPTOR_TAG})

    # Built only for the rename below.
    add_library(${MUSCRIPTOR_TARGET} STATIC EXCLUDE_FROM_ALL ${MUSCRIPTOR_CPU_SOURCES})

    # Through generator expressions, so flags a consumer adds to ggml-cpu after
    # add_subdirectory reach the variants too.
    target_include_directories(${MUSCRIPTOR_TARGET} PRIVATE $<TARGET_PROPERTY:ggml-cpu,INCLUDE_DIRECTORIES>)
    target_compile_definitions(${MUSCRIPTOR_TARGET}
        PRIVATE
            ${MUSCRIPTOR_GGML_DIR_DEFINITIONS}
            $<TARGET_PROPERTY:ggml-cpu,COMPILE_DEFINITIONS>)
    target_compile_options(${MUSCRIPTOR_TARGET}
        PRIVATE
            $<TARGET_PROPERTY:ggml-cpu,COMPILE_OPTIONS>
            ${MUSCRIPTOR_CPU_FLAGS_${MUSCRIPTOR_TAG}}
            ${MUSCRIPTOR_CPU_NO_LTO}
            ${MUSCRIPTOR_CPU_EXTRA_FLAGS})
    target_compile_features(${MUSCRIPTOR_TARGET} PRIVATE ${MUSCRIPTOR_GGML_CPU_FEATURES})
    target_link_libraries(${MUSCRIPTOR_TARGET} PRIVATE ggml-base)
    set_target_properties(${MUSCRIPTOR_TARGET} PROPERTIES
        ${MUSCRIPTOR_GGML_CPU_PROPERTIES}
        INTERPROCEDURAL_OPTIMIZATION OFF
        POSITION_INDEPENDENT_CODE ON)

    set(MUSCRIPTOR_RENAMED
        ${MUSCRIPTOR_CPU_VARIANT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cpu-${MUSCRIPTOR_TAG}${CMAKE_STATIC_LIBRARY_SUFFIX})

    add_custom_command(
        OUTPUT ${MUSCRIPTOR_RENAMED}
        COMMAND ${CMAKE_COMMAND} ${MUSCRIPTOR_CPU_TOOL_ARGS}
                -DACTION=rename
                -DINPUT=$<TARGET_FILE:${MUSCRIPTOR_TARGET}>
                -DOUTPUT=${MUSCRIPTOR_RENAMED}
                -DTAG=${MUSCRIPTOR_TAG}
                -DMANGLING=${MUSCRIPTOR_CPU_MANGLING}
                -P ${MUSCRIPTOR_CPU_TOOL}
        DEPENDS ${MUSCRIPTOR_TARGET} ${MUSCRIPTOR_CPU_TOOL}
        COMMENT "Renaming the symbols of the ${MUSCRIPTOR_TAG} ggml-cpu variant"
        VERBATIM)

    list(APPEND MUSCRIPTOR_CPU_VARIANT_LIBRARIES ${MUSCRIPTOR_RENAMED})
endforeach()

add_custom_target(muscriptor_ggml_cpu_variants DEPENDS ${MUSCRIPTOR_CPU_VARIANT_LIBRARIES})

# --- dispatcher ---------------------------------------------------------

add_library(muscriptor_ggml_cpu STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_dispatch.cpp)
add_dependencies(muscriptor_ggml_cpu muscriptor_ggml_cpu_variants)
target_link_libraries(muscriptor_ggml_cpu PUBLIC ${MUSCRIPTOR_CPU_VARIANT_LIBRARIES} ggml-base)
target_compile_features(muscriptor_ggml_cpu PRIVATE cxx_std_23)
set_target_properties(muscriptor_ggml_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(muscriptor_ggml_cpu PRIVATE /W4)
    # The UCRT marks std::getenv deprecated.
    target_compile_definitions(muscriptor_ggml_cpu PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(muscriptor_ggml_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# In ggml-cpu's place, for ggml (whose registry calls ggml_backend_cpu_reg) and
# for everything that links ggml.
foreach(MUSCRIPTOR_PROPERTY LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
    get_target_property(MUSCRIPTOR_GGML_LIBS ggml ${MUSCRIPTOR_PROPERTY})

    if(NOT "ggml-cpu" IN_LIST MUSCRIPTOR_GGML_LIBS)
        message(FATAL_ERROR "ggml's ${MUSCRIPTOR_PROPERTY} (${MUSCRIPTOR_GGML_LIBS}) no longer names ggml-cpu.")
    endif()

    list(TRANSFORM MUSCRIPTOR_GGML_LIBS REPLACE "^ggml-cpu$" muscriptor_ggml_cpu)
    set_property(TARGET ggml PROPERTY ${MUSCRIPTOR_PROPERTY} ${MUSCRIPTOR_GGML_LIBS})
endforeach()

set_property(TARGET ggml-cpu PROPERTY EXCLUDE_FROM_ALL TRUE)

# --- instruction check of everything else ---------------------------------
#
# Not part of the default build: it disassembles every archive, ggml-vulkan's
# included. `cmake --build <dir> --target muscriptor_check_cpu_baseline`.

set(MUSCRIPTOR_BASELINE_LIBRARIES)

foreach(MUSCRIPTOR_TARGET muscriptor_ggml muscriptor_ggml_cpu ggml ggml-base ggml-metal ggml-vulkan pffft)
    if(TARGET ${MUSCRIPTOR_TARGET})
        list(APPEND MUSCRIPTOR_BASELINE_LIBRARIES $<TARGET_FILE:${MUSCRIPTOR_TARGET}>)
    endif()
endforeach()

list(JOIN MUSCRIPTOR_BASELINE_LIBRARIES "|" MUSCRIPTOR_BASELINE_INPUTS)

add_custom_target(muscriptor_check_cpu_baseline
    COMMAND ${CMAKE_COMMAND} ${MUSCRIPTOR_CPU_TOOL_ARGS}
            -DACTION=check
            -DLEVEL=v2
            "-DINPUTS=${MUSCRIPTOR_BASELINE_INPUTS}"
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/ggml-cpu-variants
            -P ${MUSCRIPTOR_CPU_TOOL}
    DEPENDS muscriptor_ggml muscriptor_ggml_cpu
    COMMENT "Checking the code outside the ggml-cpu variants for instructions above x86-64-v2"
    VERBATIM)
