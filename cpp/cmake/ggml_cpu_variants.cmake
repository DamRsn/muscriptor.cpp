#[[
ggml-cpu once per x86-64 instruction-set level, in one static binary
(docs/BUILDING.md#cpu-variants).

Each variant compiles ggml-cpu's sources with ggml-cpu's own flags, read at
generate time, plus its instruction set. cpu_variant_tool.cmake then renames
every symbol a variant defines to carry its tag, inline and template copies
included, so the linker can neither mix variants nor hand variant code to the
rest of the program. src/cpu_dispatch.cpp, compiled for the baseline, takes
ggml-cpu's place in ggml's link interface and forwards to the best variant the
CPU runs. ggml's own ggml-cpu is left out of the build.

Included from CMakeLists.txt after the library targets, with every GGML_<ISA>
option off.
]]

set(MUSCRIPTOR_CPU_VARIANT_TAGS sse42 ivybridge haswell)

if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # clang-cl ignores /arch:SSE4.2 on x64, with a warning only. /arch:AVX2
    # brings FMA, F16C, BMI1, BMI2, LZCNT and MOVBE.
    set(MUSCRIPTOR_CPU_FLAGS_sse42 -msse4.2)
    set(MUSCRIPTOR_CPU_FLAGS_ivybridge /arch:AVX -mf16c)
    set(MUSCRIPTOR_CPU_FLAGS_haswell /arch:AVX2)
    set(MUSCRIPTOR_CPU_MANGLING msvc)
elseif(MSVC)
    message(FATAL_ERROR
        "MUSCRIPTOR_CPU_VARIANTS needs clang-cl, GCC or Clang. With cl.exe, configure with "
        "-DMUSCRIPTOR_CPU_VARIANTS=OFF.")
else()
    # ggml's own spellings for the same levels.
    set(MUSCRIPTOR_CPU_FLAGS_sse42 -msse4.2)
    set(MUSCRIPTOR_CPU_FLAGS_ivybridge -msse4.2 -mavx -mf16c)
    set(MUSCRIPTOR_CPU_FLAGS_haswell -msse4.2 -mavx -mavx2 -mf16c -mfma -mbmi2)
    set(MUSCRIPTOR_CPU_MANGLING itanium)
endif()

# The definitions ggml adds next to its flags, for these levels.
set(MUSCRIPTOR_CPU_DEFINITIONS_sse42 GGML_SSE42)
set(MUSCRIPTOR_CPU_DEFINITIONS_ivybridge GGML_SSE42 GGML_AVX GGML_F16C)
set(MUSCRIPTOR_CPU_DEFINITIONS_haswell GGML_SSE42 GGML_AVX GGML_F16C GGML_FMA GGML_AVX2 GGML_BMI2)

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

if(NOT MUSCRIPTOR_GGML_ARCH STREQUAL "x86")
    message(FATAL_ERROR "ggml detected '${MUSCRIPTOR_GGML_ARCH}' rather than x86, so its x86 sources are not in ggml-cpu.")
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

# --- no instruction-set flags outside the variants -----------------------
#
# A flag anywhere else would compile code that the lowest CPU might not run.
# Global flags reach the variants and the dispatcher too.

set(MUSCRIPTOR_ISA_FLAG_REGEX
    "(^|[ ;:\"])(/arch:|-march=|-mcpu=|-m(sse|ssse|avx|f16c|fma|bmi|lzcnt|popcnt|movbe|amx|xop|aes|pclmul|sha|gfni|vaes|vpclmul))")
set(MUSCRIPTOR_ISA_DEFINITION_REGEX
    "(^|[ ;:\"])(GGML_(SSE42|AVX|F16C|FMA|BMI2|AMX)|__(SSE|SSSE|AVX|F16C|FMA|BMI|LZCNT|POPCNT|MOVBE|AMX)[A-Z0-9_]*__)")

function(muscriptor_check_no_isa inWhat inFlags inDefinitions)
    set(found "")

    if("${inFlags}" MATCHES "${MUSCRIPTOR_ISA_FLAG_REGEX}")
        set(found ${CMAKE_MATCH_2})
    elseif("${inDefinitions}" MATCHES "${MUSCRIPTOR_ISA_DEFINITION_REGEX}")
        set(found ${CMAKE_MATCH_2})
    endif()

    if(found)
        message(FATAL_ERROR
            "${inWhat} sets an instruction set (${found}...). With MUSCRIPTOR_CPU_VARIANTS only the ggml-cpu "
            "variants may, or code built with it can fault on a CPU the dispatcher sends to a lower variant.")
    endif()
endfunction()

foreach(MUSCRIPTOR_LANG C CXX)
    set(MUSCRIPTOR_GLOBAL_FLAGS "${CMAKE_${MUSCRIPTOR_LANG}_FLAGS}")

    foreach(MUSCRIPTOR_CONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL ${CMAKE_BUILD_TYPE} ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${MUSCRIPTOR_CONFIG} MUSCRIPTOR_CONFIG)
        string(APPEND MUSCRIPTOR_GLOBAL_FLAGS " ${CMAKE_${MUSCRIPTOR_LANG}_FLAGS_${MUSCRIPTOR_CONFIG}}")
    endforeach()

    muscriptor_check_no_isa("CMAKE_${MUSCRIPTOR_LANG}_FLAGS" "${MUSCRIPTOR_GLOBAL_FLAGS}" "${MUSCRIPTOR_GLOBAL_FLAGS}")
endforeach()

get_directory_property(MUSCRIPTOR_GGML_DIR_OPTIONS DIRECTORY ${MUSCRIPTOR_GGML_SRC_DIR} COMPILE_OPTIONS)
muscriptor_check_no_isa("ggml's source directory" "${MUSCRIPTOR_GGML_DIR_OPTIONS}" "${MUSCRIPTOR_GGML_DIR_DEFINITIONS}")

foreach(MUSCRIPTOR_TARGET ggml-cpu ggml-base ggml ggml-metal ggml-vulkan muscriptor_ggml pffft)
    if(TARGET ${MUSCRIPTOR_TARGET})
        set(MUSCRIPTOR_TARGET_FLAGS)
        set(MUSCRIPTOR_TARGET_DEFINITIONS)

        foreach(MUSCRIPTOR_PROPERTY COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
            get_target_property(MUSCRIPTOR_VALUE ${MUSCRIPTOR_TARGET} ${MUSCRIPTOR_PROPERTY})

            if(MUSCRIPTOR_VALUE)
                list(APPEND MUSCRIPTOR_TARGET_FLAGS ${MUSCRIPTOR_VALUE})
            endif()
        endforeach()

        foreach(MUSCRIPTOR_PROPERTY COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
            get_target_property(MUSCRIPTOR_VALUE ${MUSCRIPTOR_TARGET} ${MUSCRIPTOR_PROPERTY})

            if(MUSCRIPTOR_VALUE)
                list(APPEND MUSCRIPTOR_TARGET_DEFINITIONS ${MUSCRIPTOR_VALUE})
            endif()
        endforeach()

        muscriptor_check_no_isa("Target ${MUSCRIPTOR_TARGET}" "${MUSCRIPTOR_TARGET_FLAGS}" "${MUSCRIPTOR_TARGET_DEFINITIONS}")
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
            $<TARGET_PROPERTY:ggml-cpu,COMPILE_DEFINITIONS>
            ${MUSCRIPTOR_CPU_DEFINITIONS_${MUSCRIPTOR_TAG}})
    target_compile_options(${MUSCRIPTOR_TARGET}
        PRIVATE
            $<TARGET_PROPERTY:ggml-cpu,COMPILE_OPTIONS>
            ${MUSCRIPTOR_CPU_FLAGS_${MUSCRIPTOR_TAG}}
            ${MUSCRIPTOR_CPU_NO_LTO})
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
            -DLEVEL=sse42
            "-DINPUTS=${MUSCRIPTOR_BASELINE_INPUTS}"
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/ggml-cpu-variants
            -P ${MUSCRIPTOR_CPU_TOOL}
    DEPENDS muscriptor_ggml muscriptor_ggml_cpu
    COMMENT "Checking the code outside the ggml-cpu variants for instructions above SSE4.2"
    VERBATIM)
