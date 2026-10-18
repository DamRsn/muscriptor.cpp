#[[
Build-time half of the CPU variants (ggml_cpu_variants.cmake). Run with
`cmake -P`; LLVM_NM, LLVM_OBJCOPY, LLVM_OBJDUMP and LLVM_AR name the tools.

ACTION=rename  Checks the static library INPUT, then writes it to OUTPUT with
               every symbol it defines renamed to carry TAG, except constants
               shared by name. MANGLING is `msvc` or `itanium`. Fails if
               INPUT holds LTO bitcode, a static initialiser, or an
               instruction above TAG's level.
ACTION=check   Fails if any of the static libraries in INPUTS (separated by
               `|`) holds an instruction above level LEVEL. Disassembles
               into WORK_DIR.
]]
cmake_minimum_required(VERSION 3.24)

# An instruction is the line's first tab-separated field after the address.
# AT&T syntax, which llvm-objdump prints for every object format.
set(AVX512 "%zmm|%k[0-7]")
set(BMI_LZCNT_MOVBE "\t(andn|bextr|blsi|blsmsk|blsr|bzhi|mulx|pdep|pext|rorx|sarx|shlx|shrx|lzcnt|movbe)[bwlq]?(\t|$)")
set(AVX2_ONLY "\t(vpbroadcast[bwdq]|vbroadcasti128|vperm2i128|vpermd|vpermq|vpermps|vpermpd|vinserti128|vextracti128|vp?gather[a-z0-9]*|vpmaskmov[dq]|vpsllv[dq]|vpsrlv[dq]|vpsravd|vpblendd)(\t|$)")
set(AVX2_REGISTER_BROADCAST "\tvbroadcasts[sd]\t%xmm")
set(AVX2_YMM_INTEGER "\tvp[a-z0-9]*\t[^#]*%ymm")
set(FMA "\tvf(n)?m(add|sub|addsub|subadd)[a-z0-9]*\t")
set(VEX "\tv[a-z0-9]+(\t|$)")

#[[
Fails if `inLibrary` holds an instruction above `inLevel`: sse42, ivybridge
(AVX and F16C) or haswell (AVX2, FMA, BMI1, BMI2, LZCNT and MOVBE). TZCNT is
allowed everywhere: older CPUs run it as BSF, which compilers rely on.
]]
function(check_instructions inLibrary inLevel)
    get_filename_component(name ${inLibrary} NAME)
    set(listing ${WORK_DIR}/${name}.${inLevel}.s)

    execute_process(
        COMMAND ${LLVM_OBJDUMP} --disassemble --no-show-raw-insn --no-leading-addr ${inLibrary}
        OUTPUT_FILE ${listing}
        RESULT_VARIABLE result)

    if(NOT result EQUAL 0)
        message(FATAL_ERROR "llvm-objdump could not disassemble ${inLibrary}")
    endif()

    if(inLevel STREQUAL "sse42")
        set(forbidden "${AVX512}|${VEX}|${BMI_LZCNT_MOVBE}")
    elseif(inLevel STREQUAL "ivybridge")
        set(forbidden "${AVX512}|${FMA}|${BMI_LZCNT_MOVBE}|${AVX2_ONLY}|${AVX2_REGISTER_BROADCAST}|${AVX2_YMM_INTEGER}")
    elseif(inLevel STREQUAL "haswell")
        set(forbidden "${AVX512}")
    else()
        message(FATAL_ERROR "unknown instruction-set level '${inLevel}'")
    endif()

    file(STRINGS ${listing} hits REGEX "${forbidden}")

    # Matched by the patterns above, but in the level below: VERR and VERW are
    # not VEX, and these take a ymm register in AVX itself.
    list(FILTER hits EXCLUDE REGEX "\t(verr|verw)(\t|$)")

    if(inLevel STREQUAL "ivybridge")
        list(FILTER hits EXCLUDE REGEX "\t(vpermilps|vpermilpd|vperm2f128|vptest)\t")
    endif()

    if(hits)
        list(LENGTH hits count)
        list(SUBLIST hits 0 10 shown)
        list(JOIN shown "\n" shown)
        message(FATAL_ERROR
            "${inLibrary} has ${count} instructions above the ${inLevel} level, e.g.\n${shown}\n"
            "The listing is ${listing}.")
    endif()

    file(REMOVE ${listing})
endfunction()

#[[
Fails unless every member of `inLibrary` is a native object file without a
static initialiser. Either would run code before the dispatcher has checked
the CPU, or let the linker compile code for a level the CPU may lack.
]]
function(check_objects inLibrary)
    execute_process(COMMAND ${LLVM_AR} t ${inLibrary} OUTPUT_VARIABLE members RESULT_VARIABLE result)

    if(NOT result EQUAL 0)
        message(FATAL_ERROR "llvm-ar could not list ${inLibrary}")
    endif()

    string(REGEX MATCHALL "[^\n]+" members "${members}")
    list(LENGTH members member_count)

    execute_process(COMMAND ${LLVM_OBJDUMP} --section-headers ${inLibrary}
                    OUTPUT_VARIABLE headers RESULT_VARIABLE result)

    if(NOT result EQUAL 0)
        message(FATAL_ERROR "llvm-objdump could not read ${inLibrary}")
    endif()

    # llvm-objdump passes over a bitcode member without a word.
    string(REGEX MATCHALL "file format [^\n]+" formats "${headers}")
    list(LENGTH formats object_count)

    if(NOT object_count EQUAL member_count)
        message(FATAL_ERROR
            "${inLibrary} has ${member_count} members but only ${object_count} native object files. The "
            "others are likely LTO bitcode, which the variants must not be built as.")
    endif()

    string(REGEX MATCHALL "[^\n]+" lines "${headers}")
    set(member "")

    foreach(line IN LISTS lines)
        if(line MATCHES "^(.*):\tfile format")
            set(member ${CMAKE_MATCH_1})
        elseif(line MATCHES " (\\.CRT\\$X[CIL][^ ]*|\\.init_array[^ ]*|\\.ctors[^ ]*|__mod_init_func|__init_offsets) ")
            message(FATAL_ERROR
                "${member} has a static initialiser (section ${CMAKE_MATCH_1}). It would run before the "
                "dispatcher has checked the CPU.")
        endif()
    endforeach()
endfunction()

#[[
The symbols `inLibrary` defines and exports, each once.
]]
function(defined_symbols inLibrary outSymbols)
    # Its stderr names each member without symbols, such as ggml's AMX sources
    # on a build without AMX.
    execute_process(
        COMMAND ${LLVM_NM} --defined-only --extern-only --just-symbol-name ${inLibrary}
        OUTPUT_VARIABLE listing ERROR_VARIABLE errors RESULT_VARIABLE result)

    if(NOT result EQUAL 0)
        message(FATAL_ERROR "llvm-nm could not read ${inLibrary}:\n${errors}")
    endif()

    # Each member's symbols follow a `member:` line.
    string(REGEX MATCHALL "[^\n]+" symbols "${listing}")
    list(FILTER symbols EXCLUDE REGEX ":$")
    list(REMOVE_DUPLICATES symbols)
    set(${outSymbols} ${symbols} PARENT_SCOPE)
endfunction()

#[[
True for a symbol whose name spells out its contents, so that every copy is
the same constant: MSVC's string literals and floating-point and vector
constants. Those stay shared. Everything else is renamed, RTTI included: a
ggml-cpu class with a key function has a strong typeinfo in every variant.
]]
function(is_shared inSymbol outShared)
    if(MANGLING STREQUAL "msvc" AND inSymbol MATCHES "^(\\?\\?_C@|__real@|__xmm@|__ymm@|__zmm@)")
        set(${outShared} TRUE PARENT_SCOPE)
    else()
        set(${outShared} FALSE PARENT_SCOPE)
    endif()
endfunction()

#[[
`inSymbol` with TAG in it. Itanium names get a `.tag` clone suffix, which
demanglers print. MSVC names get `__tag` before their first `@`, the end of
the function or class name they start with, so they still demangle. C names
end with `__tag`.
]]
function(renamed_symbol inSymbol outSymbol)
    string(FIND "${inSymbol}" "@" at)

    if(MANGLING STREQUAL "itanium" AND inSymbol MATCHES "^_?_Z")
        set(${outSymbol} "${inSymbol}.${TAG}" PARENT_SCOPE)
    elseif(MANGLING STREQUAL "msvc" AND at GREATER 0)
        string(SUBSTRING "${inSymbol}" 0 ${at} head)
        string(SUBSTRING "${inSymbol}" ${at} -1 tail)
        set(${outSymbol} "${head}__${TAG}${tail}" PARENT_SCOPE)
    else()
        set(${outSymbol} "${inSymbol}__${TAG}" PARENT_SCOPE)
    endif()
endfunction()

foreach(tool LLVM_NM LLVM_OBJCOPY LLVM_OBJDUMP LLVM_AR)
    if(NOT ${tool})
        message(FATAL_ERROR "${tool} is not set")
    endif()
endforeach()

if(ACTION STREQUAL "rename")
    get_filename_component(WORK_DIR ${OUTPUT} DIRECTORY)
    file(MAKE_DIRECTORY ${WORK_DIR})

    check_objects(${INPUT})
    check_instructions(${INPUT} ${TAG})

    defined_symbols(${INPUT} symbols)
    set(map "")

    foreach(symbol IN LISTS symbols)
        is_shared(${symbol} shared)

        if(NOT shared)
            renamed_symbol(${symbol} new_name)
            string(APPEND map "${symbol} ${new_name}\n")
        endif()
    endforeach()

    set(map_file ${OUTPUT}.symbols)
    file(WRITE ${map_file} "${map}")
    file(REMOVE ${OUTPUT})

    execute_process(COMMAND ${LLVM_OBJCOPY} --redefine-syms=${map_file} ${INPUT} ${OUTPUT} RESULT_VARIABLE result)

    if(NOT result EQUAL 0)
        message(FATAL_ERROR "llvm-objcopy could not rename the symbols of ${INPUT}")
    endif()

    # Every definition left with its own name would be shared with, or taken
    # over by, the rest of the program.
    defined_symbols(${OUTPUT} symbols)

    foreach(symbol IN LISTS symbols)
        is_shared(${symbol} shared)
        string(FIND "${symbol}" "__${TAG}" infix)
        string(FIND "${symbol}" ".${TAG}" suffix)

        if(NOT shared AND infix EQUAL -1 AND suffix EQUAL -1)
            file(REMOVE ${OUTPUT})
            message(FATAL_ERROR "${OUTPUT} still defines '${symbol}' under its own name")
        endif()
    endforeach()
elseif(ACTION STREQUAL "check")
    file(MAKE_DIRECTORY ${WORK_DIR})
    string(REPLACE "|" ";" libraries "${INPUTS}")

    foreach(library IN LISTS libraries)
        check_instructions(${library} ${LEVEL})
    endforeach()
else()
    message(FATAL_ERROR "ACTION must be rename or check, got '${ACTION}'")
endif()
