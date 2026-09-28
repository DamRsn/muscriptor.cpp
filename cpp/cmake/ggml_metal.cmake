# ggml's precompiled Metal library: the fix its build needs, and the helper that
# puts the result where ggml looks for it at run time.

# ggml builds ggml-tensor.metallib with its common Metal flags plus
# -mtargetos=macos26.0, and the Metal compiler rejects that next to the
# -mmacosx-version-min the common flags carry once a macOS floor is set.
set(MUSCRIPTOR_GGML_TENSOR_FLAGS_ORIGINAL
    "set(XC_FLAGS_TENSOR \${XC_FLAGS} -mtargetos=\${GGML_METAL_TARGET_OS}26.0\${METAL_TARGET_SIM})")
set(MUSCRIPTOR_GGML_TENSOR_FLAGS_FIXED
    "set(XC_FLAGS_TENSOR \${XC_FLAGS})
        list(FILTER XC_FLAGS_TENSOR EXCLUDE REGEX \"^-mmacosx-version-min=\")
        list(APPEND XC_FLAGS_TENSOR -mtargetos=\${GGML_METAL_TARGET_OS}26.0\${METAL_TARGET_SIM})")

#[[
Applies the ggml-tensor.metallib flag fix to a fetched ggml, or reverts it, so
the source matches the requested build on every configure. A checkout given
through FETCHCONTENT_SOURCE_DIR_GGML is never written to: it is only checked.

inSourceDir  ggml's source directory.
inFixed      Whether the fix should be present.
inWritable   Whether this function may edit the source.
]]
function(muscriptor_fix_ggml_metal_tensor_flags inSourceDir inFixed inWritable)
    set(metal_cmake "${inSourceDir}/src/ggml-metal/CMakeLists.txt")
    file(READ "${metal_cmake}" contents)

    string(FIND "${contents}" "${MUSCRIPTOR_GGML_TENSOR_FLAGS_ORIGINAL}" original_at)
    string(FIND "${contents}" "${MUSCRIPTOR_GGML_TENSOR_FLAGS_FIXED}" fixed_at)

    if(original_at EQUAL -1 AND fixed_at EQUAL -1)
        message(FATAL_ERROR
            "${metal_cmake} no longer contains the XC_FLAGS_TENSOR line cpp/cmake/ggml_metal.cmake "
            "rewrites. Check whether this ggml still needs the fix, and update or drop it.")
    endif()

    if(inFixed AND original_at EQUAL -1)
        return()
    endif()

    if(NOT inFixed AND fixed_at EQUAL -1)
        return()
    endif()

    if(NOT inWritable)
        if(inFixed)
            message(FATAL_ERROR
                "The ggml checkout at ${inSourceDir} builds ggml-tensor.metallib with flags the Metal "
                "compiler rejects when a macOS floor is set. Apply the XC_FLAGS_TENSOR fix from "
                "cpp/cmake/ggml_metal.cmake to it, or configure with -DMUSCRIPTOR_METAL_PRECOMPILED=OFF.")
        endif()

        return()
    endif()

    if(inFixed)
        string(REPLACE "${MUSCRIPTOR_GGML_TENSOR_FLAGS_ORIGINAL}" "${MUSCRIPTOR_GGML_TENSOR_FLAGS_FIXED}"
               contents "${contents}")
    else()
        string(REPLACE "${MUSCRIPTOR_GGML_TENSOR_FLAGS_FIXED}" "${MUSCRIPTOR_GGML_TENSOR_FLAGS_ORIGINAL}"
               contents "${contents}")
    endif()

    file(WRITE "${metal_cmake}" "${contents}")
endfunction()

#[[
Makes ggml's precompiled Metal libraries available to an executable at run
time. ggml looks for default.metallib in the Resources of the bundle holding
its code, then next to the running executable: a bundle target gets the files
in Contents/Resources, any other executable gets them copied beside it. Does
nothing unless MUSCRIPTOR_METAL_PRECOMPILED is on.

A missing default.metallib is not an error at run time: Metal fails to
initialise and Model::load falls back to the CPU.

inTarget  An executable, app bundle or plugin bundle target.
]]
function(muscriptor_add_metal_library inTarget)
    get_property(libraries GLOBAL PROPERTY MUSCRIPTOR_METAL_LIBRARIES)

    if(NOT libraries)
        return()
    endif()

    add_dependencies(${inTarget} ggml-metal-lib)

    get_target_property(is_app ${inTarget} MACOSX_BUNDLE)
    get_target_property(is_bundle ${inTarget} BUNDLE)

    if(is_app OR is_bundle)
        target_sources(${inTarget} PRIVATE ${libraries})
        set_source_files_properties(${libraries}
            TARGET_DIRECTORY ${inTarget}
            PROPERTIES GENERATED TRUE MACOSX_PACKAGE_LOCATION Resources)
    else()
        add_custom_command(TARGET ${inTarget} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${libraries} $<TARGET_FILE_DIR:${inTarget}>
            VERBATIM)
    endif()
endfunction()
